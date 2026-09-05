"""
ETL Analyst Agent.

A tool-calling loop: bind extract/transform tools to the LLM, let it
decide which tool(s) to call, execute them, and report back. Before a
transform_load_tool call is allowed to run, check_source_quality inspects
the SOURCE file it's about to load - the same deterministic checks used
on output, applied up front so an already-bad source doesn't waste an
LLM-driven transform attempt before anyone notices. Once the loop
finishes (no more tool calls):
  1. check_data_quality (deterministic, pandas-based) inspects whatever
     file the tools produced - nulls, duplicates, outliers, empty output.
     A "critical" finding (e.g. zero rows) triggers a retry with feedback,
     the same way the judge does.
  2. judge_run (LLM-as-judge) reviews the full transcript - including the
     quality report - against the original request.
All three retry loops are capped (MAX_JUDGE_RETRIES / MAX_DQ_RETRIES /
MAX_SOURCE_DQ_RETRIES) so a stubbornly bad source can't loop forever; it
surfaces a caveat instead.
"""
import os
import re
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from Models.schema import ETLAgentSchema, ETLJudgeSchema
from utils.etl_tools import ETL_TOOLS
from utils.llm_pick import invoke_with_resilience
from utils.audit import log_event
from utils.data_quality import run_quality_checks

SYSTEM_PROMPT = (
    "You are an ETL assistant. You have two tools: extract_load_tool "
    "(pull JSON from an API and save it) and transform_load_tool "
    "(load a file into a pandas DataFrame `df`, run pandas code that "
    "reassigns df, and save the result). Choose the right tool(s) based "
    "on the user's request. When writing pandas_code for transform_load_tool, "
    "only use `df` and `pd` - no imports, no file I/O inside the code."
)

MAX_JUDGE_RETRIES = 2
MAX_DQ_RETRIES = 1
MAX_SOURCE_DQ_RETRIES = 1
OUTPUT_PATH_PATTERN = re.compile(r"-> (\S+\.(?:csv|json|parquet))")

# Previously `llm = pick_llm("high").bind_tools(ETL_TOOLS)` ran here at
# MODULE IMPORT TIME - meaning this whole module (and anything that
# imports it, including agents/data_agent.py's router) would fail to
# import without API access configured, even for code that never touches
# the ETL agent. invoke_with_resilience() below calls pick_llm() lazily,
# per-invocation, which also fixes that.
tool_node = ToolNode(ETL_TOOLS)


def agent_node(state: ETLAgentSchema) -> ETLAgentSchema:
    messages = state.messages
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    if not state.original_request:
        first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        state.original_request = first_human.content if first_human else ""
    resp = invoke_with_resilience("high", messages, tools=ETL_TOOLS)
    state.messages = state.messages + [resp]
    return state


def should_continue(state: ETLAgentSchema) -> str:
    last = state.messages[-1]
    if getattr(last, "tool_calls", None):
        return "check_source_quality"
    return "data_quality"


def _pending_transform_input_path(state: ETLAgentSchema) -> str:
    """If the next tool call is transform_load_tool, return its input_path
    (if that file already exists locally); otherwise empty string. Extract
    calls have nothing to pre-check - there's no local file yet."""
    last = state.messages[-1]
    for call in getattr(last, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if name == "transform_load_tool":
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            input_path = args.get("input_path", "")
            if input_path and os.path.exists(input_path):
                return input_path
    return ""


def check_source_quality(state: ETLAgentSchema) -> ETLAgentSchema:
    """
    Deterministic (no LLM) quality pass on the SOURCE file a pending
    transform_load_tool call is about to load - before the transform runs,
    not just after. Catches an already-bad source (empty file, mostly-null
    columns) up front instead of spending an LLM-driven transform attempt
    on it and only noticing via check_data_quality afterward.
    """
    input_path = _pending_transform_input_path(state)
    if not input_path:
        state.source_dq_severity = "ok"
        return state

    report = run_quality_checks(input_path)
    state.source_quality_report = report["report_text"]
    state.source_dq_severity = report["severity"]

    log_event(
        "source_data_quality",
        request=state.original_request,
        input_path=input_path,
        severity=report["severity"],
        row_count=report["row_count"],
        issues=report["issues"],
        retry_count=state.source_dq_retries,
    )

    if report["severity"] == "critical":
        # Unconditional increment - see the matching comment in
        # check_data_quality for why a conditional increment here would
        # cause source_quality_decision to retry forever once capped.
        state.source_dq_retries += 1
        if state.source_dq_retries <= MAX_SOURCE_DQ_RETRIES:
            state.messages = state.messages + [
                HumanMessage(content=(
                    f"Automated data quality check found a critical issue with the source "
                    f"file {input_path} before any transform ran: {report['report_text']} "
                    "Please address this (e.g. check the source, adjust the extraction) and try again."
                ))
            ]
    return state


def source_quality_decision(state: ETLAgentSchema) -> str:
    if state.source_dq_severity == "critical" and state.source_dq_retries <= MAX_SOURCE_DQ_RETRIES:
        return "retry"
    return "proceed"


def _last_output_path(state: ETLAgentSchema) -> str:
    """Find the most recent file path a tool reported writing to."""
    for m in reversed(state.messages):
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            match = OUTPUT_PATH_PATTERN.search(m.content)
            if match:
                return match.group(1)
    return ""


def check_data_quality(state: ETLAgentSchema) -> ETLAgentSchema:
    """
    Deterministic (no LLM) quality pass on whatever file the tools just
    produced: null rates, duplicate rows, outliers, empty output. Runs
    before the judge so the judge's correctness review has quality
    context, and so a "technically ran without error but produced
    garbage" case gets caught even if it would otherwise look correct.
    """
    output_path = _last_output_path(state)
    if not output_path:
        state.data_quality_report = "No output file detected (nothing to check)."
        state.dq_severity = "ok"
        return state

    report = run_quality_checks(output_path)
    state.data_quality_report = report["report_text"]
    state.dq_severity = report["severity"]

    log_event(
        "data_quality",
        request=state.original_request,
        output_path=output_path,
        severity=report["severity"],
        row_count=report["row_count"],
        issues=report["issues"],
        retry_count=state.dq_retries,
    )

    if report["severity"] == "critical":
        # Unconditional increment (mirrors judge_run's pattern below) - if this
        # were gated on `dq_retries < MAX_DQ_RETRIES`, the counter would cap at
        # MAX and data_quality_decision's `<= MAX` check would then stay true
        # forever once capped, looping indefinitely on a source that never
        # gets fixed. Incrementing unconditionally means the counter keeps
        # climbing past MAX, so the decision below correctly stops retrying.
        state.dq_retries += 1
        if state.dq_retries <= MAX_DQ_RETRIES:
            state.messages = state.messages + [
                HumanMessage(content=(
                    f"Automated data quality check found a critical issue with "
                    f"{output_path}: {report['report_text']} Please address this "
                    "(e.g. check the source, adjust filters) and try again."
                ))
            ]
    return state


def data_quality_decision(state: ETLAgentSchema) -> str:
    if state.dq_severity == "critical" and state.dq_retries <= MAX_DQ_RETRIES:
        return "retry"
    return "judge"


def judge_run(state: ETLAgentSchema) -> ETLAgentSchema:
    """
    LLM-as-judge: reviews the full tool-call transcript (plus the data
    quality report) against the original request. Catches wrong source
    URLs, wrong output format, transform logic that doesn't match what
    was asked, skipped steps, or silent tool errors.
    """
    transcript = "\n".join(
        f"{type(m).__name__}: {getattr(m, 'content', '') or getattr(m, 'tool_calls', '')}"
        for m in state.messages
    )
    prompt = (
        f"Original user request: {state.original_request}\n\n"
        f"Full run transcript (messages + tool calls + tool results):\n{transcript}\n\n"
        f"Automated data quality report on the output: {state.data_quality_report}\n\n"
        "Act as an independent reviewer. Check whether the tool calls made actually "
        "satisfy the request: correct API/source, correct output format, correct "
        "pandas transform logic, no skipped steps, no swallowed tool errors. Factor "
        "the data quality report into your verdict - e.g. a technically-correct "
        "transform that still produced mostly-null output should be 'incorrect'."
    )
    result = invoke_with_resilience("high", [HumanMessage(content=prompt)], structured_schema=ETLJudgeSchema)
    state.judge_verdict = result.verdict
    state.judge_feedback = result.feedback if result.verdict == "incorrect" else ""

    log_event(
        "etl_judge",
        request=state.original_request,
        verdict=result.verdict,
        confidence=result.confidence,
        feedback=result.feedback,
        retry_count=state.judge_retries,
    )

    if result.verdict == "incorrect":
        state.judge_retries += 1
        if state.judge_retries <= MAX_JUDGE_RETRIES:
            # Inject feedback as a new user turn so the agent loop picks it up
            state.messages = state.messages + [
                HumanMessage(content=(
                    f"Automated review found an issue: {result.feedback} "
                    "Please redo the affected step(s)."
                ))
            ]
        else:
            # Out of retries - flag it visibly in the final message instead of looping forever
            state.messages = state.messages + [
                AIMessage(content=(
                    f"⚠️ Note: automated review still flagged an issue after "
                    f"{state.judge_retries} attempt(s): {result.feedback} "
                    "Treat this result with caution."
                ))
            ]
    elif state.dq_severity == "warning":
        # Judge approved correctness, but quality check found something worth
        # surfacing (e.g. moderate null rate) that isn't severe enough to block.
        state.messages = state.messages + [
            AIMessage(content=f"ℹ️ Data quality note: {state.data_quality_report}")
        ]
    return state


def judge_decision(state: ETLAgentSchema) -> str:
    if state.judge_verdict == "incorrect" and state.judge_retries <= MAX_JUDGE_RETRIES:
        return "retry"
    return END


def build_etl_agent():
    graph = StateGraph(ETLAgentSchema)
    graph.add_node("agent", agent_node)
    graph.add_node("check_source_quality", check_source_quality)
    graph.add_node("tools", tool_node)
    graph.add_node("data_quality", check_data_quality)
    graph.add_node("judge", judge_run)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"check_source_quality": "check_source_quality", "data_quality": "data_quality"}
    )
    graph.add_conditional_edges(
        "check_source_quality", source_quality_decision, {"retry": "agent", "proceed": "tools"}
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("data_quality", data_quality_decision, {"retry": "agent", "judge": "judge"})
    graph.add_conditional_edges("judge", judge_decision, {"retry": "agent", END: END})

    return graph.compile()


etl_analyst = build_etl_agent()

if __name__ == "__main__":
    result = etl_analyst.invoke(
        ETLAgentSchema(messages=[HumanMessage(content=(
            "Extract data from https://pokeapi.co/api/v2/pokemon and save it "
            "to data/extract folder in CSV format"
        ))])
    )
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    print(msgs[-1].content)
