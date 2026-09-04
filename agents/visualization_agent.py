"""
Visualization Agent.

Pipeline: get data (from a CSV path or an already-fetched SQL result) ->
LLM picks chart type + columns (structured output) -> LLM-as-judge reviews
the spec against the request and data shape before anything is rendered ->
render with matplotlib. On an "incorrect" verdict, feedback is fed back
into planning and retried, up to MAX_JUDGE_RETRIES.
"""
import ast
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import VizAgentSchema, ChartSpecSchema, VizJudgeSchema
from utils.llm_pick import invoke_with_resilience
from utils.viz_tools import render_chart
from utils.audit import log_event

MAX_JUDGE_RETRIES = 2


def load_data(state: VizAgentSchema) -> pd.DataFrame:
    """data_source is either a CSV path or a stringified list-of-dicts (SQL result)."""
    src = state.data_source.strip()
    if src.endswith(".csv"):
        return pd.read_csv(src)
    # Otherwise treat it as a Python literal (e.g. "[{'a': 1, 'b': 2}, ...]")
    rows = ast.literal_eval(src)
    return pd.DataFrame(rows)


def plan_chart(state: VizAgentSchema) -> VizAgentSchema:
    df = load_data(state)
    columns = list(df.columns)
    prompt = (
        f"User request: {state.user_request}\n"
        f"Available columns: {columns}\n"
        f"Sample rows: {df.head(3).to_dict(orient='records')}\n\n"
        "Pick the best chart type and columns to visualize this data for the user's request."
    )
    if state.judge_feedback:
        prompt += (
            f"\n\nA previous attempt was judged incorrect. Feedback:\n{state.judge_feedback}\n"
            f"Previous spec: chart_type={state.chart_type}, x={state.x_column}, "
            f"y={state.y_column}, title={state.chart_title}\n"
            "Pick a better spec based on this feedback."
        )
    spec = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=ChartSpecSchema)

    state.chart_type = spec.chart_type
    state.x_column = spec.x_column
    state.y_column = spec.y_column or columns[-1]
    state.chart_title = spec.title
    return state


def judge_chart_spec(state: VizAgentSchema) -> VizAgentSchema:
    """
    LLM-as-judge: reviews the chosen chart type/columns against the request
    and the data's shape BEFORE anything is rendered - catches a poor chart
    type (e.g. pie chart with too many slices), wrong columns, or a spec
    that doesn't match what the user actually asked for.
    """
    df = load_data(state)
    prompt = (
        f"User request: {state.user_request}\n"
        f"Available columns: {list(df.columns)}\n"
        f"Column dtypes: {df.dtypes.astype(str).to_dict()}\n"
        f"Row count: {len(df)}\n\n"
        f"Proposed chart spec: type={state.chart_type}, x_column={state.x_column}, "
        f"y_column={state.y_column}, title={state.chart_title}\n\n"
        "Act as an independent reviewer. Check whether this chart type and column "
        "choice actually fits the request and the data's shape (e.g. flag a pie "
        "chart with too many categories, a line chart on non-sequential data, or "
        "columns that don't match what was asked for)."
    )
    result = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=VizJudgeSchema)
    state.judge_verdict = result.verdict
    state.judge_feedback = result.feedback if result.verdict == "incorrect" else ""

    log_event(
        "viz_judge",
        request=state.user_request,
        chart_type=state.chart_type,
        x_column=state.x_column,
        y_column=state.y_column,
        verdict=result.verdict,
        confidence=result.confidence,
        feedback=result.feedback,
        retry_count=state.judge_retries,
    )

    if result.verdict == "incorrect":
        state.judge_retries += 1
    return state


def judge_decision(state: VizAgentSchema) -> str:
    if state.judge_verdict == "incorrect" and state.judge_retries <= MAX_JUDGE_RETRIES:
        return "retry"
    return "render"


def render_node(state: VizAgentSchema) -> VizAgentSchema:
    df = load_data(state)
    chart_path = render_chart(df, state.chart_type, state.x_column, state.y_column, state.chart_title)
    state.chart_path = chart_path
    answer = f"Created a {state.chart_type} chart titled '{state.chart_title}' -> {chart_path}"
    if state.judge_verdict == "incorrect":
        answer += (
            f"\n\n⚠️ Note: automated review still flagged an issue after "
            f"{state.judge_retries} attempt(s): {state.judge_feedback} "
            "Treat this chart with caution."
        )
    state.final_answer = answer
    return state


def build_visualization_agent():
    graph = StateGraph(VizAgentSchema)
    graph.add_node("plan_chart", plan_chart)
    graph.add_node("judge_chart_spec", judge_chart_spec)
    graph.add_node("render", render_node)

    graph.set_entry_point("plan_chart")
    graph.add_edge("plan_chart", "judge_chart_spec")
    graph.add_conditional_edges(
        "judge_chart_spec",
        judge_decision,
        {"retry": "plan_chart", "render": "render"},
    )
    graph.add_edge("render", END)
    return graph.compile()


visualization_agent = build_visualization_agent()

if __name__ == "__main__":
    result = visualization_agent.invoke(
        VizAgentSchema(
            user_request="Show average rating per vehicle type as a bar chart",
            data_source="data/vehicles.csv",
        )
    )
    print(result["final_answer"] if isinstance(result, dict) else result.final_answer)
