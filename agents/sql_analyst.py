"""
SQL Analyst Agent.

Pipeline: curate question -> gather schema -> check cache (skip straight
to cost/safety/execute on a hit) -> generate SQL -> optimize (performance)
-> judge (LLM-as-judge correctness check, retries generation on failure,
up to MAX_JUDGE_RETRIES; a fresh "correct" verdict gets cached) -> estimate
cost (deterministic, EXPLAIN QUERY PLAN based) -> safety-check (destructive
ops) -> execute -> answer in natural language.
"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import AgentSchema, SQLSafetySchema, QueryOptimizationSchema, SQLJudgeSchema
from utils.database import DatabaseUtil
from utils.llm_pick import invoke_with_resilience
from utils.audit import log_event
from utils.cost_estimator import estimate_query_cost
from utils.query_cache import build_cache_key, get_cached_query, set_cached_query

db = DatabaseUtil()
MAX_JUDGE_RETRIES = 2


def curate_question(state: AgentSchema) -> AgentSchema:
    prompt = (
        "Rewrite the user's question into a single, precise, unambiguous "
        "analytical question suitable for translation into SQL. "
        f"User question: {state.user_question}"
    )
    resp = invoke_with_resilience("low", [HumanMessage(content=prompt)])
    state.curated_ques = resp.content
    return state


def gather_context(state: AgentSchema) -> AgentSchema:
    schema = db.schema_details()
    state.prompt_query_context = (
        f"Database schema:\n{schema}\n\n"
        f"Question: {state.curated_ques}\n\n"
        "Write a single SQLite SELECT query that answers this question. "
        "Only use tables/columns that exist in the schema above. "
        "Return ONLY the SQL query, no explanation, no markdown fences."
    )
    return state


def check_cache(state: AgentSchema) -> AgentSchema:
    """
    A judge-approved query for this exact (curated question, schema) pair
    means generate_sql -> optimize_query -> judge_query - three LLM calls -
    can be skipped entirely. Execution still always runs fresh below, so
    the answer reflects current data even when the SQL is reused. Only
    entries the judge previously marked "correct" ever get cached (see
    estimate_cost), so a cache hit can't replay a bad query.
    """
    schema = db.schema_details()
    cache_key = build_cache_key(state.curated_ques, schema)
    entry = get_cached_query(cache_key)

    if entry:
        state.generated_sql_query = entry["sql"]
        state.optimizer_notes = entry["optimizer_notes"] + " (served from cache)"
        state.judge_verdict = "correct"
        state.from_cache = True
        log_event("sql_cache_hit", question=state.user_question, sql=entry["sql"])
    else:
        state.from_cache = False

    return state


def cache_decision(state: AgentSchema) -> str:
    return "hit" if state.from_cache else "miss"


def generate_sql(state: AgentSchema) -> AgentSchema:
    prompt = state.prompt_query_context
    if state.judge_feedback:
        prompt += (
            f"\n\nA previous attempt was judged incorrect. Feedback:\n{state.judge_feedback}\n"
            f"Previous query:\n{state.generated_sql_query}\n\n"
            "Fix the query based on this feedback. Return ONLY the corrected SQL."
        )
    resp = invoke_with_resilience("high", [HumanMessage(content=prompt)])
    sql = resp.content.strip()
    # strip accidental markdown fences
    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql.replace("sql\n", "", 1) if sql.startswith("sql\n") else sql
    state.generated_sql_query = sql.strip()
    return state


def optimize_query(state: AgentSchema) -> AgentSchema:
    """Review the generated SQL for performance issues before it ever runs."""
    schema = db.schema_details()
    prompt = (
        f"Database schema:\n{schema}\n\n"
        f"Proposed query:\n{state.generated_sql_query}\n\n"
        "Review this SQLite query for performance issues: missing LIMIT on "
        "unbounded scans, SELECT * where specific columns would do, missing "
        "WHERE/indexable filters, unnecessary subqueries. If it's already fine, "
        "return it unchanged with needs_rewrite=false."
    )
    result = invoke_with_resilience("low", [HumanMessage(content=prompt)], structured_schema=QueryOptimizationSchema)
    if result.needs_rewrite:
        state.optimizer_notes = result.reasoning
        state.generated_sql_query = result.optimized_query.strip()
    else:
        state.optimizer_notes = "No changes needed."
    return state


def judge_query(state: AgentSchema) -> AgentSchema:
    """
    LLM-as-judge: independently checks whether the generated SQL actually
    answers the user's question correctly - separate from safety (destructive
    ops) and the optimizer (performance). Catches wrong joins, wrong
    aggregations, wrong filters, wrong columns, etc. before execution.
    """
    schema = db.schema_details()
    prompt = (
        f"Database schema:\n{schema}\n\n"
        f"Original question: {state.user_question}\n"
        f"Curated question: {state.curated_ques}\n\n"
        f"Generated SQL:\n{state.generated_sql_query}\n\n"
        "Act as an independent reviewer, not the query's author. Check step by "
        "step whether this query would actually return the correct data to "
        "answer the question: right tables, right join conditions, right "
        "aggregation/grouping, right filters, right date ranges, right sort "
        "order. Do not check for performance or safety - only correctness."
    )
    result = invoke_with_resilience("high", [HumanMessage(content=prompt)], structured_schema=SQLJudgeSchema)
    state.judge_verdict = result.verdict
    state.judge_feedback = result.feedback if result.verdict == "incorrect" else ""

    log_event(
        "sql_judge",
        question=state.user_question,
        sql=state.generated_sql_query,
        verdict=result.verdict,
        confidence=result.confidence,
        feedback=result.feedback,
        retry_count=state.judge_retries,
    )

    if result.verdict == "incorrect":
        state.judge_retries += 1
    return state


def judge_decision(state: AgentSchema) -> str:
    if state.judge_verdict == "incorrect" and state.judge_retries <= MAX_JUDGE_RETRIES:
        return "retry"
    return "proceed"


def estimate_cost(state: AgentSchema) -> AgentSchema:
    """
    Deterministic cost check (no LLM) on the judge-approved query: runs
    SQLite's own EXPLAIN QUERY PLAN and flags full table scans on large
    tables. Doesn't block execution - this is a read-mostly analytics
    agent, not a shared production DB - but surfaces the warning to the
    user and the audit log instead of silently running an expensive query.

    Also where a freshly judge-approved query gets written to the cache
    (skipped if this run was already served from cache, to avoid
    re-writing the identical entry on every repeat question).
    """
    result = estimate_query_cost(db.db_path, state.generated_sql_query)
    state.cost_level = result["cost_level"]
    state.cost_notes = result["notes"]

    if not state.from_cache and state.judge_verdict == "correct":
        schema = db.schema_details()
        cache_key = build_cache_key(state.curated_ques, schema)
        set_cached_query(cache_key, state.generated_sql_query, state.optimizer_notes)

    log_event(
        "sql_cost_estimate",
        question=state.user_question,
        sql=state.generated_sql_query,
        cost_level=state.cost_level,
        notes=state.cost_notes,
        plan=result["plan"],
    )
    return state


def safety_check(state: AgentSchema) -> AgentSchema:
    is_safe = db.is_query_safe(state.generated_sql_query)
    state.is_safe = "Yes" if is_safe else "No"
    state.comments = (
        "Read-only SELECT, approved." if is_safe
        else "Query contains destructive or non-SELECT operations; blocked."
    )
    return state


def execute_query(state: AgentSchema) -> AgentSchema:
    if state.is_safe != "Yes":
        state.sql_query_execution_result = f"BLOCKED: {state.comments}"
        return state
    try:
        rows = db.run_query(state.generated_sql_query)
        state.sql_query_execution_result = str(rows)
    except Exception as e:
        state.sql_query_execution_result = f"ERROR: {e}"

    log_event(
        "sql_run",
        question=state.user_question,
        sql=state.generated_sql_query,
        optimizer_notes=state.optimizer_notes,
        judge_verdict=state.judge_verdict,
        judge_retries=state.judge_retries,
        cost_level=state.cost_level,
        is_safe=state.is_safe,
        result_preview=state.sql_query_execution_result[:500],
    )
    return state


def generate_answer(state: AgentSchema) -> AgentSchema:
    prompt = (
        f"User asked: {state.user_question}\n"
        f"SQL used: {state.generated_sql_query}\n"
        f"Result: {state.sql_query_execution_result}\n\n"
        "Answer the user's question in plain, concise language based on this result."
    )
    resp = invoke_with_resilience("medium", [HumanMessage(content=prompt)])
    answer = resp.content

    # If the judge never cleared the query after exhausting retries, surface
    # that uncertainty rather than presenting the answer as fully verified.
    if state.judge_verdict == "incorrect":
        answer += (
            f"\n\n⚠️ Note: an automated review flagged possible issues with this "
            f"query after {state.judge_retries} attempt(s): {state.judge_feedback} "
            "Treat this result with caution."
        )

    if state.cost_level == "high":
        answer += f"\n\n💰 Cost note: {state.cost_notes}"
    elif state.cost_level == "medium":
        # Previously only "high" cost surfaced anything to the user, so a
        # moderate full-table scan (cheap here, but worth an index if the
        # table grows) was computed by estimate_cost and audit-logged, but
        # silently dropped from the answer the user actually sees.
        answer += f"\n\nℹ️ Cost note: {state.cost_notes}"

    state.final_answer = answer
    return state


def build_sql_agent():
    graph = StateGraph(AgentSchema)
    graph.add_node("curate_question", curate_question)
    graph.add_node("gather_context", gather_context)
    graph.add_node("check_cache", check_cache)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("optimize_query", optimize_query)
    graph.add_node("judge_query", judge_query)
    graph.add_node("estimate_cost", estimate_cost)
    graph.add_node("safety_check", safety_check)
    graph.add_node("execute_query", execute_query)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("curate_question")
    graph.add_edge("curate_question", "gather_context")
    graph.add_edge("gather_context", "check_cache")
    graph.add_conditional_edges(
        "check_cache",
        cache_decision,
        {"hit": "estimate_cost", "miss": "generate_sql"},
    )
    graph.add_edge("generate_sql", "optimize_query")
    graph.add_edge("optimize_query", "judge_query")
    graph.add_conditional_edges(
        "judge_query",
        judge_decision,
        {"retry": "generate_sql", "proceed": "estimate_cost"},
    )
    graph.add_edge("estimate_cost", "safety_check")
    graph.add_edge("safety_check", "execute_query")
    graph.add_edge("execute_query", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


sql_analyst = build_sql_agent()

if __name__ == "__main__":
    result = sql_analyst.invoke(AgentSchema(user_question="Show me the top 5 users with the highest ratings"))
    print(result["final_answer"] if isinstance(result, dict) else result.final_answer)
