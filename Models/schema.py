"""
Pydantic schemas that define the state passed between LangGraph nodes.
Each agent (router, sql, etl, visualization) has its own state shape.
"""
from typing import List, Literal, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class RouterSchema(BaseModel):
    """Structured output the router LLM must produce."""
    answer: Literal["sql", "etl", "visualization", "catalog", "clarify"] = Field(
        description="Which sub-agent should handle this request."
    )
    comments: str = Field(description="Short reasoning for the routing decision.")


class DataAgentSchema(BaseModel):
    """Top-level graph state for the main router agent."""
    messages: List[Any] = Field(default_factory=list)
    conversation_history: List[Any] = Field(
        default_factory=list, description="Prior turns (user + assistant) for follow-up context."
    )
    route_response: str = ""
    final_answer: str = ""
    needs_clarification: bool = False


# ---------------------------------------------------------------------------
# SQL Analyst
# ---------------------------------------------------------------------------
class AgentSchema(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    user_question: str = ""
    curated_ques: str = ""
    prompt_query_context: str = ""
    generated_sql_query: str = ""
    is_safe: Literal["Yes", "No", ""] = ""
    comments: str = ""
    sql_query_execution_result: str = ""
    optimizer_notes: str = ""
    judge_verdict: Literal["correct", "incorrect", ""] = ""
    judge_feedback: str = ""
    judge_retries: int = 0
    cost_level: Literal["low", "medium", "high", ""] = ""
    cost_notes: str = ""
    from_cache: bool = False
    final_answer: str = ""


class SQLSafetySchema(BaseModel):
    is_safe: Literal["Yes", "No"] = Field(
        description="Yes if the query is a read-only SELECT with no destructive operations."
    )
    comments: str = Field(description="Explanation of the safety verdict.")


class QueryOptimizationSchema(BaseModel):
    needs_rewrite: bool = Field(
        description="True if the query has a performance issue worth fixing before running."
    )
    optimized_query: str = Field(
        description="The improved query. If no rewrite is needed, return the original query unchanged."
    )
    reasoning: str = Field(description="Brief explanation: e.g. missing WHERE clause, unbounded scan, unnecessary SELECT *, no LIMIT, etc.")


class SQLJudgeSchema(BaseModel):
    """LLM-as-judge verdict on whether the generated SQL actually answers the question."""
    verdict: Literal["correct", "incorrect"] = Field(
        description="'correct' if the query would return the right data to answer the question; "
        "'incorrect' if it has a logical flaw (wrong table/column, wrong join, wrong aggregation, "
        "wrong filter, off-by-one in date ranges, groups by the wrong column, etc.)."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident the judge is in this verdict."
    )
    feedback: str = Field(
        description="If incorrect: specific, actionable feedback on what's wrong and how to fix it. "
        "If correct: brief confirmation of why the query matches the question."
    )


# ---------------------------------------------------------------------------
# ETL Analyst
# ---------------------------------------------------------------------------
class ETLAgentSchema(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    tool_result: str = ""
    original_request: str = ""
    judge_verdict: Literal["correct", "incorrect", ""] = ""
    judge_feedback: str = ""
    judge_retries: int = 0
    data_quality_report: str = ""
    dq_severity: Literal["ok", "warning", "critical", ""] = ""
    dq_retries: int = 0
    source_quality_report: str = ""
    source_dq_severity: Literal["ok", "warning", "critical", ""] = ""
    source_dq_retries: int = 0


class ETLJudgeSchema(BaseModel):
    """LLM-as-judge verdict on whether the ETL run actually satisfied the request."""
    verdict: Literal["correct", "incorrect"] = Field(
        description="'correct' if the tool calls made (extract/transform) actually fulfill what the "
        "user asked for - right source, right output format, right transformation logic. "
        "'incorrect' if a tool errored, the wrong data was pulled, the transform logic doesn't "
        "match the request, or a step the user asked for was skipped."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident the judge is in this verdict."
    )
    feedback: str = Field(
        description="If incorrect: specific, actionable feedback on what to redo or fix. "
        "If correct: brief confirmation of why the run satisfies the request."
    )


# ---------------------------------------------------------------------------
# Visualization Agent
# ---------------------------------------------------------------------------
class VizAgentSchema(BaseModel):
    messages: List[Any] = Field(default_factory=list)
    user_request: str = ""
    data_source: str = ""          # path to CSV or a SQL result reference
    chart_type: Literal["bar", "line", "scatter", "pie", ""] = ""
    x_column: str = ""
    y_column: str = ""
    chart_title: str = ""
    chart_path: str = ""
    judge_verdict: Literal["correct", "incorrect", ""] = ""
    judge_feedback: str = ""
    judge_retries: int = 0
    final_answer: str = ""


class ChartSpecSchema(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "pie"] = Field(
        description="Best chart type for this request."
    )
    x_column: str = Field(description="Column to use for the x-axis / categories.")
    y_column: Optional[str] = Field(
        default=None, description="Column to use for the y-axis / values, if applicable."
    )
    title: str = Field(description="Chart title.")


class VizJudgeSchema(BaseModel):
    """LLM-as-judge verdict on whether the chart spec matches the request and data shape."""
    verdict: Literal["correct", "incorrect"] = Field(
        description="'correct' if the chosen chart type and columns would produce a chart that "
        "matches the user's request and suits the data's shape. 'incorrect' if the chart type is "
        "a poor fit (e.g. pie chart with too many slices, line chart on non-sequential/non-numeric "
        "x-axis), the wrong columns were picked, or the request asked for something different."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident the judge is in this verdict."
    )
    feedback: str = Field(
        description="If incorrect: specific, actionable feedback on what to change. "
        "If correct: brief confirmation of why this chart spec fits."
    )


# ---------------------------------------------------------------------------
# Data Catalog Agent
# ---------------------------------------------------------------------------
class DataCatalogSchema(BaseModel):
    """State for the agent that maintains column descriptions across tables."""
    tables_filter: List[str] = Field(
        default_factory=list, description="Optional subset of tables to catalog; empty means all tables."
    )
    refresh: bool = Field(
        default=False, description="If True, regenerate descriptions even for already-cataloged columns."
    )
    schema_text: str = ""
    updated_columns: int = 0
    catalog_report: str = ""
    final_answer: str = ""


class ColumnDescriptionSchema(BaseModel):
    table: str = Field(description="Table this column belongs to.")
    column: str = Field(description="Column name.")
    description: str = Field(
        description="Concise, human-readable description of what this column represents, "
        "inferred from its name, declared type, and sample values."
    )


class ColumnDescriptionListSchema(BaseModel):
    descriptions: List[ColumnDescriptionSchema] = Field(default_factory=list)

