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
    answer: Literal[
        "sql",
        "etl",
        "visualization",
        "catalog",
        "quality",
        "lineage",
        "forecast",
        "security",
        "summary",
        "clarify",
    ] = Field(
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
    audience: Literal["general", "executive", "analyst", "operator"] = "general"
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


class QualityFindingSchema(BaseModel):
    table: str = Field(description="Table where the issue was found.")
    field: str = Field(description="Column or field implicated in the issue.")
    issue_type: Literal["missing_values", "duplicate_rows", "outliers", "schema_drift", "numeric_skew"] = Field(
        description="Type of quality problem identified."
    )
    severity: Literal["low", "medium", "high"] = Field(description="Severity of the issue.")
    evidence: str = Field(description="What in the data supports this finding.")
    recommendation: str = Field(description="Action to take to remediate or monitor it.")


class QualityAnalysisSchema(BaseModel):
    summary: str = Field(description="Overall assessment of data quality based on the actual data scanned.")
    findings: List[QualityFindingSchema] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = Field(description="Overall risk level for the dataset.")


class LineageRelationshipSchema(BaseModel):
    source_table: str = Field(description="Upstream table in the relationship.")
    source_field: str = Field(description="Source table field used in the relationship.")
    target_table: str = Field(description="Downstream table in the relationship.")
    target_field: str = Field(description="Target table field used in the relationship.")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in the inferred relationship.")
    reason: str = Field(description="Why this relationship is likely valid.")


class LineageAnalysisSchema(BaseModel):
    summary: str = Field(description="Overall explanation of the likely data lineage.")
    relationships: List[LineageRelationshipSchema] = Field(default_factory=list)


class ForecastAnalysisSchema(BaseModel):
    time_column: str = Field(description="The most likely time dimension for forecasting.")
    metric_column: str = Field(description="The metric that should be forecast.")
    method: str = Field(description="Recommended forecasting method or family of methods.")
    trend_summary: str = Field(description="Direction and shape of the trend.")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in the recommendation.")


class SecurityRiskSchema(BaseModel):
    field: str = Field(description="Field or table.column suspected to be sensitive.")
    risk_level: Literal["low", "medium", "high"] = Field(description="Sensitivity level of the field.")
    evidence: str = Field(description="Reason the field looks sensitive.")
    recommendation: str = Field(description="Control to apply to this field.")


class SecurityAssessmentSchema(BaseModel):
    summary: str = Field(description="Overall security posture assessment for the scanned tables.")
    risks: List[SecurityRiskSchema] = Field(default_factory=list)


class BusinessRecommendationSchema(BaseModel):
    title: str = Field(description="Actionable recommendation for the business or data team.")
    priority: Literal["low", "medium", "high"] = Field(description="Priority of this recommendation.")
    impact: str = Field(description="Expected business impact of this recommendation.")
    owner: str = Field(description="Likely owner or team responsible for action.")
    dashboard_tile: str = Field(description="Short label for a dashboard card or KPI tile.")


class BusinessNarrativeSchema(BaseModel):
    headline: str = Field(description="Short headline summarizing the business implication of the data findings.")
    narrative: str = Field(description="Business-facing narrative that explains what is happening and why it matters.")
    key_insights: List[str] = Field(default_factory=list, description="Most important points to highlight to stakeholders.")
    recommendations: List[BusinessRecommendationSchema] = Field(default_factory=list, description="Priority actions for the business team.")
    dashboard_ready: str = Field(description="Concise, card-friendly summary suitable for a dashboard or executive briefing.")


class QualityAgentSchema(BaseModel):
    user_question: str = ""
    table_name: str = ""
    summary: str = ""
    findings: List[dict] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    final_answer: str = ""


class LineageAgentSchema(BaseModel):
    user_question: str = ""
    relationships: List[dict] = Field(default_factory=list)
    final_answer: str = ""


class ForecastAgentSchema(BaseModel):
    user_question: str = ""
    table_name: str = ""
    metric_column: str = ""
    trend_summary: str = ""
    final_answer: str = ""


class SecurityAgentSchema(BaseModel):
    user_question: str = ""
    sensitive_fields: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    final_answer: str = ""


class BusinessSummaryAgentSchema(BaseModel):
    user_question: str = ""
    raw_analysis: str = ""
    audience: Literal["executive", "analyst", "operator", "general"] = "general"
    headline: str = ""
    narrative: str = ""
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[dict] = Field(default_factory=list)
    dashboard_ready: str = ""
    executive_briefing: str = ""
    analyst_briefing: str = ""
    operator_briefing: str = ""
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

