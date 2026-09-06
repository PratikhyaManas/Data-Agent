from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import ForecastAgentSchema, ForecastAnalysisSchema
from utils.database import DatabaseUtil
from utils.llm_pick import invoke_with_resilience


db = DatabaseUtil()


def _forecast_candidates() -> str:
    tables = db.list_tables()
    if not tables:
        return "No tables available for forecasting."

    candidates = []
    for table in tables:
        cols = db.column_info(table)
        time_cols = [c["name"] for c in cols if any(token in c["name"].lower() for token in ["date", "time", "day", "month", "year", "created", "updated"]) ]
        numeric_cols = [c["name"] for c in cols if c["type"] and any(token in c["type"].lower() for token in ["int", "real", "float", "numeric", "decimal"]) ]
        if time_cols or numeric_cols:
            candidates.append(f"Table '{table}': time_candidates={time_cols or ['none']}, numeric_candidates={numeric_cols or ['none']}")
    return "\n".join(candidates) if candidates else "No clear forecasting candidates found by schema naming."


def forecast_trend(state: ForecastAgentSchema) -> ForecastAgentSchema:
    schema = db.schema_details()
    forecast_context = _forecast_candidates()
    prompt = (
        "Recommend a forecasting strategy based on the actual table and column names in this database, not generic analytics advice.\n"
        f"Request: {state.user_question}\n\nDatabase schema:\n{schema}\n\nForecast candidate scan:\n{forecast_context}\n\n"
        "Return a structured recommendation with time column, metric column, method, trend summary, and confidence."
    )
    result = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=ForecastAnalysisSchema)
    state.table_name = result.time_column.split(".")[0] if "." in result.time_column else ""
    state.metric_column = result.metric_column
    state.trend_summary = result.trend_summary
    state.final_answer = (
        f"Recommended method: {result.method}\n"
        f"Time dimension: {result.time_column}\n"
        f"Metric: {result.metric_column}\n"
        f"Trend: {result.trend_summary}\n"
        f"Confidence: {result.confidence}"
    )
    return state


def build_forecast_agent():
    graph = StateGraph(ForecastAgentSchema)
    graph.add_node("forecast_trend", forecast_trend)
    graph.set_entry_point("forecast_trend")
    graph.add_edge("forecast_trend", END)
    return graph.compile()


forecast_agent = build_forecast_agent()
