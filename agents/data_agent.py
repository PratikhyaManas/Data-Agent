"""
Data Agent (Main Router).

Classifies the incoming natural-language request and dispatches to
the SQL, ETL, or Visualization sub-agent. Falls back to asking a
clarifying question if intent is ambiguous.
"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import (
    DataAgentSchema,
    RouterSchema,
    AgentSchema,
    ETLAgentSchema,
    VizAgentSchema,
    DataCatalogSchema,
    QualityAgentSchema,
    LineageAgentSchema,
    ForecastAgentSchema,
    SecurityAgentSchema,
    BusinessSummaryAgentSchema,
)
from utils.llm_pick import invoke_with_resilience
from utils.audit import log_event
from agents.sql_analyst import sql_analyst
from agents.etl_analyst import etl_analyst
from agents.visualization_agent import visualization_agent
from agents.data_catalog_agent import data_catalog_agent
from agents.data_quality_agent import data_quality_agent
from agents.data_lineage_agent import data_lineage_agent
from agents.forecast_agent import forecast_agent
from agents.security_agent import security_agent
from agents.business_summary_agent import business_summary_agent

ROUTER_PROMPT = (
    "Classify the user's request into exactly one category:\n"
    "- 'sql': questions that require querying a database (aggregations, filters, lookups)\n"
    "- 'etl': requests to extract data from an API, or transform/reshape a data file\n"
    "- 'visualization': requests to chart, plot, graph, or visualize data\n"
    "- 'catalog': requests to describe, document, or explain what columns/tables mean\n"
    "- 'quality': requests to assess data quality, null rates, duplicate rows, anomalies, or dataset health\n"
    "- 'lineage': requests to explain data lineage, table relationships, or how data flows between tables\n"
    "- 'forecast': requests to forecast trends, estimate future values, or identify directional patterns\n"
    "- 'security': requests to find PII, sensitive columns, or compliance/privacy risks in the data\n"
    "- 'summary': requests to turn data findings into a business narrative, executive summary, or dashboard-ready recommendations\n"
    "- 'clarify': the request is too ambiguous to route confidently\n\n"
    "Recent conversation (most recent last, may be empty):\n{history}\n\n"
    "Current user request: {question}\n\n"
    "Use the recent conversation to resolve follow-ups like 'now filter that by "
    "region' or 'chart the same thing' - infer intent from prior turns when the "
    "current request alone is ambiguous, rather than defaulting to 'clarify'."
)


def _format_history(history: list, max_turns: int = 6) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history[-max_turns:]:
        role = "User" if isinstance(turn, HumanMessage) else "Assistant"
        lines.append(f"{role}: {turn.content}")
    return "\n".join(lines)


def route_node(state: DataAgentSchema) -> DataAgentSchema:
    last_user_msg = state.messages[-1].content
    prompt = ROUTER_PROMPT.format(
        history=_format_history(state.conversation_history), question=last_user_msg
    )
    result = invoke_with_resilience("low", [HumanMessage(content=prompt)], structured_schema=RouterSchema)
    state.route_response = result.answer
    log_event("route", question=last_user_msg, decision=result.answer, reasoning=result.comments)
    return state


def route_decision(state: DataAgentSchema) -> str:
    return state.route_response


def sql_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    result = sql_analyst.invoke(AgentSchema(user_question=question))
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    return state


def etl_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    result = etl_analyst.invoke(ETLAgentSchema(messages=[HumanMessage(content=question)]))
    msgs = result["messages"] if isinstance(result, dict) else result.messages
    state.final_answer = msgs[-1].content
    log_event("etl_run", question=question, result=state.final_answer[:500])
    return state


def viz_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    # naive default: expects the user to have mentioned a CSV path; otherwise
    # a production version would first check recent SQL results in memory.
    data_source = _extract_path(question)
    result = visualization_agent.invoke(
        VizAgentSchema(user_request=question, data_source=data_source)
    )
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("viz_run", question=question, data_source=data_source, result=answer)
    return state


def catalog_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    refresh = any(kw in question.lower() for kw in ("refresh", "regenerate", "update all"))
    result = data_catalog_agent.invoke(DataCatalogSchema(refresh=refresh))
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("catalog_run", question=question, refresh=refresh, result=answer)
    return state


def quality_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    result = data_quality_agent.invoke(QualityAgentSchema(user_question=question))
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("quality_run", question=question, result=answer)
    return state


def lineage_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    result = data_lineage_agent.invoke(LineageAgentSchema(user_question=question))
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("lineage_run", question=question, result=answer)
    return state


def forecast_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    result = forecast_agent.invoke(ForecastAgentSchema(user_question=question))
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("forecast_run", question=question, result=answer)
    return state


def security_node(state: DataAgentSchema) -> DataAgentSchema:
    question = state.messages[-1].content
    result = security_agent.invoke(SecurityAgentSchema(user_question=question))
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("security_run", question=question, result=answer)
    return state


def summary_node(state: DataAgentSchema) -> DataAgentSchema:
    last_message = state.messages[-1] if state.messages else ""
    question = getattr(last_message, "content", str(last_message)) if last_message else ""
    raw_analysis = state.final_answer or "No prior analysis available."
    audience = state.audience if state.audience != "general" else "general"
    normalized = (question or "").lower()
    if audience == "general":
        if any(keyword in normalized for keyword in ["executive briefing", "leadership briefing", "board", "cxo", "exec"]):
            audience = "executive"
        elif any(keyword in normalized for keyword in ["analyst briefing", "analyst", "data team", "deep dive"]):
            audience = "analyst"
        elif any(keyword in normalized for keyword in ["operator briefing", "ops", "operations", "operator", "runbook", "monitoring"]):
            audience = "operator"

    result = business_summary_agent.invoke(
        BusinessSummaryAgentSchema(
            user_question=question,
            raw_analysis=raw_analysis,
            audience=audience,
        )
    )
    answer = result["final_answer"] if isinstance(result, dict) else result.final_answer
    state.final_answer = answer
    log_event("summary_run", question=question, result=answer, audience=audience)
    return state


def clarify_node(state: DataAgentSchema) -> DataAgentSchema:
    state.final_answer = (
        "I'm not sure whether this is a database query, a data extraction/transform "
        "task, or a chart request. Could you clarify what you'd like to do?"
    )
    state.needs_clarification = True
    log_event("clarify", question=state.messages[-1].content)
    return state


def _extract_path(text: str) -> str:
    for token in text.split():
        if token.endswith(".csv"):
            return token
    return "data/vehicles.csv"  # fallback sample


def build_data_agent():
    graph = StateGraph(DataAgentSchema)
    graph.add_node("route", route_node)
    graph.add_node("sql", sql_node)
    graph.add_node("etl", etl_node)
    graph.add_node("visualization", viz_node)
    graph.add_node("catalog", catalog_node)
    graph.add_node("quality", quality_node)
    graph.add_node("lineage", lineage_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("security", security_node)
    graph.add_node("summary", summary_node)
    graph.add_node("clarify", clarify_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "sql": "sql",
            "etl": "etl",
            "visualization": "visualization",
            "catalog": "catalog",
            "quality": "quality",
            "lineage": "lineage",
            "forecast": "forecast",
            "security": "security",
            "summary": "summary",
            "clarify": "clarify",
        },
    )
    graph.add_edge("sql", END)
    graph.add_edge("etl", END)
    graph.add_edge("visualization", END)
    graph.add_edge("catalog", END)
    graph.add_edge("quality", "summary")
    graph.add_edge("lineage", "summary")
    graph.add_edge("forecast", "summary")
    graph.add_edge("security", "summary")
    graph.add_edge("summary", END)
    graph.add_edge("clarify", END)

    return graph.compile()


data_agent = build_data_agent()

if __name__ == "__main__":
    response = data_agent.invoke(
        DataAgentSchema(messages=[HumanMessage(content="Show me the top 5 users with the highest ratings")])
    )
    answer = response["final_answer"] if isinstance(response, dict) else response.final_answer
    print(answer)
