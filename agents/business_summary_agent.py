from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import BusinessSummaryAgentSchema, BusinessNarrativeSchema
from utils.llm_pick import invoke_with_resilience


def detect_audience(question: str) -> str:
    normalized = (question or "").lower()
    if any(keyword in normalized for keyword in ["executive briefing", "leadership briefing", "board", "cxo", "exec"]):
        return "executive"
    if any(keyword in normalized for keyword in ["analyst briefing", "analyst", "data team", "deep dive"]):
        return "analyst"
    if any(keyword in normalized for keyword in ["operator briefing", "ops", "operations", "operator", "runbook", "monitoring"]):
        return "operator"
    return "general"


def summarize_business_context(state: BusinessSummaryAgentSchema) -> BusinessSummaryAgentSchema:
    if state.audience == "general":
        state.audience = detect_audience(state.user_question)

    prompt = (
        "Turn the technical analysis into a concise business narrative and a dashboard-ready summary.\n"
        f"Request: {state.user_question}\n\n"
        f"Raw analysis:\n{state.raw_analysis}\n\n"
        "Produce a headline, a business-facing narrative, 2-4 key insights, and a list of concrete recommendations. "
        "Each recommendation should include title, priority, impact, owner, and dashboard_tile. "
        "Also provide a short dashboard_ready summary that fits into an executive KPI card or briefing."
    )

    result = invoke_with_resilience(
        "medium",
        [HumanMessage(content=prompt)],
        structured_schema=BusinessNarrativeSchema,
    )

    state.headline = result.headline
    state.narrative = result.narrative
    state.key_insights = result.key_insights
    state.recommendations = [item.model_dump() for item in result.recommendations]
    dashboard_summary = result.dashboard_ready
    if "dashboard" not in dashboard_summary.lower():
        dashboard_summary = f"Dashboard: {dashboard_summary}"
    state.dashboard_ready = dashboard_summary

    state.executive_briefing = (
        f"Executive briefing\n"
        f"Headline: {result.headline}\n"
        f"Story: {result.narrative}\n"
        f"Key points: {'; '.join(result.key_insights)}\n"
        f"Next step: {result.recommendations[0].title if result.recommendations else 'Review operating plan'}"
    )
    state.analyst_briefing = (
        f"Analyst briefing\n"
        f"Summary: {result.narrative}\n"
        f"Drivers: {'; '.join(result.key_insights)}\n"
        f"Actions: {'; '.join(item.title for item in result.recommendations)}"
    )
    state.operator_briefing = (
        f"Operator briefing\n"
        f"Monitor: {', '.join(result.key_insights)}\n"
        f"Priority tasks: {'; '.join(item.title for item in result.recommendations)}\n"
        f"Dashboard note: {state.dashboard_ready}"
    )

    audience_briefing = {
        "executive": state.executive_briefing,
        "analyst": state.analyst_briefing,
        "operator": state.operator_briefing,
        "general": state.narrative,
    }.get(state.audience, state.narrative)

    state.final_answer = (
        f"{result.headline}\n\n"
        f"{audience_briefing}\n\n"
        f"Key insights:\n" + "\n".join(f"- {item}" for item in result.key_insights) + "\n\n"
        f"Recommended actions:\n" + "\n".join(
            f"- {item.title} [{item.priority}] - {item.impact} ({item.owner}; dashboard tile: {item.dashboard_tile})"
            for item in result.recommendations
        ) + "\n\n"
        f"Dashboard summary: {state.dashboard_ready}"
    )
    return state


def build_business_summary_agent():
    graph = StateGraph(BusinessSummaryAgentSchema)
    graph.add_node("summarize_business_context", summarize_business_context)
    graph.set_entry_point("summarize_business_context")
    graph.add_edge("summarize_business_context", END)
    return graph.compile()


business_summary_agent = build_business_summary_agent()
