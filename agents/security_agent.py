from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import SecurityAgentSchema, SecurityAssessmentSchema
from utils.database import DatabaseUtil
from utils.llm_pick import invoke_with_resilience


db = DatabaseUtil()


def _detect_sensitive_columns() -> List[str]:
    sensitive = []
    for table in db.list_tables():
        for column in db.column_info(table):
            name = column['name'].lower()
            if any(token in name for token in [
                'email', 'phone', 'mobile', 'ssn', 'passport', 'address',
                'street', 'zip', 'postal', 'dob', 'birth', 'salary',
                'payment', 'card', 'cvv', 'token', 'key', 'secret', 'password',
                'api', 'auth', 'ip_address', 'device_id', 'driver_license', 'national_id'
            ]):
                sensitive.append(f"{table}.{column['name']}")
    return sensitive


def assess_security(state: SecurityAgentSchema) -> SecurityAgentSchema:
    schema = db.schema_details()
    sensitive_fields = _detect_sensitive_columns()
    prompt = (
        "Identify likely sensitive or privacy-sensitive columns using the database schema and probable field names.\n"
        f"Request: {state.user_question}\n\nDatabase schema:\n{schema}\n\n"
        f"Sensitive field candidates from naming patterns:\n{sensitive_fields or 'No obvious sensitive columns detected by name'}\n\n"
        "Return a structured security assessment: summary and a list of risk entries with field, risk level, evidence, and recommendation."
    )
    result = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=SecurityAssessmentSchema)
    state.sensitive_fields = sensitive_fields
    state.recommendations = [risk.recommendation for risk in result.risks]
    risk_lines = []
    for risk in result.risks:
        field_name = risk.field if risk.field else "unknown_field"
        risk_lines.append(f"- {field_name} [{risk.risk_level}]: {risk.evidence} -> {risk.recommendation}")
    state.final_answer = result.summary + "\n\n" + "\n".join(risk_lines)
    return state


def build_security_agent():
    graph = StateGraph(SecurityAgentSchema)
    graph.add_node("assess_security", assess_security)
    graph.set_entry_point("assess_security")
    graph.add_edge("assess_security", END)
    return graph.compile()


security_agent = build_security_agent()
