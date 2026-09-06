from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import LineageAgentSchema, LineageAnalysisSchema
from utils.database import DatabaseUtil
from utils.llm_pick import invoke_with_resilience


db = DatabaseUtil()


def _infer_relationship_candidates() -> list[dict]:
    tables = db.list_tables()
    relationships = []
    seen = set()

    for table in tables:
        cols = [c["name"] for c in db.column_info(table)]
        for col in cols:
            norm = col.lower().replace("_", "")
            for other in tables:
                if other == table:
                    continue
                for other_col in [c["name"] for c in db.column_info(other)]:
                    other_norm = other_col.lower().replace("_", "")
                    if ("id" in col.lower() or "id" in other_col.lower()) and (norm == other_norm or norm.endswith(other_norm) or other_norm.endswith(norm)):
                        key = (table, col, other, other_col)
                        if key not in seen:
                            seen.add(key)
                            relationships.append({
                                "source_table": table,
                                "source_field": col,
                                "target_table": other,
                                "target_field": other_col,
                                "confidence": "medium",
                                "reason": "matching key-like column names suggest a parent-child relationship",
                            })
    return relationships


def assess_lineage(state: LineageAgentSchema) -> LineageAgentSchema:
    schema = db.schema_details()
    relation_candidates = _infer_relationship_candidates()
    prompt = (
        "Explain the likely data lineage and table relationships in this database using the actual schema and inferred key matches.\n"
        f"Request: {state.user_question}\n\nDatabase schema:\n{schema}\n\n"
        f"Inferred relationship candidates:\n{relation_candidates or 'None detected from naming patterns'}\n\n"
        "Return a concise structured assessment with the likely lineage summary and a list of concrete relationships."
    )
    result = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=LineageAnalysisSchema)
    state.relationships = [item.model_dump() for item in result.relationships]
    state.final_answer = result.summary + "\n\n" + "\n".join(
        f"- {item['source_table']}.{item['source_field']} -> {item['target_table']}.{item['target_field']} ({item['confidence']}): {item['reason']}"
        for item in state.relationships
    )
    return state


def build_lineage_agent():
    graph = StateGraph(LineageAgentSchema)
    graph.add_node("assess_lineage", assess_lineage)
    graph.set_entry_point("assess_lineage")
    graph.add_edge("assess_lineage", END)
    return graph.compile()


data_lineage_agent = build_lineage_agent()
