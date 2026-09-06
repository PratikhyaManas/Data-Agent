from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import QualityAgentSchema, QualityAnalysisSchema
from utils.database import DatabaseUtil
from utils.llm_pick import invoke_with_resilience


db = DatabaseUtil()


def _table_quality_summary() -> str:
    tables = db.list_tables()
    if not tables:
        return "No tables found in the database."

    blocks: List[str] = []
    for table in tables:
        cols = db.column_info(table)
        sample = db.sample_rows(table, n=5)
        if not sample:
            blocks.append(f"Table '{table}': no rows found; columns={[(c['name'], c['type']) for c in cols]}")
            continue

        numeric_cols = []
        null_cols = []
        for col in cols:
            values = [row.get(col['name']) for row in sample]
            non_null = [v for v in values if v is not None]
            null_ratio = (len(values) - len(non_null)) / max(len(values), 1)
            if null_ratio > 0.2:
                null_cols.append(f"{col['name']} ({null_ratio:.0%} null)")
            if any(isinstance(v, (int, float)) for v in non_null):
                numeric_cols.append(col['name'])

        duplicate_rows = 0
        seen = set()
        for row in sample:
            key = tuple(sorted(row.items()))
            if key in seen:
                duplicate_rows += 1
            seen.add(key)

        parts = [f"Table '{table}'"]
        if cols:
            parts.append(f"columns={[(c['name'], c['type']) for c in cols]}")
        if null_cols:
            parts.append(f"likely_null_columns={null_cols}")
        if numeric_cols:
            parts.append(f"numeric_columns={numeric_cols}")
        if duplicate_rows:
            parts.append(f"duplicate_sample_rows={duplicate_rows}")
        parts.append(f"sample_rows={sample[:3]}")
        blocks.append("; ".join(parts))

    return "\n".join(blocks)


def assess_quality(state: QualityAgentSchema) -> QualityAgentSchema:
    quality_context = _table_quality_summary()
    prompt = (
        "Assess the data quality of the tables in this database using the schema and sample rows below.\n"
        f"Request: {state.user_question}\n\nDatabase quality context:\n{quality_context}\n\n"
        "Identify likely missing values, duplicates, outliers, or schema drift using the actual samples. "
        "Return a concise structured assessment with a summary, risk level, and a list of concrete findings."
    )
    result = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=QualityAnalysisSchema)
    state.summary = result.summary
    state.risk_level = result.risk_level
    state.findings = [item.model_dump() for item in result.findings]
    state.final_answer = result.summary + "\n\n" + "\n".join(
        f"- {item['table']}.{item['field']} [{item['issue_type']}, {item['severity']}]: {item['evidence']} -> {item['recommendation']}"
        for item in state.findings
    )
    return state


def build_quality_agent():
    graph = StateGraph(QualityAgentSchema)
    graph.add_node("assess_quality", assess_quality)
    graph.set_entry_point("assess_quality")
    graph.add_edge("assess_quality", END)
    return graph.compile()


data_quality_agent = build_quality_agent()
