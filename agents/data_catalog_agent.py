"""
Data Catalog Agent.

Maintains human-readable column descriptions across every table in the
database. Pipeline: gather_schema (list tables + columns + sample rows) ->
generate_descriptions (LLM, structured output, only for columns missing a
description unless refresh=True) -> save_catalog (persist to
data/data_catalog.json, preserving untouched entries) -> summarize.

The persistence layer (utils/data_catalog.py) is fully deterministic and
unit-tested on its own; only generate_descriptions here touches an LLM.
"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from Models.schema import DataCatalogSchema, ColumnDescriptionListSchema
from utils.database import DatabaseUtil
from utils.llm_pick import invoke_with_resilience
from utils.audit import log_event
from utils.data_catalog import (
    load_catalog,
    save_catalog,
    merge_descriptions,
    columns_needing_descriptions,
)

db = DatabaseUtil()


def gather_schema(state: DataCatalogSchema) -> DataCatalogSchema:
    state.schema_text = db.schema_details()
    return state


def _tables_to_catalog(state: DataCatalogSchema) -> list:
    tables = db.list_tables()
    if state.tables_filter:
        tables = [t for t in tables if t in state.tables_filter]
    return tables


def generate_descriptions(state: DataCatalogSchema) -> DataCatalogSchema:
    tables = _tables_to_catalog(state)
    tables_with_columns = {t: [c["name"] for c in db.column_info(t)] for t in tables}

    catalog = load_catalog()
    missing = columns_needing_descriptions(tables_with_columns, catalog, refresh=state.refresh)

    if not missing:
        state.updated_columns = 0
        state.catalog_report = "All requested columns are already cataloged; nothing to update."
        return state

    # Build one prompt covering every missing column, grouped by table, with
    # a few sample rows per table for context - cheaper than one LLM call
    # per column, and gives the model cross-column context within a table.
    tables_needed = sorted({t for t, _ in missing})
    context_blocks = []
    for table in tables_needed:
        columns = db.column_info(table)
        col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        samples = db.sample_rows(table, n=3)
        needed_cols = [c for t, c in missing if t == table]
        context_blocks.append(
            f"Table `{table}`: columns = {col_desc}\n"
            f"Sample rows: {samples}\n"
            f"Columns needing a description: {needed_cols}"
        )

    prompt = (
        "For each table below, write a concise, human-readable description "
        "for every listed column, inferred from its name, declared type, and "
        "the sample row values. Only describe the columns explicitly listed "
        "as needing one.\n\n" + "\n\n".join(context_blocks)
    )
    result = invoke_with_resilience("medium", [HumanMessage(content=prompt)], structured_schema=ColumnDescriptionListSchema)

    descriptions = [d.model_dump() for d in result.descriptions]
    merged = merge_descriptions(catalog, descriptions)
    save_catalog(merged)

    state.updated_columns = len(descriptions)
    state.catalog_report = (
        f"Updated {len(descriptions)} column description(s) across {len(tables_needed)} table(s): "
        f"{', '.join(tables_needed)}."
    )

    log_event(
        "data_catalog_update",
        tables=tables_needed,
        updated_columns=state.updated_columns,
        refresh=state.refresh,
    )
    return state


def summarize(state: DataCatalogSchema) -> DataCatalogSchema:
    state.final_answer = state.catalog_report
    return state


def build_data_catalog_agent():
    graph = StateGraph(DataCatalogSchema)
    graph.add_node("gather_schema", gather_schema)
    graph.add_node("generate_descriptions", generate_descriptions)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("gather_schema")
    graph.add_edge("gather_schema", "generate_descriptions")
    graph.add_edge("generate_descriptions", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


data_catalog_agent = build_data_catalog_agent()

if __name__ == "__main__":
    result = data_catalog_agent.invoke(DataCatalogSchema())
    print(result["final_answer"] if isinstance(result, dict) else result.final_answer)
