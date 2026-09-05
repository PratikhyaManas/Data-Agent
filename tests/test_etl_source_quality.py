"""
Tests for the pre-load source data quality check in agents/etl_analyst.py.
This node is purely deterministic (no LLM call), so it's tested directly
via node functions rather than through the full agent graph.
"""
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from agents.etl_analyst import (
    check_source_quality,
    source_quality_decision,
    MAX_SOURCE_DQ_RETRIES,
)
from Models.schema import ETLAgentSchema


def _state_with_transform_call(input_path, retries=0):
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "transform_load_tool",
                "args": {"input_path": input_path, "pandas_code": "pass", "output_filename": "out"},
                "id": "call_1",
            }
        ],
    )
    return ETLAgentSchema(
        messages=[HumanMessage(content="transform the file"), ai_msg],
        original_request="transform the file",
        source_dq_retries=retries,
    )


def test_no_tool_calls_is_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = ETLAgentSchema(messages=[HumanMessage(content="hi"), AIMessage(content="done")])
    result = check_source_quality(state)
    assert result.source_dq_severity == "ok"
    assert result.source_quality_report == ""


def test_extract_tool_call_has_nothing_to_precheck(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "extract_load_tool", "args": {"api_url": "http://x"}, "id": "call_1"}],
    )
    state = ETLAgentSchema(messages=[HumanMessage(content="extract"), ai_msg])
    result = check_source_quality(state)
    assert result.source_dq_severity == "ok"


def test_nonexistent_input_path_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = _state_with_transform_call(str(tmp_path / "does_not_exist.csv"))
    result = check_source_quality(state)
    assert result.source_dq_severity == "ok"


def test_clean_source_is_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "clean.csv"
    pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": list("vwxyz")}).to_csv(src, index=False)

    state = _state_with_transform_call(str(src))
    result = check_source_quality(state)

    assert result.source_dq_severity == "ok"
    assert source_quality_decision(result) == "proceed"


def test_critical_source_issue_injects_feedback_and_retries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "empty.csv"
    pd.DataFrame({"a": []}).to_csv(src, index=False)

    state = _state_with_transform_call(str(src))
    result = check_source_quality(state)

    assert result.source_dq_severity == "critical"
    assert result.source_dq_retries == 1
    assert source_quality_decision(result) == "retry"
    # feedback should be injected as a new HumanMessage for the agent loop to pick up
    assert isinstance(result.messages[-1], HumanMessage)
    assert "critical issue" in result.messages[-1].content


def test_retries_exhausted_proceeds_anyway(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "empty.csv"
    pd.DataFrame({"a": []}).to_csv(src, index=False)

    state = _state_with_transform_call(str(src), retries=MAX_SOURCE_DQ_RETRIES)
    result = check_source_quality(state)

    assert result.source_dq_severity == "critical"
    # counter keeps climbing past the cap (unconditional increment) so the
    # decision below correctly stops retrying instead of looping forever
    assert result.source_dq_retries == MAX_SOURCE_DQ_RETRIES + 1
    assert source_quality_decision(result) == "proceed"
