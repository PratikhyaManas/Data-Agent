"""
Tests for check_data_quality / data_quality_decision in agents/etl_analyst.py.
Deterministic (no LLM), so tested directly via node functions.
"""
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.etl_analyst import check_data_quality, data_quality_decision
from Models.schema import ETLAgentSchema


def _state_with_output(output_path, retries=0):
    tool_msg = ToolMessage(content=f"Transformed x -> {output_path} (0 rows)", tool_call_id="call_1")
    return ETLAgentSchema(
        messages=[HumanMessage(content="transform"), AIMessage(content=""), tool_msg],
        original_request="transform",
        dq_retries=retries,
    )


def test_clean_output_is_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "clean.csv"
    pd.DataFrame({"a": [1, 2, 3, 4, 5]}).to_csv(out, index=False)

    state = _state_with_output(str(out))
    result = check_data_quality(state)

    assert result.dq_severity == "ok"
    assert data_quality_decision(result) == "judge"


def test_critical_output_retries_then_gives_up_instead_of_looping_forever(tmp_path, monkeypatch):
    """
    Regression test: check_data_quality used to only increment dq_retries
    while it was still below MAX_DQ_RETRIES, which capped the counter at
    MAX forever. data_quality_decision's `dq_retries <= MAX_DQ_RETRIES`
    check then stayed true indefinitely once capped, so a source that
    never got fixed would retry forever. The counter must climb past MAX
    (unconditional increment) so the decision eventually stops retrying.
    """
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "empty.csv"
    pd.DataFrame({"a": []}).to_csv(out, index=False)

    state = _state_with_output(str(out))

    # First critical hit: retry is granted.
    state = check_data_quality(state)
    assert state.dq_retries == 1
    assert data_quality_decision(state) == "retry"

    # Simulate the retry attempt producing the same critical output again.
    state = check_data_quality(state)
    assert state.dq_retries == 2  # keeps climbing past MAX_DQ_RETRIES (1)
    assert data_quality_decision(state) == "judge"  # must stop, not "retry" forever
