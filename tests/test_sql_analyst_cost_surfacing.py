"""
Tests for agents/sql_analyst.py's generate_answer cost-note surfacing.
The LLM call inside generate_answer is stubbed via monkeypatch (dependency
injection at the same invoke_with_resilience boundary every other agent
uses) so this tests real node logic - the answer-string construction -
without needing an API key.
"""
import agents.sql_analyst as sql_analyst_module
from Models.schema import AgentSchema


class _FakeResponse:
    def __init__(self, content):
        self.content = content


def _stub_llm(monkeypatch, content="Here is your answer."):
    monkeypatch.setattr(
        sql_analyst_module, "invoke_with_resilience", lambda *a, **k: _FakeResponse(content)
    )


def test_high_cost_note_is_surfaced(monkeypatch):
    _stub_llm(monkeypatch)
    state = AgentSchema(
        user_question="q", generated_sql_query="SELECT * FROM big",
        sql_query_execution_result="[]", cost_level="high", cost_notes="full scan on big_table",
    )
    result = sql_analyst_module.generate_answer(state)
    assert "💰 Cost note: full scan on big_table" in result.final_answer


def test_medium_cost_note_is_surfaced(monkeypatch):
    """
    Regression test: estimate_cost() already computes a 'medium' cost note
    (small-table full scan) and audit-logs it, but generate_answer used to
    only ever append the 'high' cost note to the user-facing answer -
    medium-level notes were silently dropped from what the user sees.
    """
    _stub_llm(monkeypatch)
    state = AgentSchema(
        user_question="q", generated_sql_query="SELECT * FROM small",
        sql_query_execution_result="[]", cost_level="medium", cost_notes="full scan on small_table",
    )
    result = sql_analyst_module.generate_answer(state)
    assert "ℹ️ Cost note: full scan on small_table" in result.final_answer


def test_low_cost_has_no_cost_note(monkeypatch):
    _stub_llm(monkeypatch)
    state = AgentSchema(
        user_question="q", generated_sql_query="SELECT * FROM t WHERE id=1",
        sql_query_execution_result="[]", cost_level="low", cost_notes="indexed lookup",
    )
    result = sql_analyst_module.generate_answer(state)
    assert "Cost note" not in result.final_answer


def test_judge_incorrect_caveat_still_appended_alongside_cost_note(monkeypatch):
    _stub_llm(monkeypatch)
    state = AgentSchema(
        user_question="q", generated_sql_query="SELECT * FROM big",
        sql_query_execution_result="[]", cost_level="high", cost_notes="full scan",
        judge_verdict="incorrect", judge_feedback="wrong join", judge_retries=2,
    )
    result = sql_analyst_module.generate_answer(state)
    assert "⚠️ Note" in result.final_answer
    assert "💰 Cost note: full scan" in result.final_answer
