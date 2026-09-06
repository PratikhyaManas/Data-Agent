from Models.schema import (
    QualityAnalysisSchema,
    LineageAnalysisSchema,
    ForecastAnalysisSchema,
    SecurityAssessmentSchema,
    BusinessNarrativeSchema,
)
from agents import data_quality_agent as data_quality_module
from agents import data_lineage_agent as lineage_module
from agents import forecast_agent as forecast_module
from agents import security_agent as security_module
from agents import business_summary_agent as summary_module
from agents import data_agent as data_agent_module
from Models.schema import (
    QualityAgentSchema,
    LineageAgentSchema,
    ForecastAgentSchema,
    SecurityAgentSchema,
    BusinessSummaryAgentSchema,
)


def test_quality_agent_uses_structured_result(monkeypatch):
    monkeypatch.setattr(data_quality_module.db, "list_tables", lambda: ["users"])
    monkeypatch.setattr(data_quality_module.db, "column_info", lambda table: [
        {"name": "id", "type": "INTEGER"},
        {"name": "email", "type": "TEXT"},
    ])
    monkeypatch.setattr(data_quality_module.db, "sample_rows", lambda table, n=5: [
        {"id": 1, "email": "a@example.com"},
        {"id": 2, "email": None},
        {"id": 3, "email": None},
    ])

    monkeypatch.setattr(
        data_quality_module,
        "invoke_with_resilience",
        lambda *args, **kwargs: QualityAnalysisSchema(
            summary="Important null risk in email field",
            findings=[
                {"table": "users", "field": "email", "issue_type": "missing_values", "severity": "high", "evidence": "2/3 rows null", "recommendation": "backfill or mask"}
            ],
            risk_level="high",
        ),
    )

    state = data_quality_module.assess_quality(QualityAgentSchema(user_question="Assess quality"))
    assert state.risk_level == "high"
    assert "null" in state.summary.lower()
    assert len(state.findings) == 1


def test_lineage_agent_uses_structured_result(monkeypatch):
    monkeypatch.setattr(lineage_module.db, "list_tables", lambda: ["users", "rides"])
    monkeypatch.setattr(lineage_module.db, "column_info", lambda table: [
        {"name": "id", "type": "INTEGER"},
        {"name": "user_id", "type": "INTEGER"},
    ])
    monkeypatch.setattr(
        lineage_module,
        "invoke_with_resilience",
        lambda *args, **kwargs: LineageAnalysisSchema(
            summary="Likely users-to-rides relationship",
            relationships=[
                {"source_table": "users", "source_field": "id", "target_table": "rides", "target_field": "user_id", "confidence": "high", "reason": "matching key names"}
            ],
        ),
    )

    state = lineage_module.assess_lineage(LineageAgentSchema(user_question="Show lineage"))
    assert state.relationships[0]["source_table"] == "users"
    assert "users" in state.final_answer.lower()


def test_forecast_agent_uses_structured_result(monkeypatch):
    monkeypatch.setattr(forecast_module.db, "list_tables", lambda: ["rides"])
    monkeypatch.setattr(forecast_module.db, "column_info", lambda table: [
        {"name": "ride_date", "type": "TEXT"},
        {"name": "rating", "type": "REAL"},
    ])
    monkeypatch.setattr(
        forecast_module,
        "invoke_with_resilience",
        lambda *args, **kwargs: ForecastAnalysisSchema(
            time_column="ride_date",
            metric_column="rating",
            method="time_series_trend",
            trend_summary="Positive trend with seasonality",
            confidence="medium",
        ),
    )

    state = forecast_module.forecast_trend(ForecastAgentSchema(user_question="Forecast rating"))
    assert state.metric_column == "rating"
    assert "trend" in state.trend_summary.lower()


def test_security_agent_uses_structured_result(monkeypatch):
    monkeypatch.setattr(security_module.db, "list_tables", lambda: ["users"])
    monkeypatch.setattr(security_module.db, "column_info", lambda table: [
        {"name": "email", "type": "TEXT"},
        {"name": "name", "type": "TEXT"},
    ])
    monkeypatch.setattr(
        security_module,
        "invoke_with_resilience",
        lambda *args, **kwargs: SecurityAssessmentSchema(
            summary="Email field is likely PII",
            risks=[
                {"field": "users.email", "risk_level": "high", "evidence": "contains email pattern", "recommendation": "mask in logs and encrypt at rest"}
            ],
        ),
    )

    state = security_module.assess_security(SecurityAgentSchema(user_question="Check security"))
    assert state.sensitive_fields[0].endswith("email")
    assert "PII" in state.final_answer.upper()


def test_business_summary_agent_creates_dashboard_ready_summary(monkeypatch):
    monkeypatch.setattr(
        summary_module,
        "invoke_with_resilience",
        lambda *args, **kwargs: BusinessNarrativeSchema(
            headline="Revenue is growing, but privacy and quality risks need attention.",
            narrative="The data is healthy enough to act on, but the email field needs masking and monitoring.",
            key_insights=[
                "Email risk remains high",
                "Quality issues are concentrated in a subset of records",
            ],
            recommendations=[
                {
                    "title": "Mask PII in downstream views",
                    "priority": "high",
                    "impact": "Reduces exposure in logs and dashboards",
                    "owner": "Data Platform",
                    "dashboard_tile": "PII exposure",
                }
            ],
            dashboard_ready="Revenue up; PII exposure needs masking; quality risks require follow-up.",
        ),
    )

    state = summary_module.summarize_business_context(
        BusinessSummaryAgentSchema(
            user_question="Summarize the business impact",
            raw_analysis="Email field is likely PII; quality risks are moderate; forecast says demand is rising.",
        )
    )
    assert "Revenue" in state.headline
    assert state.recommendations[0]["priority"] == "high"
    assert "dashboard" in state.dashboard_ready.lower()


def test_summary_node_auto_chains_after_specialized_analysis(monkeypatch):
    monkeypatch.setattr(
        data_agent_module.business_summary_agent,
        "invoke",
        lambda schema: {
            "final_answer": "Executive briefing: revenue is rising, but privacy controls are recommended."
        },
    )

    state = data_agent_module.summary_node(
        data_agent_module.DataAgentSchema(
            messages=["dummy"],
            final_answer="Quality check: moderate null risk and a likely PII field were found.",
        )
    )
    assert "Executive briefing" in state.final_answer
    assert "privacy" in state.final_answer.lower()


def test_summary_agent_supports_audience_specific_briefings(monkeypatch):
    monkeypatch.setattr(
        summary_module,
        "invoke_with_resilience",
        lambda *args, **kwargs: BusinessNarrativeSchema(
            headline="Revenue is growing, but privacy and quality risks need attention.",
            narrative="The data is healthy enough to act on, but the email field needs masking and monitoring.",
            key_insights=[
                "Email risk remains high",
                "Quality issues are concentrated in a subset of records",
            ],
            recommendations=[
                {
                    "title": "Mask PII in downstream views",
                    "priority": "high",
                    "impact": "Reduces exposure in logs and dashboards",
                    "owner": "Data Platform",
                    "dashboard_tile": "PII exposure",
                }
            ],
            dashboard_ready="Revenue up; PII exposure needs masking; quality risks require follow-up.",
        ),
    )

    state = summary_module.summarize_business_context(
        BusinessSummaryAgentSchema(
            user_question="Summarize the business impact for the analyst",
            raw_analysis="Email field is likely PII; quality risks are moderate; forecast says demand is rising.",
            audience="analyst",
        )
    )
    assert "analyst" in state.analyst_briefing.lower()
    assert "operator" in state.operator_briefing.lower()
    assert "executive" in state.executive_briefing.lower()


def test_detect_audience_from_question():
    assert summary_module.detect_audience("Give me the executive briefing") == "executive"
    assert summary_module.detect_audience("Provide an analyst brief") == "analyst"
    assert summary_module.detect_audience("Ops runbook summary") == "operator"
    assert summary_module.detect_audience("Summarize the findings") == "general"


def test_summary_node_respects_explicit_audience_state(monkeypatch):
    monkeypatch.setattr(
        data_agent_module.business_summary_agent,
        "invoke",
        lambda schema: {
            "final_answer": "Executive briefing: revenue is rising, and privacy controls are recommended."
        },
    )

    state = data_agent_module.summary_node(
        data_agent_module.DataAgentSchema(
            messages=["dummy"],
            audience="executive",
            final_answer="Quality check: moderate null risk and a likely PII field were found.",
        )
    )
    assert "Executive briefing" in state.final_answer
