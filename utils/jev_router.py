"""
Jev-backed router classification (TypeSafe's `langchain-typesafe` package).

Jev is a "System One" model: instead of generating prose, it returns a
typed choice plus a calibrated probability for it - a much better fit
for a fixed-category router than a full LLM call, and much cheaper/
faster since the router fires on every single turn (including
follow-ups).

This module is a thin, best-effort wrapper. Any of the following
degrades to `None` rather than raising:
  - `JEV_ROUTING_ENABLED=false`
  - the `langchain-typesafe` package isn't installed
  - `TYPESAFE_API_KEY` isn't set
  - the TypeSafe API errors, times out, or the response can't be parsed
  - Jev's confidence in its top answer is below the configured threshold

The caller (`agents/data_agent.py:route_node`) treats `None` exactly
like a transient LLM failure elsewhere in this repo: fall back to the
existing LLM router. Jev is a fast path layered in front of it, never
the only way to route.
"""
import os
from typing import Optional, Tuple

# Kept in sync with the categories in agents/data_agent.py's ROUTER_PROMPT.
# 'clarify' is deliberately not a Jev category - low confidence on the
# categories below IS the "this is ambiguous" signal, and falls through
# to the LLM router, which can still choose 'clarify' itself.
JEV_ROUTES = {
    "sql": "questions that require querying a database (aggregations, filters, lookups)",
    "etl": "requests to extract data from an API, or transform/reshape a data file",
    "visualization": "requests to chart, plot, graph, or visualize data",
    "catalog": "requests to describe, document, or explain what columns/tables mean",
    "quality": "requests to assess data quality, null rates, duplicate rows, anomalies, or dataset health",
    "lineage": "requests to explain data lineage, table relationships, or how data flows between tables",
    "forecast": "requests to forecast trends, estimate future values, or identify directional patterns",
    "security": "requests to find PII, sensitive columns, or compliance/privacy risks in the data",
    "summary": (
        "requests to turn data findings into a business narrative, executive summary, "
        "or dashboard-ready recommendations"
    ),
}

JEV_ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv("JEV_ROUTE_CONFIDENCE_THRESHOLD", "0.6"))
JEV_ROUTING_ENABLED = os.getenv("JEV_ROUTING_ENABLED", "true").strip().lower() not in ("0", "false", "no")

_classifier = None
_unavailable = False


def _reset_cache_for_tests() -> None:
    """Test-only hook: clears the lazily-built classifier + unavailable
    flag so tests can exercise both the "not configured" and "configured"
    paths without import-order flakiness."""
    global _classifier, _unavailable
    _classifier = None
    _unavailable = False


def _get_classifier():
    """Lazily build (and cache) the TypeSafeClassifier. Returns None if
    the package isn't installed or no API key is configured, so callers
    fall back instead of crashing at import time."""
    global _classifier, _unavailable
    if _classifier is not None:
        return _classifier
    if _unavailable:
        return None
    if not os.getenv("TYPESAFE_API_KEY"):
        _unavailable = True
        return None
    try:
        from langchain_typesafe import Choice, TypeSafeClassifier
    except ImportError:
        _unavailable = True
        return None

    _classifier = TypeSafeClassifier(
        questions={
            "route": Choice(
                instructions=(
                    "Which category best fits this data request? Use the recent "
                    "conversation to resolve follow-ups like 'now filter that by "
                    "region' or 'chart the same thing' when the current request "
                    "alone is ambiguous, rather than guessing."
                ),
                criteria=JEV_ROUTES,
            ),
        }
    )
    return _classifier


def jev_route(question: str, history: str) -> Optional[Tuple[str, float]]:
    """
    Ask Jev to classify `question` (with `history` for follow-up context)
    into one of JEV_ROUTES.

    Returns (route, confidence) on a confident answer, or None if Jev is
    disabled, unavailable, erroring, or unsure - in every None case the
    caller should fall back to the LLM router.
    """
    if not JEV_ROUTING_ENABLED:
        return None

    classifier = _get_classifier()
    if classifier is None:
        return None

    try:
        result = classifier.invoke({"history": history, "question": question})
        answer = result.choices["route"]
        confidence = answer.probabilities[answer.choice]
    except Exception:
        # Any TypeSafe/network/parsing failure degrades to the LLM router,
        # the same way a transient LLM failure does elsewhere in this repo.
        return None

    if confidence < JEV_ROUTE_CONFIDENCE_THRESHOLD:
        return None

    return answer.choice, confidence
