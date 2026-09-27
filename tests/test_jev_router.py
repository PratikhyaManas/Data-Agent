"""
Tests for utils/jev_router.py. Uses a fake TypeSafeClassifier stand-in
throughout, so these run with no TYPESAFE_API_KEY, no network, and no
dependency on langchain-typesafe actually being installed.
"""
import utils.jev_router as jev_router


class FakeAnswer:
    def __init__(self, choice, probabilities):
        self.choice = choice
        self.probabilities = probabilities


class FakeResult:
    def __init__(self, choice, probabilities):
        self.choices = {"route": FakeAnswer(choice, probabilities)}


class FakeClassifier:
    def __init__(self, choice="sql", confidence=0.9, raises=None):
        self._choice = choice
        self._confidence = confidence
        self._raises = raises

    def invoke(self, state):
        if self._raises:
            raise self._raises
        return FakeResult(self._choice, {self._choice: self._confidence})


def setup_function(_):
    # Each test starts from a clean slate regardless of test order.
    jev_router._reset_cache_for_tests()


def test_returns_none_when_routing_disabled(monkeypatch):
    monkeypatch.setattr(jev_router, "JEV_ROUTING_ENABLED", False)
    monkeypatch.setattr(jev_router, "_get_classifier", lambda: FakeClassifier())
    assert jev_router.jev_route("show me top users", "(none)") is None


def test_returns_none_when_classifier_unavailable(monkeypatch):
    # Simulates: package not installed, or TYPESAFE_API_KEY not set.
    monkeypatch.setattr(jev_router, "JEV_ROUTING_ENABLED", True)
    monkeypatch.setattr(jev_router, "_get_classifier", lambda: None)
    assert jev_router.jev_route("show me top users", "(none)") is None


def test_confident_answer_is_returned(monkeypatch):
    monkeypatch.setattr(jev_router, "JEV_ROUTING_ENABLED", True)
    monkeypatch.setattr(
        jev_router, "_get_classifier", lambda: FakeClassifier(choice="sql", confidence=0.92)
    )
    result = jev_router.jev_route("top 5 users by rating", "(none)")
    assert result == ("sql", 0.92)


def test_low_confidence_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(jev_router, "JEV_ROUTING_ENABLED", True)
    monkeypatch.setattr(jev_router, "JEV_ROUTE_CONFIDENCE_THRESHOLD", 0.6)
    monkeypatch.setattr(
        jev_router, "_get_classifier", lambda: FakeClassifier(choice="etl", confidence=0.4)
    )
    assert jev_router.jev_route("do something with the data", "(none)") is None


def test_api_error_falls_back_to_none(monkeypatch):
    monkeypatch.setattr(jev_router, "JEV_ROUTING_ENABLED", True)
    monkeypatch.setattr(
        jev_router,
        "_get_classifier",
        lambda: FakeClassifier(raises=ConnectionError("typesafe unreachable")),
    )
    assert jev_router.jev_route("chart average rating", "(none)") is None


def test_get_classifier_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev_router._get_classifier() is None
