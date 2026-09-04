"""
Tests for utils/llm_pick.py's _resolve_with_fallback - the core retry
and model-fallback logic. Uses fake exception classes and fake LLM
stand-ins throughout, so these run with no API key, no network, and no
real backoff delay (sleep is injected and captured, never actually called).
"""
import pytest

from utils.llm_pick import _resolve_with_fallback


class FakeRateLimitError(Exception):
    """Stands in for a transient/retryable API error."""


class FakeAuthError(Exception):
    """Stands in for a non-retryable client error."""


class FakeNotFoundError(Exception):
    """Stands in for an unknown-model error - worth trying a different tier for."""


RETRYABLE = (FakeRateLimitError,)
NON_RETRYABLE = (FakeAuthError,)


@pytest.fixture
def fake_sleep():
    calls = []

    def _sleep(seconds):
        calls.append(seconds)

    _sleep.calls = calls
    return _sleep


def _get_llm(tier):
    return f"llm_{tier}"


def test_immediate_success_no_retry_no_fallback(fake_sleep):
    result, tier = _resolve_with_fallback(
        tier_chain=["high", "medium", "low"],
        get_llm_for_tier=_get_llm,
        invoke_fn=lambda llm: f"result_from_{llm}",
        sleep_fn=fake_sleep,
        retryable_exceptions=RETRYABLE,
        non_retryable_exceptions=NON_RETRYABLE,
    )
    assert result == "result_from_llm_high"
    assert tier == "high"
    assert fake_sleep.calls == []


def test_retries_then_succeeds_on_same_tier_with_increasing_backoff(fake_sleep):
    attempts = {"n": 0}

    def invoke_flaky(llm):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FakeRateLimitError("rate limited")
        return "success_after_retries"

    result, tier = _resolve_with_fallback(
        tier_chain=["high", "medium"],
        get_llm_for_tier=_get_llm,
        invoke_fn=invoke_flaky,
        sleep_fn=fake_sleep,
        retryable_exceptions=RETRYABLE,
        non_retryable_exceptions=NON_RETRYABLE,
        max_retries_per_model=3,
    )
    assert result == "success_after_retries"
    assert tier == "high"
    assert len(fake_sleep.calls) == 2
    assert fake_sleep.calls[1] > fake_sleep.calls[0]  # exponential, not constant


def test_exhausting_retries_falls_back_to_next_tier(fake_sleep):
    fallback_events = []

    def invoke_fails_only_on_high(llm):
        if llm == "llm_high":
            raise FakeRateLimitError("always rate limited on high")
        return f"success_on_{llm}"

    result, tier = _resolve_with_fallback(
        tier_chain=["high", "medium", "low"],
        get_llm_for_tier=_get_llm,
        invoke_fn=invoke_fails_only_on_high,
        sleep_fn=fake_sleep,
        retryable_exceptions=RETRYABLE,
        non_retryable_exceptions=NON_RETRYABLE,
        max_retries_per_model=3,
        on_fallback=lambda frm, to, e: fallback_events.append((frm, to)),
    )
    assert result == "success_on_llm_medium"
    assert tier == "medium"
    assert fallback_events == [("high", "medium")]


def test_non_retryable_error_skips_straight_to_fallback_no_wasted_retries(fake_sleep):
    def invoke_auth_fails_on_high(llm):
        if llm == "llm_high":
            raise FakeAuthError("bad key")
        return "ok"

    result, tier = _resolve_with_fallback(
        tier_chain=["high", "medium"],
        get_llm_for_tier=_get_llm,
        invoke_fn=invoke_auth_fails_on_high,
        sleep_fn=fake_sleep,
        retryable_exceptions=RETRYABLE,
        non_retryable_exceptions=NON_RETRYABLE,
        max_retries_per_model=3,
    )
    assert result == "ok"
    assert tier == "medium"
    assert fake_sleep.calls == []  # no retry attempted on a non-retryable error


def test_not_found_error_triggers_fallback(fake_sleep):
    def invoke_not_found_on_high(llm):
        if llm == "llm_high":
            raise FakeNotFoundError("model not found")
        return "ok"

    result, tier = _resolve_with_fallback(
        tier_chain=["high", "medium"],
        get_llm_for_tier=_get_llm,
        invoke_fn=invoke_not_found_on_high,
        sleep_fn=fake_sleep,
        retryable_exceptions=RETRYABLE,
        non_retryable_exceptions=NON_RETRYABLE,
        not_found_exception=FakeNotFoundError,
    )
    assert result == "ok"
    assert tier == "medium"


def test_all_tiers_exhausted_raises_with_cause(fake_sleep):
    def invoke_always_fails(llm):
        raise FakeRateLimitError(f"always fails on {llm}")

    with pytest.raises(RuntimeError, match="exhausted"):
        _resolve_with_fallback(
            tier_chain=["high", "low"],
            get_llm_for_tier=_get_llm,
            invoke_fn=invoke_always_fails,
            sleep_fn=fake_sleep,
            retryable_exceptions=RETRYABLE,
            non_retryable_exceptions=NON_RETRYABLE,
            max_retries_per_model=2,
        )


def test_backoff_never_exceeds_max_backoff(fake_sleep):
    def invoke_always_retryable(llm):
        raise FakeRateLimitError("nope")

    with pytest.raises(RuntimeError):
        _resolve_with_fallback(
            tier_chain=["high"],
            get_llm_for_tier=_get_llm,
            invoke_fn=invoke_always_retryable,
            sleep_fn=fake_sleep,
            retryable_exceptions=RETRYABLE,
            non_retryable_exceptions=NON_RETRYABLE,
            max_retries_per_model=6,
            base_backoff=1.0,
            max_backoff=5.0,
        )
    # every sleep call should respect the cap (plus up to 0.5s jitter)
    assert all(s <= 5.5 for s in fake_sleep.calls)
