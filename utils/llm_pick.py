"""
Central place to pick which Claude model handles a task, based on
complexity tier - and, since this module wraps every LLM call site in
the project, the central place to make those calls resilient:

  - retry with exponential backoff on TRANSIENT errors (rate limits,
    timeouts, connection drops, 5xx) - these usually succeed on retry
  - model fallback down the complexity chain (high -> medium -> low) if
    a model is persistently unavailable (capacity, a bad/retired model
    name, a temporary suspension like the Fable/Mythos export-control
    pause) - these do NOT usually succeed on retry, only on a different
    model
  - fail fast, no retry, on NON-transient errors (bad request, auth,
    permission) - retrying a malformed request just wastes time and money

Previously, every agent called `llm.invoke(...)` directly with no
handling at all: any transient API error crashed the whole LangGraph run.
`invoke_with_resilience()` is the fix - agents should call it instead of
calling `.invoke()` on a `pick_llm()` result directly.
"""
import os
import random
import time
from typing import Any, Callable, List, Optional, Tuple, Type

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - exercised only when langchain_anthropic isn't installed
    ChatAnthropic = None  # pick_llm() raises a clear error below if this is ever actually called

_MODEL_MAP = {
    "low": os.getenv("LLM_MODEL_LOW", "claude-haiku-4-5-20251001"),
    "medium": os.getenv("LLM_MODEL_MEDIUM", "claude-sonnet-5"),
    "high": os.getenv("LLM_MODEL_HIGH", "claude-opus-4-8"),
}

# If the requested tier's model is unavailable, degrade to a cheaper/more
# available tier rather than failing outright. `low` has nowhere further
# to fall back to.
FALLBACK_CHAIN = {
    "high": ["high", "medium", "low"],
    "medium": ["medium", "low"],
    "low": ["low"],
}

MAX_RETRIES_PER_MODEL = int(os.getenv("LLM_MAX_RETRIES_PER_MODEL", "3"))
BASE_BACKOFF_SECONDS = float(os.getenv("LLM_BASE_BACKOFF_SECONDS", "1.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("LLM_MAX_BACKOFF_SECONDS", "20.0"))

_cache = {}


def pick_llm(complexity: str = "medium", temperature: float = 0.0) -> "ChatAnthropic":
    """
    complexity: "low" | "medium" | "high"
    Returns a cached ChatAnthropic instance for that tier.
    """
    if ChatAnthropic is None:
        raise ImportError(
            "langchain_anthropic is not installed. Run `uv sync --group dev` or `uv pip install .`."
        )
    key = (complexity, temperature)
    if key not in _cache:
        model_name = _MODEL_MAP.get(complexity, _MODEL_MAP["medium"])
        _cache[key] = ChatAnthropic(model=model_name, temperature=temperature)
    return _cache[key]


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------
# Imported lazily/defensively: this module should still be importable (and
# its retry/fallback *logic* still unit-testable via dependency injection -
# see _resolve_with_fallback) even in an environment where the `anthropic`
# package isn't installed yet.
try:
    import anthropic as _anthropic_sdk

    RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
        _anthropic_sdk.RateLimitError,
        _anthropic_sdk.APITimeoutError,
        _anthropic_sdk.APIConnectionError,
        _anthropic_sdk.InternalServerError,  # 5xx
    )
    NON_RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
        _anthropic_sdk.BadRequestError,       # 400 - malformed request, retrying won't fix it
        _anthropic_sdk.AuthenticationError,   # 401 - bad API key
        _anthropic_sdk.PermissionDeniedError, # 403
        _anthropic_sdk.UnprocessableEntityError,  # 422
        # NotFoundError (404, e.g. an unknown/retired model name) is treated
        # as fallback-worthy rather than non-retryable or blindly-retryable:
        # retrying the SAME model won't help, but the NEXT tier's model might
        # be valid. See _resolve_with_fallback's handling below.
    )
    NOT_FOUND_EXCEPTION: Optional[Type[BaseException]] = _anthropic_sdk.NotFoundError
except ImportError:  # pragma: no cover - exercised only when anthropic isn't installed
    RETRYABLE_EXCEPTIONS = ()
    NON_RETRYABLE_EXCEPTIONS = ()
    NOT_FOUND_EXCEPTION = None


# ---------------------------------------------------------------------------
# Core retry/fallback logic - deliberately decoupled from ChatAnthropic so
# it can be unit-tested with fake LLMs and fake exceptions, no API needed.
# ---------------------------------------------------------------------------
def _resolve_with_fallback(
    tier_chain: List[str],
    get_llm_for_tier: Callable[[str], Any],
    invoke_fn: Callable[[Any], Any],
    max_retries_per_model: int = MAX_RETRIES_PER_MODEL,
    base_backoff: float = BASE_BACKOFF_SECONDS,
    max_backoff: float = MAX_BACKOFF_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    retryable_exceptions: Tuple[Type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
    non_retryable_exceptions: Tuple[Type[BaseException], ...] = NON_RETRYABLE_EXCEPTIONS,
    not_found_exception: Optional[Type[BaseException]] = NOT_FOUND_EXCEPTION,
    on_retry: Optional[Callable[[str, int, float, BaseException], None]] = None,
    on_fallback: Optional[Callable[[str, str, BaseException], None]] = None,
) -> Tuple[Any, str]:
    """
    Walks tier_chain in order. For each tier: retries invoke_fn(llm) up to
    max_retries_per_model times with exponential backoff on a retryable
    error, moves to the next tier immediately on a non-retryable or
    not-found error, and moves to the next tier after exhausting retries
    on a persistently-retryable error.

    Returns (result, tier_used). Raises the last error if every tier in
    the chain is exhausted.
    """
    last_exc: Optional[BaseException] = None

    for tier_index, tier in enumerate(tier_chain):
        llm = get_llm_for_tier(tier)
        is_last_tier = tier_index == len(tier_chain) - 1

        for attempt in range(max_retries_per_model):
            try:
                result = invoke_fn(llm)
                return result, tier
            except non_retryable_exceptions as e:
                last_exc = e
                break  # this tier can't succeed no matter how we retry it
            except retryable_exceptions as e:
                last_exc = e
                if attempt < max_retries_per_model - 1:
                    wait = min(max_backoff, base_backoff * (2 ** attempt)) + random.uniform(0, 0.5)
                    if on_retry:
                        on_retry(tier, attempt, wait, e)
                    sleep_fn(wait)
                # else: retries exhausted for this tier, fall through to next tier below
            except Exception as e:
                # not_found_exception (e.g. an unknown model name) is worth
                # trying a different tier for. Anything else unclassified we
                # don't blindly retry forever - treat as non-retryable.
                last_exc = e
                if not_found_exception is not None and isinstance(e, not_found_exception):
                    break
                break

        if on_fallback and not is_last_tier and last_exc is not None:
            next_tier = tier_chain[tier_index + 1]
            on_fallback(tier, next_tier, last_exc)

    raise RuntimeError(
        f"All models in fallback chain exhausted ({tier_chain}). Last error: {last_exc}"
    ) from last_exc


def invoke_with_resilience(
    complexity: str,
    messages: List[Any],
    structured_schema: Optional[type] = None,
    tools: Optional[list] = None,
    temperature: float = 0.0,
) -> Any:
    """
    Drop-in replacement for `pick_llm(complexity).invoke(messages)` (or
    `.with_structured_output(schema).invoke(messages)`, or
    `.bind_tools(tools).invoke(messages)`) that retries transient failures
    and falls back down the model tier chain on persistent ones. This is
    what agents should call instead of invoking a pick_llm() result
    directly. Pass exactly one of structured_schema / tools, or neither
    for a plain text call.
    """
    from utils.audit import log_event  # local import - avoids a circular import at module load

    tier_chain = FALLBACK_CHAIN.get(complexity, [complexity])

    def get_llm_for_tier(tier: str):
        llm = pick_llm(tier, temperature)
        if structured_schema:
            return llm.with_structured_output(structured_schema)
        if tools:
            return llm.bind_tools(tools)
        return llm

    def invoke_fn(llm):
        return llm.invoke(messages)

    def on_retry(tier, attempt, wait, exc):
        log_event(
            "llm_retry",
            requested_complexity=complexity,
            tier=tier,
            attempt=attempt + 1,
            wait_seconds=round(wait, 2),
            error=str(exc),
        )

    def on_fallback(from_tier, to_tier, exc):
        log_event(
            "llm_fallback",
            requested_complexity=complexity,
            from_tier=from_tier,
            to_tier=to_tier,
            error=str(exc),
        )

    result, tier_used = _resolve_with_fallback(
        tier_chain=tier_chain,
        get_llm_for_tier=get_llm_for_tier,
        invoke_fn=invoke_fn,
        on_retry=on_retry,
        on_fallback=on_fallback,
    )

    if tier_used != complexity:
        log_event("llm_served_by_fallback_tier", requested_complexity=complexity, served_by=tier_used)

    return result
