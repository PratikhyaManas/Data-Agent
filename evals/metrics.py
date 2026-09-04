"""
Scoring functions for the eval harness. Deterministic and dependency-free
(no LLM calls) so they're fast and testable on their own.
"""
from typing import Any, Dict, List


def _row_signature(row: Dict[str, Any]) -> tuple:
    """
    Order-independent, column-name-independent signature for a single row:
    sorted, stringified values. This is a heuristic, not an exact match -
    it means a generated query that returns the same values under
    different column aliases or column order still counts as correct,
    which is what we want for "does this answer the question" grading.
    It will NOT catch a query that happens to return the same values in
    a coincidentally-matching but semantically different shape; for that,
    pair execution-accuracy with the LLM-as-judge verdict already in the
    pipeline rather than relying on this alone.
    """
    return tuple(sorted(str(v) for v in row.values()))


def normalize_rows(rows: List[Dict[str, Any]]) -> List[tuple]:
    return sorted(_row_signature(r) for r in rows)


def rows_equal(golden: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> bool:
    """Order-independent, alias-independent row-set comparison."""
    return normalize_rows(golden) == normalize_rows(actual)


def rows_subset(golden: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> bool:
    """
    True if every golden row appears somewhere in actual. Useful when the
    agent's query is allowed to return extra context columns beyond what
    the golden query strictly asked for.
    """
    actual_sigs = set(normalize_rows(actual))
    return all(_row_signature(r) in actual_sigs for r in golden)


def accuracy(results: List[Dict[str, Any]], key: str = "passed") -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get(key)) / len(results)
