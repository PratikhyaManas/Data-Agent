"""Tests for evals/metrics.py - the scoring logic behind the eval harness."""
from evals.metrics import rows_equal, normalize_rows, rows_subset, accuracy


def test_rows_equal_matches_despite_different_column_aliases():
    golden = [{"vehicle_type": "suv", "AVG(rating)": 4.43}]
    actual = [{"vehicle_type": "suv", "avg_rating": 4.43}]
    assert rows_equal(golden, actual) is True


def test_rows_equal_ignores_row_order():
    golden = [{"a": 1}, {"a": 2}, {"a": 3}]
    actual = [{"a": 3}, {"a": 1}, {"a": 2}]
    assert rows_equal(golden, actual) is True


def test_rows_equal_detects_different_values():
    golden = [{"a": 1}, {"a": 2}]
    actual = [{"a": 1}, {"a": 99}]
    assert rows_equal(golden, actual) is False


def test_rows_equal_detects_different_row_counts():
    golden = [{"a": 1}, {"a": 2}]
    actual = [{"a": 1}]
    assert rows_equal(golden, actual) is False


def test_rows_equal_preserves_duplicate_values_within_row():
    golden = [{"a": 5, "b": 5}]
    matching = [{"a": 5, "b": 5}]
    different = [{"a": 5, "b": 6}]
    assert rows_equal(golden, matching) is True
    assert rows_equal(golden, different) is False


def test_rows_equal_empty_sets_match():
    assert rows_equal([], []) is True


def test_rows_subset_true_when_golden_rows_all_present():
    golden = [{"a": 1}]
    actual = [{"a": 1}, {"a": 2}, {"a": 3}]
    assert rows_subset(golden, actual) is True


def test_rows_subset_false_when_golden_row_missing():
    golden = [{"a": 99}]
    actual = [{"a": 1}, {"a": 2}]
    assert rows_subset(golden, actual) is False


def test_accuracy_computes_pass_rate():
    results = [{"passed": True}, {"passed": True}, {"passed": False}, {"passed": True}]
    assert accuracy(results) == 0.75


def test_accuracy_empty_list_is_zero():
    assert accuracy([]) == 0.0


def test_accuracy_custom_key():
    results = [{"ok": True}, {"ok": False}]
    assert accuracy(results, key="ok") == 0.5
