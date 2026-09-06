"""Tests for utils/data_quality.py."""
import pandas as pd

from utils.data_quality import run_quality_checks


def _write_csv(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_clean_data_is_ok(tmp_path):
    path = _write_csv(tmp_path, "clean.csv", pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": list("xyzwv")}))
    result = run_quality_checks(path)
    assert result["severity"] == "ok"
    assert result["issues"] == []
    assert result["row_count"] == 5


def test_empty_file_is_critical(tmp_path):
    path = _write_csv(tmp_path, "empty.csv", pd.DataFrame({"a": []}))
    result = run_quality_checks(path)
    assert result["severity"] == "critical"
    assert any("zero rows" in i for i in result["issues"])


def test_mostly_null_column_is_critical(tmp_path):
    df = pd.DataFrame({"a": [None] * 19 + [1], "b": list(range(20))})
    path = _write_csv(tmp_path, "mostly_null.csv", df)
    result = run_quality_checks(path)
    assert result["severity"] == "critical"
    assert any("effectively empty" in i for i in result["issues"])


def test_moderate_nulls_is_warning(tmp_path):
    df = pd.DataFrame({"a": [None] * 4 + [1] * 6, "b": list(range(10))})
    path = _write_csv(tmp_path, "some_null.csv", df)
    result = run_quality_checks(path)
    assert result["severity"] == "warning"


def test_duplicates_flagged(tmp_path):
    df = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": ["x", "x", "x", "y", "z"]})
    path = _write_csv(tmp_path, "dupes.csv", df)
    result = run_quality_checks(path)
    assert result["severity"] == "warning"
    assert any("duplicate" in i for i in result["issues"])


def test_outliers_flagged(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    path = _write_csv(tmp_path, "outliers.csv", df)
    result = run_quality_checks(path)
    assert any("outlier" in i for i in result["issues"])


def test_unreadable_file_is_critical(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("this is not,,, valid\ncsv\"\"\"data")
    result = run_quality_checks(str(path) + ".unsupported_ext")
    assert result["severity"] == "critical"
