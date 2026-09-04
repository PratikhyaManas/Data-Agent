"""
Tests for utils/etl_tools.py. Focused on the size guardrail and the
transform sandbox, since those are the security-relevant pieces.
Does NOT make real network calls (no CI environment should depend on
external API availability).
"""
import pandas as pd
import pytest

from utils.etl_tools import transform_load_tool, MAX_RESPONSE_BYTES, SAFE_BUILTINS


def test_size_cap_constant_is_reasonable():
    # Guards against someone accidentally setting this to unlimited
    assert 0 < MAX_RESPONSE_BYTES <= 100 * 1024 * 1024


def test_safe_builtins_excludes_dangerous_names():
    for dangerous in ("open", "eval", "exec", "__import__", "compile", "input", "getattr", "globals", "locals"):
        assert dangerous not in SAFE_BUILTINS


def test_safe_builtins_includes_common_pure_functions():
    for safe in ("str", "int", "float", "round", "len", "abs", "min", "max", "sorted"):
        assert safe in SAFE_BUILTINS


def test_transform_load_applies_pandas_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "input.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(src, index=False)

    result = transform_load_tool.invoke({
        "input_path": str(src),
        "pandas_code": "df['c'] = df['a'] + df['b']",
        "output_filename": "out",
        "fmt": "csv",
    })

    assert "Transformed" in result
    out_df = pd.read_csv(tmp_path / "data" / "transform" / "out.csv")
    assert list(out_df["c"]) == [5, 7, 9]


def test_transform_can_use_common_builtins(tmp_path, monkeypatch):
    """
    Regression test: an earlier version emptied __builtins__ entirely,
    which silently broke any transform using str(), round(), len(), etc.
    - all routine in real pandas code. This confirms the whitelist fix.
    """
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "input.csv"
    pd.DataFrame({"a": [1.2345, 2.3456, 3.4567]}).to_csv(src, index=False)

    result = transform_load_tool.invoke({
        "input_path": str(src),
        "pandas_code": "df['label'] = df['a'].apply(lambda v: 'val_' + str(round(v, 1)))",
        "output_filename": "out",
        "fmt": "csv",
    })

    assert "Transformed" in result, result
    out_df = pd.read_csv(tmp_path / "data" / "transform" / "out.csv")
    assert out_df["label"].tolist() == ["val_1.2", "val_2.3", "val_3.5"]


def test_transform_sandbox_blocks_builtins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "input.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)

    # __import__ isn't in SAFE_BUILTINS, so attempts to reach outside
    # pandas (e.g. open a file, import os) should still fail.
    result = transform_load_tool.invoke({
        "input_path": str(src),
        "pandas_code": "import os\ndf['x'] = 1",
        "output_filename": "out",
        "fmt": "csv",
    })
    assert "Transform failed" in result


def test_transform_sandbox_blocks_open(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "input.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)

    result = transform_load_tool.invoke({
        "input_path": str(src),
        "pandas_code": "f = open('/etc/passwd')\ndf['x'] = 1",
        "output_filename": "out",
        "fmt": "csv",
    })
    assert "Transform failed" in result


def test_transform_rejects_unsupported_input_type(tmp_path):
    result = transform_load_tool.invoke({
        "input_path": "data.xyz",
        "pandas_code": "pass",
        "output_filename": "out",
        "fmt": "csv",
    })
    assert "Unsupported input file type" in result
