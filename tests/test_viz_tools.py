"""Tests for utils/viz_tools.py."""
import os
import pandas as pd
import pytest

from utils.viz_tools import render_chart


@pytest.fixture
def sample_df():
    return pd.DataFrame({"category": ["a", "b", "c"], "value": [1, 2, 3]})


@pytest.mark.parametrize("chart_type", ["bar", "line", "scatter", "pie"])
def test_render_chart_produces_png(tmp_path, monkeypatch, sample_df, chart_type):
    monkeypatch.chdir(tmp_path)
    path = render_chart(sample_df, chart_type, "category", "value", f"Test {chart_type}")
    assert os.path.exists(path)
    assert path.endswith(".png")
    assert os.path.getsize(path) > 0


def test_render_chart_rejects_unsupported_type(sample_df):
    with pytest.raises(ValueError):
        render_chart(sample_df, "surface3d", "category", "value", "bad")
