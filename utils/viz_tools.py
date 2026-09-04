"""
Chart rendering helper. Kept separate from the LLM logic so the
visualization agent just needs to produce a ChartSpecSchema and hand
it here.
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd

CHART_DIR = "data/charts"


def render_chart(df: pd.DataFrame, chart_type: str, x_column: str, y_column: str, title: str) -> str:
    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "bar":
        ax.bar(df[x_column], df[y_column])
    elif chart_type == "line":
        ax.plot(df[x_column], df[y_column], marker="o")
    elif chart_type == "scatter":
        ax.scatter(df[x_column], df[y_column])
    elif chart_type == "pie":
        ax.pie(df[y_column], labels=df[x_column], autopct="%1.1f%%")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    ax.set_title(title)
    if chart_type != "pie":
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:50]
    out_path = os.path.join(CHART_DIR, f"{safe_title}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
