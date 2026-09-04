"""
Deterministic data quality checks for ETL outputs. Runs after any
extract/transform tool call finishes, before the ETL judge reviews the
run - so the judge has quality context, and severe issues (empty output,
fully-null columns) get caught even if the judge would otherwise approve
the run as "correctness-wise fine."
"""
import pandas as pd
from typing import Any, Dict

NULL_WARNING_THRESHOLD = 0.30    # 30%+ nulls in a column -> warning
NULL_CRITICAL_THRESHOLD = 0.95   # 95%+ nulls in a column -> critical
DUPLICATE_WARNING_THRESHOLD = 0.20  # 20%+ duplicate rows -> warning


def _load(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if file_path.endswith(".json"):
        return pd.read_json(file_path, lines=True)
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported file type for quality check: {file_path}")


def run_quality_checks(file_path: str) -> Dict[str, Any]:
    """
    Returns a dict: {severity, row_count, column_count, issues, report_text}
    severity is 'ok' | 'warning' | 'critical'.
    """
    try:
        df = _load(file_path)
    except Exception as e:
        return {
            "severity": "critical",
            "row_count": 0,
            "column_count": 0,
            "issues": [f"Could not read output file: {e}"],
            "report_text": f"CRITICAL: failed to read {file_path}: {e}",
        }

    issues = []
    row_count = len(df)
    column_count = len(df.columns)

    if row_count == 0:
        issues.append("Output has zero rows.")

    null_pct_by_col = (df.isna().mean()).to_dict()
    for col, pct in null_pct_by_col.items():
        if pct >= NULL_CRITICAL_THRESHOLD:
            issues.append(f"Column '{col}' is {pct:.0%} null (effectively empty).")
        elif pct >= NULL_WARNING_THRESHOLD:
            issues.append(f"Column '{col}' is {pct:.0%} null.")

    if row_count > 0:
        dup_pct = df.duplicated().mean()
        if dup_pct >= DUPLICATE_WARNING_THRESHOLD:
            issues.append(f"{dup_pct:.0%} of rows are exact duplicates.")

    # crude outlier check on numeric columns via IQR
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = ((series < lower) | (series > upper)).sum()
        outlier_pct = outlier_count / len(series)
        if outlier_pct >= 0.10:
            issues.append(f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.0%} of values, IQR method).")

    if row_count == 0 or any("effectively empty" in i or "Could not read" in i for i in issues):
        severity = "critical"
    elif issues:
        severity = "warning"
    else:
        severity = "ok"

    report_text = (
        f"{row_count} rows, {column_count} columns. "
        + (f"Issues: {'; '.join(issues)}" if issues else "No data quality issues detected.")
    )

    return {
        "severity": severity,
        "row_count": row_count,
        "column_count": column_count,
        "issues": issues,
        "report_text": report_text,
    }
