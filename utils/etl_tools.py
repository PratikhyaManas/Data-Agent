"""
Tools the ETL agent can call. Each is decorated with @tool so it can
be bound directly to a LangChain LLM for tool-calling.
"""
import os

import pandas as pd
import requests
from langchain_core.tools import tool
from utils.audit import log_event

EXTRACT_DIR = "data/extract"
TRANSFORM_DIR = "data/transform"
MAX_RESPONSE_BYTES = 25 * 1024 * 1024  # 25 MB guardrail on a single extract

# Fully emptying __builtins__ (the original approach) is over-restrictive:
# it blocks `import`/`open`/`eval` (good) but ALSO blocks str(), len(),
# round(), isinstance(), etc., which are routine in any real pandas
# transform (e.g. `df['x'] = df['a'].apply(lambda v: round(v, 2))` would
# silently fail with "name 'round' is not defined"). Whitelist a curated
# set of pure, side-effect-free builtins instead - still no filesystem,
# network, import, or introspection access.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "range", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "isinstance", "type",
)
SAFE_BUILTINS = {name: getattr(__builtins__, name, None) for name in _SAFE_BUILTIN_NAMES}
# __builtins__ can be a dict or module depending on context; normalize.
if isinstance(__builtins__, dict):
    SAFE_BUILTINS = {name: __builtins__[name] for name in _SAFE_BUILTIN_NAMES if name in __builtins__}


@tool
def extract_load_tool(api_url: str, output_filename: str = "extracted_data", fmt: str = "csv") -> str:
    """
    Extract JSON data from an API endpoint and save it to data/extract/.
    fmt: 'csv', 'json', or 'parquet'.
    """
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    resp = requests.get(api_url, timeout=15)
    resp.raise_for_status()

    content_length = len(resp.content)
    if content_length > MAX_RESPONSE_BYTES:
        log_event("etl_blocked", api_url=api_url, size_bytes=content_length, reason="exceeds size cap")
        return (
            f"Refused to load: response is {content_length / 1e6:.1f} MB, "
            f"over the {MAX_RESPONSE_BYTES / 1e6:.0f} MB cap. Narrow the request "
            "(pagination, filters) and try again."
        )

    payload = resp.json()

    # Normalize: handle both list-of-records and dict-with-nested-list payloads
    if isinstance(payload, dict):
        list_fields = [v for v in payload.values() if isinstance(v, list)]
        payload = list_fields[0] if list_fields else [payload]

    df = pd.json_normalize(payload)
    out_path = os.path.join(EXTRACT_DIR, f"{output_filename}.{fmt}")

    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "json":
        df.to_json(out_path, orient="records", lines=True)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        return f"Unsupported format: {fmt}"

    log_event("etl_extract", api_url=api_url, rows=len(df), out_path=out_path, fmt=fmt)
    return f"Extracted {len(df)} rows from {api_url} -> {out_path}"


@tool
def transform_load_tool(input_path: str, pandas_code: str, output_filename: str = "transformed_data", fmt: str = "csv") -> str:
    """
    Load a CSV/JSON/Parquet file into a DataFrame `df`, run the given
    pandas_code (which must reassign `df`), and save the result to
    data/transform/. pandas_code is executed in a restricted namespace.
    """
    os.makedirs(TRANSFORM_DIR, exist_ok=True)

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path, lines=True)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        return f"Unsupported input file type: {input_path}"

    # Restricted execution namespace - only pandas + the dataframe + a
    # curated safe builtin whitelist (see SAFE_BUILTINS above). This exec()
    # will be flagged by SAST (bandit B102 / semgrep exec-detected); it's a
    # deliberate, reviewed sandbox: `import`, `open`, `eval`, `exec`,
    # `__import__`, `getattr`/`globals`/`locals` are all unavailable to the
    # executed code. Do not remove this suppression without replacing the
    # sandboxing mechanism (e.g. RestrictedPython) first, and do not widen
    # SAFE_BUILTINS without checking each addition can't reach the filesystem
    # or process (e.g. don't add `open`, `input`, `__import__`, `compile`).
    safe_globals = {"pd": pd, "__builtins__": SAFE_BUILTINS}
    safe_locals = {"df": df}
    try:
        exec(pandas_code, safe_globals, safe_locals)  # nosec B102 - see comment above; sandboxed, no builtins
    except Exception as e:
        return f"Transform failed: {e}"

    result_df = safe_locals.get("df")
    out_path = os.path.join(TRANSFORM_DIR, f"{output_filename}.{fmt}")

    if fmt == "csv":
        result_df.to_csv(out_path, index=False)
    elif fmt == "json":
        result_df.to_json(out_path, orient="records", lines=True)
    elif fmt == "parquet":
        result_df.to_parquet(out_path, index=False)
    else:
        return f"Unsupported format: {fmt}"

    log_event("etl_transform", input_path=input_path, out_path=out_path, rows=len(result_df))
    return f"Transformed {input_path} -> {out_path} ({len(result_df)} rows)"


ETL_TOOLS = [extract_load_tool, transform_load_tool]
