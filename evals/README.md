# Evals

Tests whether the *LLM agents* behave correctly, as opposed to `tests/`
which tests deterministic logic (safety checks, cost heuristics, data
quality rules) that doesn't need a model call.

## Suites

| Suite | What it grades | How | Pass threshold |
|---|---|---|---|
| `run_router_eval.py` | Does the router classify requests correctly? | Exact match against `expected_route` | 85% |
| `run_sql_eval.py` | Does the generated SQL answer the question? | **Execution accuracy** - runs a golden query and the agent's query against the same seeded DB, compares result rows (not SQL text) | 80% |
| `run_viz_eval.py` | Is the chosen chart type reasonable? | Membership in a per-case `acceptable_chart_types` set | 75% |

Execution-accuracy grading (the SQL suite) is deliberately used instead of
comparing SQL text: many different queries can correctly answer the same
question (different column aliases, `AS` clauses, join order), so a text
diff would produce false failures. `evals/metrics.py:rows_equal` compares
row *values*, ignoring column names and row order.

## Running

```bash
# one-off setup
uv sync --no-dev
uv run python feed_db.py          # SQL eval needs the seeded demo DB

# run everything
uv run python -m evals.run_all

# or run one suite
uv run python -m evals.run_sql_eval
uv run python -m evals.run_router_eval
uv run python -m evals.run_viz_eval
```

Each run writes a JSON report to `evals/results/` (gitignored - these are
run artifacts, not fixtures) with per-case pass/fail, the actual output,
and timing. `run_all.py` exits non-zero if any suite falls below its
threshold, so it can gate a CI job.

## Adding a case

Datasets are JSONL - one JSON object per line - in `evals/datasets/`.
For SQL cases, write the golden query yourself and let execution-accuracy
grading handle the comparison rather than trying to predict the exact SQL
the agent will generate:

```json
{"id": "sql_008", "question": "How many hatchbacks are there?", "golden_sql": "SELECT COUNT(*) FROM vehicles WHERE vehicle_type = 'hatchback'"}
```

## Why this isn't in the main CI/CD gate

`.github/workflows/evals.yml` is separate from `ci-cd.yml` on purpose:
eval runs make real API calls (cost + latency) and grade probabilistic
output (a model can reasonably vary run to run), which doesn't fit a
must-pass-every-PR gate the way deterministic tests and SAST/SCA do. It
runs weekly to catch drift, on manual dispatch, and on PRs labeled
`run-evals` for changes that touch a prompt or agent graph.
