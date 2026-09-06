<p align="center">
  <img src="assets/banner.svg" alt="Data Agent — Multi-Agent Data System (LangGraph + Claude)" width="100%">
</p>

<h1 align="center">🤖 Data Agent — Multi-Agent Data System (LangGraph + Claude)</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-StateGraph-1e293b?logo=graphql&logoColor=white">
  <img alt="Claude" src="https://img.shields.io/badge/Anthropic-Claude-a855f7">
  <img alt="uv" src="https://img.shields.io/badge/deps-uv-38bdf8">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-22c55e">
</p>

## Quick start

```bash
uv sync --group dev
python feed_db.py
python main.py
```

Then try a prompt like:

```text
executive: Summarize the key quality and security risks in the current dataset.
analyst: Show the likely lineage between the users and rides tables.
operator: Forecast the next 30 days of ride demand and call out any anomalies.
```

Type `exit` to leave the CLI.

A router agent classifies the user request and then dispatches to a specialist data-analysis stack. The system is result-grounded: each specialist agent works from real schema metadata and sample rows, then a final briefing layer turns the technical findings into a business-ready summary for the chosen audience.

- **SQL Analyst** — NL → SQL, cache-checked, optimizer-reviewed, judge-verified, cost-estimated, safety-checked, and executed against SQLite
- **ETL Analyst** — extracts or transforms data, then validates the output with deterministic quality checks and a correctness judge
- **Visualization Agent** — selects a chart plan, validates it against the data shape, and renders the final PNG
- **Data Quality Agent** — scans schema + sample rows to assess null rates, duplicates, drift, anomalies, and overall dataset health
- **Data Lineage Agent** — infers likely relationships between tables from keys and schema naming, then summarizes likely upstream/downstream lineage
- **Forecast Agent** — detects promising time and metric columns and recommends a forecasting strategy grounded in the actual table structure
- **Security Agent** — highlights likely privacy or compliance exposures such as PII-like fields and returns a structured risk assessment with recommendations
- **Business Summary Agent** — converts specialist results into a concise narrative for leadership, analysis, or operations; it produces a headline, executive summary, action items, and dashboard-ready recommendations
- **Audience-aware briefing flow** — the CLI accepts `executive:`, `analyst:`, and `operator:` prefixes so the same underlying analysis can be rendered for different stakeholders
- **Query Cache** — a judge-approved query for a repeat question (same text, same schema) skips `generate_sql → optimize_query → judge_query` entirely - 3 fewer LLM calls. Execution always re-runs fresh against live data even on a cache hit, so answers stay current
- **Query Optimizer** — reviews every generated SQL query for performance issues (missing LIMIT, `SELECT *`, unindexed scans) before it's judged and run
- **Cost Estimator** — deterministic (no LLM) pass using SQLite's own `EXPLAIN QUERY PLAN`: flags full table scans on large tables so an expensive query doesn't run silently
- **Pre-load source data quality check** — before `transform_load_tool` even runs, the source file it's about to load gets the same deterministic quality pass. Catches an already-bad source up front instead of only noticing after a wasted LLM-driven transform attempt (see `check_source_quality` in `agents/etl_analyst.py`)
- **Data Catalog Agent** — maintains human-readable column descriptions across every table, file-backed in `data/data_catalog.json` (`utils/data_catalog.py`)
- **Scheduled/recurring ETL runs** — `utils/scheduler.py` + `run_scheduler.py` register ETL requests for repeated execution; files are saved under `data` so schedules survive restarts
- **LLM-as-Judge** — an independent reviewer validates the output before it is trusted, and the summary layer is auto-chained after specialist findings so the technical result becomes a richer insight briefing
- **Audit Agent** — every routing decision, judge verdict, cost estimate, cache hit/miss, and quality check is logged to `data/audit_log.jsonl` (view with `python view_audit_log.py`)
- **Conversation memory** — the CLI carries prior turns forward so follow-ups like "now filter that by region" or "chart the same thing" resolve without re-explaining context
- **Clarify loop** — when the router can't confidently classify a request, it asks a follow-up instead of guessing or dead-ending

## Performance & correctness notes

A few things fixed/added in an optimization pass, worth knowing if you're extending this:

- **Schema caching** — `DatabaseUtil.schema_details()` used to re-query `sqlite_master` + `PRAGMA table_info` on every single question. It's now cached for 60s (`SCHEMA_CACHE_TTL_SECONDS`), invalidated automatically on expiry or manually via `db.invalidate_schema_cache()` after a reseed.
- **Connection safety** — `DatabaseUtil` and `cost_estimator.py` previously closed their SQLite connections only on the success path; an exception mid-query would leak the handle. Both now use `try/finally` (via a context manager in `DatabaseUtil._connection()`), so a connection is released even when the query fails.
- **ETL sandbox bug fix** — `transform_load_tool`'s exec sandbox used to empty `__builtins__` entirely, which silently broke any transform using `str()`, `round()`, `len()`, etc. - all routine in real pandas code. It now uses a curated whitelist (`SAFE_BUILTINS` in `utils/etl_tools.py`): common pure functions are available, while `open`, `eval`, `exec`, `__import__`, `compile`, `input`, `getattr`/`globals`/`locals` remain blocked. See the regression test `test_transform_can_use_common_builtins` in `tests/test_etl_tools.py`.
- **Query cache** — a judge-approved SQL query for a repeat question (same text + schema) skips `generate_sql → optimize_query → judge_query` entirely (3 fewer LLM calls). See `utils/query_cache.py`.
- **LLM call resilience** — every `llm.invoke(...)` across all four agents used to be unguarded: any transient API error (rate limit, timeout, 5xx) crashed the whole run, and there was no way to degrade to a different model if one was unavailable. `utils/llm_pick.py:invoke_with_resilience()` now wraps every LLM call site (10 total) with retry-with-exponential-backoff on transient errors and automatic fallback down the model tier chain (`high → medium → low`) on persistent ones; non-retryable errors (bad request, auth) fail fast instead of wasting retries. `main.py` also now catches a fully-exhausted failure per turn instead of crashing the whole CLI session. See `tests/test_llm_resilience.py` and the **Resilience** section below.
- **Import-time bug fix** — `agents/etl_analyst.py` used to instantiate its LLM (`pick_llm("high").bind_tools(...)`) at *module import time*, meaning the module - and anything that imported it, including the router - couldn't even be imported without API access configured. LLM instantiation is now fully lazy (inside `invoke_with_resilience`), confirmed via AST scan across all four agent files.
- **Audit log resilience** — `log_event()` used to have zero error handling: a write failure (disk full, bad path, permission error) crashed whatever agent step called it, even though logging is a side effect that should never take down the primary task. It's now best-effort (logs a stderr warning, never raises). The log file is also rotated at 10MB (`AUDIT_LOG_MAX_BYTES`, single backup generation) so it doesn't grow forever on a long-running deployment, and `read_recent()` now tails the file with a bounded `deque` instead of loading the whole thing into memory.
- **Query cache resilience + cleanup** — `set_cached_query()` had the same unguarded-write problem (now best-effort, same pattern as audit logging) and never purged expired entries from disk - every distinct question ever asked stayed in the file forever even long past its TTL. Expired entries are now purged on every write.
- **Bounded conversation memory** — `main.py`'s in-memory `history` list grew for the entire lifetime of the CLI process. `_format_history()` already capped what's sent to the router prompt at 6 turns, but the stored list itself is now also capped (`MAX_HISTORY_MESSAGES = 40`, ~20 turns) so a very long-running session doesn't accumulate memory indefinitely.
- **Query cache concurrency fix** — `main.py` is a single-user CLI, so this cache is normally only touched by one process, but it can still be hit concurrently (e.g. an eval run sharing the same `data/` directory while a CLI session is live). The read-modify-write cycle in `set_cached_query()` was a real lost-update race - **confirmed by actually reproducing it**: 30 concurrent writer processes lost 3 entries, and worse, could crash outright when two processes collided on the same shared `.tmp` filename mid-write. Fixed with `fcntl.flock` (POSIX) serializing the read-modify-write-save cycle, plus a unique per-writer temp filename. Verified with the same reproduction: 5 runs × 40 concurrent writers, 100% survival, zero crashes. See `tests/test_query_cache.py::test_concurrent_writers_lose_no_updates`. Degrades to unlocked (with a warning) on non-POSIX platforms or if lock acquisition itself fails - locking is a safety net, not something that should crash a write that would otherwise succeed.

## Resilience

`utils/llm_pick.py:invoke_with_resilience()` is what every agent calls instead of invoking a `pick_llm()` result directly. Three failure modes, handled differently on purpose:

| Failure type | Example | Behavior |
|---|---|---|
| Transient | rate limit, timeout, connection drop, 5xx | Retry same model, exponential backoff (`LLM_BASE_BACKOFF_SECONDS` → `LLM_MAX_BACKOFF_SECONDS`), up to `LLM_MAX_RETRIES_PER_MODEL` attempts |
| Persistent on this model | retries exhausted, or model not found | Fall back to the next tier down (`high → medium → low`) |
| Non-retryable | bad request, auth failure, permission denied | Fail fast to the next tier immediately - no wasted retries |

If every tier in the chain is exhausted, `invoke_with_resilience` raises, and `main.py` catches that per-turn so one bad request doesn't kill the whole CLI session. Every retry and fallback is logged (`llm_retry`, `llm_fallback`, `llm_served_by_fallback_tier` events) to `data/audit_log.jsonl`.

The retry/fallback decision logic (`_resolve_with_fallback`) is factored out from the actual Anthropic API calls specifically so it's unit-testable with fake exceptions and fake LLMs - no API key, network, or real wait time needed. See `tests/test_llm_resilience.py`.



```mermaid
flowchart TD
    U["👤 User request"] -->|conversation_history<br/>prior turns, follow-up context| R{{"🧭 Router"}}

    R -->|sql| SQL["🗄️ SQL Analyst"]
    R -->|etl| ETL["🔧 ETL Analyst"]
    R -->|visualization| VIZ["📊 Visualization Agent"]
    R -->|catalog| CAT["📚 Data Catalog Agent"]
    R -->|quality| Q["✅ Data Quality Agent"]
    R -->|lineage| L["🧬 Data Lineage Agent"]
    R -->|forecast| F["📈 Forecast Agent"]
    R -->|security| S["🔐 Security Agent"]
    R -->|summary| BRIEF["🧾 Insight Briefing"]
    R -.->|unclear intent| CLR(["❓ Clarify follow-up"])
    CLR -.-> R

    SQL -->|analysis| BRIEF
    ETL -->|analysis| BRIEF
    VIZ -->|analysis| BRIEF
    CAT -->|analysis| BRIEF
    Q -->|analysis| BRIEF
    L -->|analysis| BRIEF
    F -->|analysis| BRIEF
    S -->|analysis| BRIEF

    BRIEF -->|executive / analyst / operator| AUD["🎯 Audience-aware briefing"]
    AUD --> DASH["📊 Dashboard-ready summary"]

    subgraph SQLFLOW["SQL pipeline"]
        direction TB
        SQL --> CACHE{"Cache hit?"}
        CACHE -->|yes| COST
        CACHE -->|no| GEN["Generate SQL"] --> OPT["Optimize query"] --> JUDGE1{"Judge: correct?"}
        JUDGE1 -->|no, retry ≤2x| GEN
        JUDGE1 -->|yes, cache it| COST["💰 Cost estimate"]
        COST --> SAFE["🔒 Safety check"] --> EXEC["Execute on SQLite"]
    end

    subgraph ETLFLOW["ETL pipeline"]
        direction TB
        ETL --> SRCQ["🔎 Source quality check"]
        SRCQ --> TOOLS["Extract / Transform tools"]
        TOOLS --> DQ["✅ Output quality check"]
        DQ --> JUDGE2{"Judge: correct?"}
        JUDGE2 -->|no, retry ≤2x| ETL
    end

    subgraph VIZFLOW["Visualization"]
        direction TB
        VIZ --> PLAN["Pick chart + columns"] --> JUDGE3{"Judge: correct?"}
        JUDGE3 -->|no, retry ≤2x| PLAN
        JUDGE3 -->|yes| RENDER["Render + save PNG"]
    end

    SCHED[("⏱️ data/etl_schedule.json")] -.->|due job| ETL
    EXEC --> LOG[("📜 data/audit_log.jsonl")]
    RENDER --> LOG
    DASH --> LOG

    classDef router fill:#312e81,stroke:#a855f7,stroke-width:2px,color:#f8fafc;
    classDef agent fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef judge fill:#1e293b,stroke:#f472b6,stroke-width:2px,color:#f8fafc;
    classDef store fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#f8fafc;
    classDef user fill:#0ea5e9,stroke:#0ea5e9,color:#0f172a,font-weight:bold;

    class U user;
    class R router;
    class SQL,ETL,VIZ,CAT,Q,L,F,S,BRIEF, AUD,DASH agent;
    class CACHE,JUDGE1,JUDGE2,JUDGE3 judge;
    class LOG,SCHED store;
```

**SQL Analyst flow:** curate question → gather schema → **check cache**
(judge-approved query for this exact question+schema? skip straight to
cost/safety/execute) → generate SQL → **optimize query** (rewrite if it
has perf issues) → **judge query** (independent LLM checks correctness;
on "incorrect" verdict, feeds feedback back into generation and retries,
up to 2x; a fresh "correct" verdict gets cached) → **estimate cost**
(deterministic `EXPLAIN QUERY PLAN` check; flags full scans on large
tables, doesn't block but surfaces a 💰 note) → safety check (blocks
INSERT/UPDATE/DELETE/DROP/ALTER, forces SELECT + row limit) → execute →
summarize in plain language. Every run is written to the audit log with
the question, final SQL, optimizer notes, judge verdict, cost level, and
result preview.

**ETL Analyst flow:** tool-calling loop — LLM picks `extract_load_tool`
or `transform_load_tool`, tools execute, LLM reports back. Once the loop
ends, **check data quality** (deterministic: nulls, duplicates, outliers,
empty output) runs against whatever file was produced; a critical finding
triggers a retry with feedback (up to 1x), same pattern as the judge.
Then **judge run** reviews the full transcript *plus* the quality report
against the original request; on "incorrect" it injects feedback as a
follow-up turn and the loop resumes (up to 2x).

**Visualization flow:** load data → LLM picks chart type/columns
(structured output) → **judge spec** reviews the choice against the
request and data shape *before* anything is rendered (catches a poor
chart type or wrong columns cheaply); on "incorrect" it retries planning
with feedback (up to 2x) → render with matplotlib → save PNG.

## Setup

```bash
uv sync --group dev
cp .env.example .env           # then add your ANTHROPIC_API_KEY

uv run python feed_db.py       # seeds SQLite from data/*.csv
uv run python main.py          # interactive CLI
```

No PostgreSQL server needed — this version uses SQLite (`data/data_agent.db`)
so it runs out of the box. Swap `utils/database.py` for `psycopg2` if you
want Postgres later; the rest of the code doesn't need to change.

## Example requests

```
> Show me the top 5 users with the highest ratings
> now just show the ones with over 100 rides          # follow-up, uses memory
> Extract data from https://pokeapi.co/api/v2/pokemon and save it as CSV
> Chart average rating per vehicle type as a bar chart
> Describe what each column in the vehicles table means
> exit
```

Check what actually ran:

```bash
python view_audit_log.py 10
```

## Recurring ETL runs

Register an ETL request to run on a schedule instead of typing it at the
CLI every time:

```bash
python run_scheduler.py add "Extract data from https://pokeapi.co/api/v2/pokemon and save as CSV" --interval hourly --name pokeapi_sync
python run_scheduler.py list
python run_scheduler.py run              # run whatever's due right now, once, then exit
python run_scheduler.py run --loop 300    # or keep polling every 5 minutes until Ctrl+C
```

Jobs are file-backed in `data/etl_schedule.json` so they survive restarts;
wire `python run_scheduler.py run` up to cron, Windows Task Scheduler, or a
GitHub Actions schedule for real recurring runs. See `utils/scheduler.py`.

## Project structure

```
data_agent/
├── .github/
│   ├── workflows/ci-cd.yml      # SAST → SCA → lint → test → build → deploy
│   ├── workflows/evals.yml      # weekly/manual LLM eval suite (needs API key secret)
│   └── dependabot.yml           # continuous SCA between CI runs
├── assets/
│   └── banner.svg               # README banner
├── agents/
│   ├── data_agent.py           # router
│   ├── sql_analyst.py
│   ├── etl_analyst.py
│   ├── visualization_agent.py
│   └── data_catalog_agent.py    # maintains column descriptions across tables
├── Models/schema.py             # Pydantic state schemas
├── utils/
│   ├── database.py              # SQLite access + safety checks + catalog introspection helpers
│   ├── etl_tools.py             # extract/transform tools
│   ├── viz_tools.py             # matplotlib chart renderer
│   ├── cost_estimator.py        # deterministic EXPLAIN QUERY PLAN cost check
│   ├── data_quality.py          # deterministic data quality checks (ETL source + output)
│   ├── data_catalog.py          # file-backed column description store (data/data_catalog.json)
│   ├── scheduler.py             # recurring ETL job store + due-job logic (data/etl_schedule.json)
│   ├── query_cache.py           # skips repeat LLM calls for judge-approved queries
│   ├── file_lock.py             # shared cross-platform file locking + atomic JSON writes
│   └── llm_pick.py              # model tiers + invoke_with_resilience() (retry/fallback)
├── tests/                        # pytest suite (deterministic logic, no API key)
├── evals/                        # LLM-behavior eval suite (needs API key) - see evals/README.md
│   ├── datasets/                # JSONL eval cases per agent
│   ├── metrics.py               # execution-accuracy + other scoring
│   ├── run_router_eval.py / run_sql_eval.py / run_viz_eval.py / run_all.py
│   └── results/                 # generated reports (gitignored)
├── data/                        # sample CSVs, seeded DB, charts, extracts, audit_log.jsonl,
│                                 # query_cache.json, etl_schedule.json, data_catalog.json
├── feed_db.py                   # seed SQLite from CSVs
├── run_scheduler.py             # CLI for recurring ETL jobs (add/list/enable/disable/run)
├── view_audit_log.py            # inspect recent agent runs
├── pyproject.toml               # project metadata + uv dependency groups + bandit / ruff / pytest config
├── SECURITY.md                  # pipeline gates, known accepted SAST findings
├── project_bootstrap.py         # shared repo-root path bootstrap for local imports
└── main.py                      # CLI entry point, holds conversation memory
```

## CI/CD

Every push/PR runs SAST (Bandit + Semgrep) and SCA (pip-audit) before
anything else - see [`SECURITY.md`](SECURITY.md) for the full gate order
and the two known, reviewed SAST findings in this codebase. Run the same
checks locally before pushing:

```bash
uv sync --group dev
bandit -r agents Models utils main.py feed_db.py view_audit_log.py -c pyproject.toml
semgrep scan --config=p/python --config=p/security-audit --config=p/secrets .
uv pip compile pyproject.toml --output-file .tmp-reqs.txt
uv run pip-audit -r .tmp-reqs.txt --strict
uv run ruff check .
uv run pytest tests/
```

`tests/` (pytest, deterministic, no API key) runs on every push/PR as
part of that pipeline. `evals/` (LLM behavior, needs `ANTHROPIC_API_KEY`)
runs separately - see [`evals/README.md`](evals/README.md) for why it's
not a per-PR gate and how to run/extend it:

```bash
python feed_db.py
python -m evals.run_all
```

## Extending

Add a new agent by: (1) defining its state in `Models/schema.py`,
(2) building it as a small `StateGraph` in `agents/`, (3) adding a
route in `agents/data_agent.py`'s router prompt + conditional edges,
(4) calling `log_event(...)` from `utils/audit.py` so it shows up in
the audit trail like the others, (5) adding tests under `tests/` for
anything that doesn't require an LLM call.

Implemented since the last round of ideas: pre-load data quality checks
before ETL loads (`check_source_quality` in `agents/etl_analyst.py`),
scheduled/recurring ETL runs (`utils/scheduler.py` + `run_scheduler.py`),
a data catalog agent that maintains column descriptions across tables
(`agents/data_catalog_agent.py` + `utils/data_catalog.py`), and cost
notes for expensive queries now surface at the `medium` cost level too,
not just `high` (`agents/sql_analyst.py:generate_answer`).

Ideas for further agents: a data lineage tracker that records which
ETL runs and SQL queries fed into a given chart, alerting when a
recurring ETL job's output fails its data quality check for N
consecutive runs, and a natural-language diff between two data catalog
snapshots after a schema change.
