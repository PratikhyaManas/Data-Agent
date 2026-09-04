# Security

## Pipeline gates

Every push and PR runs through `.github/workflows/ci-cd.yml` in this order:

```
SAST (Bandit + Semgrep)  ┐
SCA (pip-audit)          ├─► lint ─► test ─► build ─► deploy (main only, manual approval)
lint (Ruff)               ┘
```

- **SAST and SCA must both pass before tests even run.** There's no value
  running the test suite against code or dependencies already known to be
  unsafe.
- **Build only runs after tests pass.** Deploy only runs on `main` and
  requires a manual approval via a GitHub Environment (`production`) - add
  required reviewers under repo Settings → Environments.
- SARIF results from Bandit and Semgrep upload to the repo's **Security**
  tab (Code scanning alerts), not just the Actions log.

## What SAST covers

- **Bandit** — Python-specific static analysis (hardcoded secrets, `exec`/`eval`
  usage, weak crypto, SQL string construction, unsafe deserialization, etc.)
- **Semgrep** — broader rule-based scanning (`p/python`, `p/security-audit`,
  `p/secrets`) - catches patterns Bandit's Python-only ruleset misses.

## What SCA covers

- **pip-audit** — checks the resolved dependency set from the uv-managed
  project config against the OSV/PyPI vulnerability databases at PR time;
  `--strict` fails the build on any known CVE in a pinned or resolved dependency.
- **Dependabot** (`.github/dependabot.yml`) — weekly scan that opens a PR
  automatically when a *new* advisory is disclosed for something already
  merged, so vulnerabilities found after the fact don't sit unpatched.

## Known, reviewed risk acceptances

Two spots in this codebase will trip SAST rules by design. Both are
commented in place with `# nosec <rule-id>` and a justification - review
the comment before touching either:

1. **`utils/etl_tools.py:transform_load_tool`** — uses `exec()` to run
   LLM-generated pandas code (Bandit B102). Mitigated by a curated
   `__builtins__` whitelist (`SAFE_BUILTINS`): common pure functions
   (`str`, `round`, `len`, etc.) are available so ordinary transforms
   don't silently break, but `open`, `eval`, `exec`, `__import__`,
   `compile`, `input`, `getattr`/`globals`/`locals` are not. **Residual
   risk, accepted:** this is a builtins whitelist, not a full sandbox -
   it doesn't stop dunder-attribute traversal (e.g. reaching `object`
   via `().__class__.__bases__`), which Python's exec model can't block
   through `__builtins__` alone. The threat model here is "LLM instructed
   to write a pandas transform," not "adversarial code trying to escape
   the sandbox." If that threat model changes (e.g. this ever accepts
   pandas_code from an untrusted end user rather than an LLM following
   the ETL agent's system prompt), replace `exec()` with a real sandbox
   (RestrictedPython, a subprocess with no filesystem/network access, or
   a WASM runtime) before removing this note.
2. **`utils/database.py:run_query`** — executes LLM-generated SQL directly
   (Bandit/Semgrep SQL-injection rules). This is inherent to a NL-to-SQL
   agent - there's no fixed query to parameterize against. Mitigated by
   layered controls instead: `is_query_safe()` requires a `SELECT` and blocks
   destructive/DDL keywords, a `LIMIT` is force-appended, and the SQL Analyst
   agent runs an LLM-as-judge correctness pass before this function is ever
   called (see `agents/sql_analyst.py`).

If you add a new `# nosec` suppression anywhere, it needs the same treatment:
a comment explaining *why* the pattern is safe here, not just that it's
suppressed.

## Reporting a vulnerability

Open a private security advisory on this repo (Security tab → "Report a
vulnerability") rather than a public issue.
