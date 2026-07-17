# DataPilot

DataPilot is a local DataAgent system for the KDD Cup 2026 DataAgent-Bench
style workflow. It reads complex task contexts, plans tool calls, analyzes
structured and semi-structured data, and writes `prediction.csv` answers that
can be evaluated against public `gold.csv` files.

This repository has evolved beyond the original starter baseline. It now has a
complete local run loop, model gateway, tool registry, verification chain,
deterministic repair layer, failure analysis, and regression tests.

## What This Project Does

Each benchmark task looks like this:

```text
task_<id>/
  task.json
  context/
    csv/
    json/
    db/
    *.md
    *.pdf
```

The agent must:

1. Read the task question.
2. Inspect the context files.
3. Choose SQL, Python, document, or PDF tools.
4. Compute a grounded answer.
5. Submit a compact CSV-shaped final answer.
6. Write trace and failure-analysis artifacts for debugging.

The expected output for each task is:

```text
artifacts/runs/<run_id>/<task_id>/prediction.csv
artifacts/runs/<run_id>/<task_id>/trace.json
artifacts/runs/<run_id>/<task_id>/failure_analysis.json
```

## Current Architecture

```text
Task input
  -> config loader
  -> model gateway
  -> route decision
  -> ReAct / multi-agent runtime
  -> tool registry
  -> step controller
  -> prediction writer
  -> verification chain
  -> deterministic repair
  -> failure analysis
  -> local evaluator
```

### Core Components

- `src/data_agent_baseline/config.py`
  - YAML config loading
  - `.env` loading
  - `${ENV_VAR}` and `${ENV_VAR:-default}` resolution
  - model provider selection

- `src/data_agent_baseline/agents/`
  - OpenAI-compatible and Ollama model adapters
  - ReAct loop
  - multi-agent planner/executor
  - step controller
  - prompt and JSON parsing hardening

- `src/data_agent_baseline/tools/`
  - context profiling
  - CSV / JSON / SQLite / DuckDB SQL tools
  - Python execution
  - Markdown search/read tools
  - PDF text, search, and table extraction
  - final answer submission

- `src/data_agent_baseline/run/`
  - route decision
  - task runner
  - verification chain
  - semantic verifier
  - deterministic repair
  - guided retry
  - failure analysis

- `evaluate.py`
  - local evaluator for single-task and batch scoring

- `tests/`
  - model-free regression suite
  - PDF tool tests
  - repair tests
  - model gateway tests
  - optional real-model smoke test

## Supported Model Backends

### OpenAI-Compatible / DeepSeek

Use `configs/react_baseline.local.yaml` or
`configs/react_baseline.deepseek.example.yaml`.

Example `.env`:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com
```

Run one task:

```powershell
python -m uv run dabench run-task task_11 --config configs/react_baseline.local.yaml
```

Run all configured tasks:

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml
```

### Ollama

Use `configs/react_baseline.ollama.example.yaml`.

Example `.env`:

```env
OLLAMA_MODEL=qwen3:8b
OLLAMA_API_BASE=http://localhost:11434
```

Start Ollama and run a smoke test:

```powershell
ollama serve
python -m uv run dabench run-benchmark --config configs/react_baseline.ollama.example.yaml --limit 5
```

Ollama is useful for local connectivity and prompt-regression checks. Current
local models are slower and less accurate than DeepSeek for benchmark scoring.

## Tool System Status

The current tool system supports:

- CSV profiling and reading
- JSON profiling and flattened DuckDB registration
- SQLite schema inspection and SQL execution
- Cross-source DuckDB SQL over CSV / JSON / SQLite
- Python execution with task-context working directory
- Markdown read/search
- PDF read/search/table extraction
- final `answer` submission

The strongest path today is structured data analysis through DuckDB SQL and
Python. PDF and long-document support are usable, but complex PDF table layouts
and long narrative extraction still need more benchmark-driven hardening.

## Verification And Repair

Every answer goes through post-processing:

1. in-memory answer validation
2. CSV readability and contract checks
3. sanity and shape checks
4. semantic risk checks
5. deterministic repair
6. failure analysis

Current deterministic repairs include:

- redundant evidence-column pruning
- full-name splitting when context exposes `first_name` / `last_name`
- average-monthly candidate recomputation
- event expense-type total recomputation
- literal-rank finish-time recomputation

These repairs are intentionally conservative and are covered by tests.

## Current Progress

Validated local state:

```text
28 passed, 1 skipped
```

Recent benchmark progress:

- `limit 10` DeepSeek run completed with `overall_score = 0.985`.
- Redundant-column failures in `task_25` and `task_38` were fixed.
- Partial `limit 20` run completed 18 tasks before manual stop, with partial
  `overall_score = 0.8889`.
- `task_89` rank semantics was fixed to use literal `rank`.
- `task_163` expense type semantics was fixed to use event type plus approved
  expense total.

Known next target:

- medium large-table tasks such as `task_169` and `task_180`, where benchmark
  runtime can become too long. The next engineering focus is SQL-first routing,
  large-table performance, and timeout control.

## Setup

Install `uv` if needed:

```powershell
python -m pip install --user uv
```

Install dependencies:

```powershell
python -m uv sync
```

For development and tests:

```powershell
python -m uv sync --extra dev
```

Create a local `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env`.

## Data

Data is intentionally not tracked in git.

Expected local layout:

```text
data/public/input/
data/public/output/
```

Download the public demo dataset from the official KDD Cup / DataAgent-Bench
starter-kit resources and unpack it under `data/`.

## Common Commands

Check status:

```powershell
python -m uv run dabench status --config configs/react_baseline.local.yaml
```

Inspect a task:

```powershell
python -m uv run dabench inspect-task task_11 --config configs/react_baseline.local.yaml
```

Run one task:

```powershell
python -m uv run dabench run-task task_11 --config configs/react_baseline.local.yaml
```

Run a benchmark subset:

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 10
```

Evaluate a run:

```powershell
python -m uv run python evaluate.py batch `
  --input-dir data/public/input `
  --output-dir data/public/output `
  --predictions-dir artifacts/runs/<run_id> `
  --only-existing
```

Run tests on Windows PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m uv run --extra dev python -m pytest
```

Optional real-model smoke test:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:RUN_REAL_MODEL_TESTS = "1"
python -m uv run --extra dev python -m pytest tests/test_real_model_smoke.py -q
```

## Trace Files

Trace files are written under each task output directory:

```text
artifacts/runs/<run_id>/<task_id>/trace.json
```

Print the action sequence for one task:

```powershell
python -m uv run python -c "import json; d=json.load(open('artifacts/runs/<run_id>/task_89/trace.json', encoding='utf-8')); print([s.get('action') for s in d.get('steps', [])])"
```

## Repository Hygiene

The following are local-only and should not be committed:

- `.env`
- `.venv/`
- `data/`
- `artifacts/`
- `.pytest_cache/`
- generated `__pycache__/`

The repository keeps code, configs, documentation, tests, and lockfiles.

## Project Layout

```text
.
  configs/                  Example YAML configs
  docs/                     Runbook and experiment notes
  scripts/                  Local helper scripts
  src/data_agent_baseline/  Main package
    agents/                 Model adapters and agent runtime
    benchmark/              Dataset schema and loading
    run/                    Runner, verification, repair, analysis
    tools/                  Tool registry and data tools
  tests/                    Regression tests
  evaluate.py               Local evaluator
  main.py                   Submission-style entry point
  pyproject.toml            Project metadata
  uv.lock                   Locked dependency graph
```

## Notes

This repository is for local development and benchmark iteration. It does not
include public data, private model keys, or generated run artifacts.
