# DataAgent Local Runbook

This runbook describes how to set up, run, and evaluate the local
DataAgent-Bench starter-kit workflow.

## Environment

Validated locally on Windows with:

- Python 3.12.6
- `uv` 0.11.29
- Phase 1 demo dataset under `data/public/`

If `uv` is not available in the current PowerShell session, install it with:

```powershell
python -m pip install --user uv
```

If `uv --version` still fails because the user Scripts directory is not on
`PATH`, use `python -m uv` for the commands below.

## Install Dependencies

From the project root:

```powershell
python -m uv sync
```

This creates `.venv/` and installs the project package plus dependencies from
`uv.lock`.

## Local Configuration

Create or update `configs/react_baseline.local.yaml`:

```yaml
dataset:
  root_path: data/public/input

agent:
  provider: openai_compatible
  model: ${DEEPSEEK_MODEL:-deepseek-v4-flash}
  api_base: ${DEEPSEEK_API_BASE:-https://api.deepseek.com}
  api_key: ${DEEPSEEK_API_KEY}
  max_steps: 36
  temperature: 0.0
  use_multi_agent: true

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 2
  task_timeout_seconds: 600
```

Before running tasks that call the model, create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com

OLLAMA_MODEL=qwen2.5:14b
OLLAMA_API_BASE=http://localhost:11434
```

The app automatically loads simple `KEY=value` pairs from `.env` before it
resolves `${DEEPSEEK_API_KEY}` in the YAML config. Real system environment
variables still take priority over values in `.env`.

Alternatively, set the key only for the current PowerShell session:

```powershell
$env:DEEPSEEK_API_KEY = "your_api_key_here"
```

Without a model key, `status` and `inspect-task` still work, but `run-task`
and `run-benchmark` will create failure artifacts with:

```text
Missing model API key in config.agent.api_key.
```

## Model Gateway

The runner supports two model providers through `agent.provider`:

- `openai_compatible`: OpenAI-compatible chat-completions APIs, including
  DeepSeek. This provider requires `agent.api_key`.
- `ollama`: a local Ollama chat API at `/api/chat`. This provider ignores
  `agent.api_key` and is useful for offline smoke tests or local experiments.

DeepSeek/OpenAI-compatible runs should use:

```yaml
agent:
  provider: openai_compatible
  model: ${DEEPSEEK_MODEL:-deepseek-chat}
  api_base: ${DEEPSEEK_API_BASE:-https://api.deepseek.com}
  api_key: ${DEEPSEEK_API_KEY}
```

Local Ollama runs can use `configs/react_baseline.ollama.example.yaml`:

```powershell
ollama serve
ollama pull qwen2.5:14b
python -m uv run dabench run-task task_11 --config configs/react_baseline.ollama.example.yaml
```

For Ollama, keep `run.max_workers: 1` until the local model has enough memory
and throughput for parallel tasks.

Validated local Ollama behavior:

- `qwen3:8b` can complete the ReAct loop, but on `task_11` it produced a
  low-quality placeholder-style answer before controller hardening.
- `qwen3.6:latest` is much slower on the full Agent loop and may exhaust the
  step budget on `task_11`.
- The Ollama adapter therefore runs with `think: false`, `format: json`, and
  `num_predict: 2048` to reduce thinking-output interference and truncated JSON.
- Treat Ollama as a local connectivity and prompt-regression backend first.
  Use DeepSeek/OpenAI-compatible models for benchmark scoring unless later
  experiments show a local model is accurate enough.

## Download Demo Data

The public demo dataset can be downloaded from the Google Drive file linked in
the README. The file ID currently used by the starter kit is:

```text
1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL
```

Download it with:

```powershell
python -m uv run gdown "https://drive.google.com/uc?id=1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL" -O artifacts\dataagent_phase1_demo.zip
```

Extract it to `data/`:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
Expand-Archive -LiteralPath artifacts\dataagent_phase1_demo.zip -DestinationPath data -Force
```

After extraction, the expected structure is:

```text
data/public/
├── input/
│   └── task_<id>/
│       ├── task.json
│       └── context/
└── output/
    └── task_<id>/
        └── gold.csv
```

The validated Phase 1 demo package contains:

- 50 input tasks
- 50 output tasks with `gold.csv`

Note: the README examples mention `task_1`, but this demo package does not
include `task_1`. Use an existing task such as `task_11`.

## Smoke Checks

Check project and dataset status:

```powershell
python -m uv run dabench status --config configs/react_baseline.local.yaml
```

Expected result:

```text
Public tasks: 50
Public task counts: easy=15, extreme=1, hard=11, medium=23
```

Inspect one available task:

```powershell
python -m uv run dabench inspect-task task_11 --config configs/react_baseline.local.yaml
```

Validated `task_11` question:

```text
For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.
```

## Run One Task

With `DEEPSEEK_API_KEY` configured:

```powershell
python -m uv run dabench run-task task_11 --config configs/react_baseline.local.yaml
```

This writes a timestamped run directory:

```text
artifacts/runs/<run_id>/task_11/
├── prediction.csv
├── trace.json
└── failure_analysis.json
```

If guided retry triggers, the task directory may also contain:

```text
_retry/
prediction.original.csv
trace.original.json
```

## Run a Small Benchmark

With `DEEPSEEK_API_KEY` configured:

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 5
```

This writes:

```text
artifacts/runs/<run_id>/
├── summary.json
├── failure_taxonomy.json
├── failure_taxonomy.jsonl
├── task_11/
├── task_19/
├── task_22/
├── task_24/
└── task_25/
```

## Evaluate Predictions

Evaluate a run directory against public gold answers:

```powershell
python -m uv run python evaluate.py batch `
  --input-dir data/public/input `
  --output-dir data/public/output `
  --predictions-dir artifacts/runs/<run_id> `
  --only-existing
```

The evaluator writes:

```text
artifacts/runs/<run_id>/evaluation.json
```

Use `--only-existing` for smoke tests created with `run-benchmark --limit N`.
It evaluates only tasks that already have `prediction.csv` and reports skipped
missing predictions in the summary.

For a full 50-task benchmark run, omit `--only-existing` if you want missing
predictions to count as failures.

## Stage 5 Benchmark Loop

Use DeepSeek/OpenAI-compatible models for scoring experiments:

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 10
python -m uv run python evaluate.py batch `
  --input-dir data/public/input `
  --output-dir data/public/output `
  --predictions-dir artifacts/runs/<run_id> `
  --only-existing
```

Validated Stage 5 baseline run:

```text
run_id = 20260716T173427Z
tasks attempted = 10
succeeded tasks = 10
overall_score = 0.985
```

The score loss came from redundant evidence columns:

- `task_25`: predicted `event_name,total_cost` when the question only asked
  which event.
- `task_38`: predicted transaction id plus date/amount evidence when the
  question only asked for the withdrawals.

The repair layer now includes a conservative single-field column pruning rule.
Validated follow-up checks:

```text
task_25 -> score 1.0, kept event_name
task_38 -> score 1.0, kept trans_id
```

Second Stage 5 run:

```text
run_id = 20260716T175551Z
requested limit = 20
completed before manual stop = 18
partial overall_score = 0.8889
```

The benchmark process did not finish cleanly after 18 tasks. The likely pending
tasks were `task_169` and `task_180`, both medium utility-consumption tasks with
large `yearmonth.csv` context. Treat this as a performance/timeout investigation
target before running larger batches.

Two concrete failures from the completed subset were fixed:

- `task_89`: rank semantics. The model used `positionOrder = 2` for "ranked
  second"; the repair layer now recomputes finish time from the literal `rank`
  column for the named race. Validated follow-up: score `1.0`, output
  `+16.445`.
- `task_163`: field attachment. The model grouped approved expenses by
  `expense_description`, but the requested type came from the event's `type`
  field. The repair layer now recomputes approved total for the named event and
  emits event type + total value. Validated follow-up: score `1.0`, output
  `Meeting,175.39`.

## PDF Support

The agent can profile and read PDF files in task context directories.

Supported PDF tools:

- `read_pdf`: extract bounded text from the first pages of a PDF.
- `search_pdf`: search PDF page text for a keyword or phrase.
- `extract_pdf_tables`: extract bounded tables from PDF reports.

`profile_context` recognizes `.pdf` files and reports page count, preview text,
file size, and an estimated table count for the first pages.

PDF + CSV smoke task:

```powershell
python -m uv run dabench inspect-task task_901 --config configs/react_baseline.local.yaml
python -m uv run dabench run-task task_901 --config configs/react_baseline.local.yaml
```

Evaluate the generated prediction:

```powershell
python -m uv run python evaluate.py single `
  --pred artifacts/runs/<run_id>/task_901/prediction.csv `
  --gold data/public/output/task_901/gold.csv
```

Validated result for `task_901`:

```text
score = 1.0
```

To confirm the agent actually used the PDF tool, inspect the trace:

```powershell
python -m uv run python -c "import json; d=json.load(open('artifacts/runs/<run_id>/task_901/trace.json', encoding='utf-8')); print([s.get('action') for s in d.get('steps', [])])"
```

The validated trace included `read_pdf`.

Additional PDF tool demos:

```powershell
python -m uv run dabench run-task task_902 --config configs/react_baseline.local.yaml
python -m uv run dabench run-task task_903 --config configs/react_baseline.local.yaml
```

- `task_902` is a long PDF policy search task. Validated run:
  `artifacts/runs/20260716T102306Z/task_902`, score `1.0`, trace included
  `search_pdf`.
- `task_903` is a PDF table extraction task. Validated run:
  `artifacts/runs/20260716T102352Z/task_903`, score `1.0`, trace included
  `extract_pdf_tables`.

## Run Tests

The project has a default model-free regression suite for config loading,
context profiling, SQL safety, PDF extraction, answer validation, output
verification, and local evaluation.

On Windows PowerShell, disable globally installed pytest plugins before running
tests. This avoids unrelated user-site plugin/version conflicts:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m uv run --extra dev python -m pytest
```

Expected result:

```text
28 passed, 1 skipped
```

These default tests do not call the model API and do not require
`DEEPSEEK_API_KEY`. The skipped test is the real model smoke test.

To run the real model smoke test, configure `.env` first, then run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:RUN_REAL_MODEL_TESTS = "1"
python -m uv run --extra dev python -m pytest tests/test_real_model_smoke.py -q
```

This test calls the configured model API, runs `task_11`, writes a temporary
run artifact, and evaluates the prediction against `data/public/output/task_11/gold.csv`.

Validated result:

```text
1 passed
```

## Verified Stage 1 Status

The local setup has been verified through:

- dependency sync with `python -m uv sync`
- demo dataset download and extraction
- `dabench status`
- `dabench inspect-task task_11`
- `dabench run-task task_11`
- `dabench run-benchmark --limit 5`
- `evaluate.py batch`
- model gateway selection for OpenAI-compatible APIs and Ollama
