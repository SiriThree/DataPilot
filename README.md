# DataPilot

DataPilot 是一个面向 KDD Cup 2026 DataAgent-Bench 的本地 DataAgent 项目。它的目标是让智能体自动读取复杂任务上下文，选择合适的数据工具，完成结构化、半结构化和文档数据分析，并输出可评测的 `prediction.csv`。

当前项目已经不只是 starter baseline，而是一个具备本地闭环的 DataAgent 原型：包括模型网关、任务路由、ReAct / Multi-Agent 执行、工具系统、验证链、guided retry、确定性 repair、failure mining 和回归测试。

## 当前重点

我们现在的优化方向已经从“补单个失败任务”转向“提升泛化能力”：

- 用 `task_profile` 识别任务类型，例如 `aggregation`、`rank_lookup`、`threshold_count`、`ratio_or_percentage`。
- guided retry 根据任务类型生成更具体的重试策略。
- deterministic repair 按 reasoning pattern 组织，而不是按公开集领域命名。
- failure mining 自动聚类低分任务，并给出下一步开发建议。
- 避免继续堆 public demo 特例，优先抽象通用 verifier / solver。

## 项目能做什么

每个 benchmark 任务通常包含：

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

Agent 会完成以下流程：

1. 读取 `task.json` 中的问题。
2. 扫描 `context/` 中的 CSV、JSON、SQLite、Markdown、PDF 等文件。
3. 判断任务路线：SQL-first、Python-first、document-first 或 hybrid。
4. 调用工具完成查询、计算、文档抽取和验证。
5. 用 `answer` 工具提交 CSV 形状答案。
6. 写出 `prediction.csv`、`trace.json`、`failure_analysis.json`。

运行产物位于：

```text
artifacts/runs/<run_id>/<task_id>/
  prediction.csv
  trace.json
  failure_analysis.json
```

批量运行还会生成：

```text
artifacts/runs/<run_id>/summary.json
artifacts/runs/<run_id>/failure_taxonomy.json
```

评测和失败挖掘后还会生成：

```text
artifacts/runs/<run_id>/evaluation.json
artifacts/runs/<run_id>/failure_mining.json
artifacts/runs/<run_id>/failure_mining.md
```

## 核心架构

```text
任务输入
  -> 配置加载
  -> 模型网关
  -> 任务路由和 task_profile
  -> ReAct / Multi-Agent 执行循环
  -> 工具系统
  -> prediction.csv
  -> 验证链
  -> guided retry
  -> deterministic repair
  -> failure analysis / failure mining
  -> 本地 evaluator
```

核心目录：

```text
src/data_agent_baseline/
  agents/       模型适配器、prompt、ReAct、多智能体执行
  benchmark/    数据集 schema 和 loader
  run/          runner、route、verification、repair、retry、failure mining
  tools/        CSV / JSON / SQLite / DuckDB / Python / Doc / PDF / answer 工具
```

## 工具系统

当前工具能力包括：

- CSV profile / read
- JSON profile / read / query
- SQLite schema inspection
- SQLite SQL 执行
- DuckDB 跨 CSV / JSON / SQLite 查询
- Python 执行
- Markdown / 文本文档读取和搜索
- 长文档结构化抽取
- PDF 读取、搜索和表格抽取
- threshold grounding
- 最终 `answer` 提交

当前最稳定的数据分析路线是：

```text
profile_context -> execute_data_sql -> execute_python -> answer
```

对于大 CSV，路由会优先提示使用 DuckDB SQL，避免直接 `pandas.read_csv` 全表读取。

## Repair 去特例化

`src/data_agent_baseline/run/repairs/` 现在按通用 pattern 命名：

```text
filtered_entity_count.py
filtered_join_aggregation.py
joined_table_filter.py
rank_lookup.py
ranged_rank_lookup.py
threshold_count.py
output_shape.py
scalar_format.py
```

这一步的目的不是删除已有能力，而是把“领域补丁”逐步迁移成“通用推理模式”：

- rank lookup：处理排名字段、排序语义、附属字段查找。
- threshold count：处理阈值、群体过滤、实体级计数。
- filtered entity count：处理跨表/跨记录过滤后的实体计数。
- joined table filter：处理表连接、过滤、字段投影。
- filtered join aggregation：处理 join 后的过滤聚合。

后续新增 repair 应优先进入这些通用模式，而不是按某个 public task 或领域创建新模块。

## 安装

安装 `uv`：

```powershell
python -m pip install --user uv
```

安装依赖：

```powershell
python -m uv sync
```

安装开发依赖：

```powershell
python -m uv sync --extra dev
```

## 配置模型 Key

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填入真实 key，不要把真实 key 写进 YAML、README、截图或提交记录。

DeepSeek 示例：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com
```

Qwen / Kimi 等 OpenAI-compatible 后端也使用同样方式配置：

```env
QWEN_API_KEY=your_qwen_api_key_here
KIMI_API_KEY=your_kimi_api_key_here
```

Ollama 示例：

```env
OLLAMA_MODEL=qwen3:8b
OLLAMA_API_BASE=http://localhost:11434
```

## 数据目录

数据不进入 git。请在本地放置：

```text
data/public/input/
data/public/output/
```

其中：

- `input/` 是任务输入。
- `output/` 是 public demo 的 gold answer。

hidden test 通常只有 `input/`，不能假设一定存在 `output/`。

## 常用命令

检查项目和数据状态：

```powershell
python -m uv run dabench status --config configs/react_baseline.local.yaml
```

查看任务：

```powershell
python -m uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml
```

运行单个任务：

```powershell
python -m uv run dabench run-task task_1 --config configs/react_baseline.local.yaml
```

小批量 benchmark：

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 20
```

全量 benchmark：

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml
```

评测：

```powershell
python -m uv run python evaluate.py batch `
  --input-dir data/public/input `
  --output-dir data/public/output `
  --predictions-dir artifacts/runs/<run_id>
```

如果只跑了 `--limit N`，建议加 `--only-existing`：

```powershell
python -m uv run python evaluate.py batch `
  --input-dir data/public/input `
  --output-dir data/public/output `
  --predictions-dir artifacts/runs/<run_id> `
  --only-existing
```

失败挖掘：

```powershell
python -m uv run dabench mine-failures artifacts/runs/<run_id>
```

输出：

```text
artifacts/runs/<run_id>/failure_mining.json
artifacts/runs/<run_id>/failure_mining.md
```

`failure_mining.md` 会按低分任务、任务类型、路线、信号和领域聚类，并生成 `Development Recommendations`，用于决定下一步该补哪个 verifier / solver。

## 查看 Trace

trace 路径：

```text
artifacts/runs/<run_id>/<task_id>/trace.json
```

查看最近一次 run：

```powershell
Get-ChildItem artifacts\runs -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
```

查看某个任务的 action 流：

```powershell
python -m uv run python -c "import json; d=json.load(open('artifacts/runs/<run_id>/task_1/trace.json', encoding='utf-8')); print([s.get('action') for s in d.get('steps', [])])"
```

trace 中重点看：

- `_route_decision`
- `_route_decision.task_profile`
- `_verification`
- `_guided_retry`
- `_post_process`
- `_failure_analysis`

## 测试

推荐在 PowerShell 中运行：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m uv run --extra dev python -m pytest -q
```

当前期望结果：

```text
66 passed, 1 skipped
```

真实模型 smoke test：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:RUN_REAL_MODEL_TESTS = "1"
python -m uv run --extra dev python -m pytest tests/test_real_model_smoke.py -q
```

这个测试会调用真实模型 API。

## Docker / 提交入口

`main.py` 是 submission-style 入口：

```text
/input
/output
/logs/runtime.log
```

模型配置由环境变量提供：

```text
MODEL_NAME
MODEL_API_URL
MODEL_API_KEY
```

构建：

```powershell
docker build -t datapilot .
```

运行：

```powershell
docker run --rm `
  -e MODEL_NAME="$env:MODEL_NAME" `
  -e MODEL_API_URL="$env:MODEL_API_URL" `
  -e MODEL_API_KEY="$env:MODEL_API_KEY" `
  -v "$PWD/data/public/input:/input:ro" `
  -v "$PWD/artifacts/submission-output:/output" `
  -v "$PWD/artifacts/logs:/logs" `
  datapilot
```

## 不要提交的内容

以下内容不应提交：

- `.env`
- `.venv/`
- `data/`
- `artifacts/runs/`
- `.pytest_cache/`
- `__pycache__/`
- `configs/react_baseline.local.yaml`

仓库保留：

- 代码
- example config
- 文档
- 测试
- `uv.lock`
- `artifacts/.gitkeep`

## 当前建议工作流

现在不建议继续盲目加 repair。更稳的流程是：

1. 跑一轮 `--limit 20` benchmark。
2. 评测并生成 `evaluation.json`。
3. 运行 `dabench mine-failures`。
4. 只根据 recurring failure modes 补通用 verifier / solver。
5. 再跑 ablation，对比无 repair、通用 repair、全量 repair 的效果。

## 一句话总结

DataPilot 当前已经具备完整 DataAgent 闭环。下一阶段的重点不是继续堆公开集特例，而是通过任务路由、语义校验、guided retry、failure mining 和通用 pattern solver，提高 medium / hard 任务上的稳定性和泛化能力。
