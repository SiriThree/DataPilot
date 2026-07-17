# DataPilot

DataPilot 是一个面向 KDD Cup 2026 DataAgent-Bench 工作流的本地 DataAgent 项目。它的目标是让智能体自动读取复杂任务上下文，规划工具调用，分析结构化与半结构化数据，并最终生成可评测的 `prediction.csv`。

这个仓库已经不只是原始 starter baseline，而是一个具备完整本地闭环的 DataAgent 原型：包含模型网关、工具系统、ReAct / Multi-Agent 执行循环、验证链、确定性修复、失败分析和测试体系。

## 项目要做什么

每个 benchmark 任务通常长这样：

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

Agent 要完成的事情是：

1. 读取 `task.json` 中的问题。
2. 检查 `context/` 里的数据文件。
3. 判断该使用 SQL、Python、文档搜索还是 PDF 工具。
4. 基于上下文计算出有依据的答案。
5. 用 `answer` 工具提交紧凑的 CSV 形状答案。
6. 写出 `prediction.csv`、`trace.json` 和 `failure_analysis.json` 方便复盘。

每个任务的运行产物位于：

```text
artifacts/runs/<run_id>/<task_id>/prediction.csv
artifacts/runs/<run_id>/<task_id>/trace.json
artifacts/runs/<run_id>/<task_id>/failure_analysis.json
```

其中 `prediction.csv` 是最终答案，`trace.json` 是完整推理和工具调用轨迹。

## 当前整体架构

```text
任务输入
  -> 配置加载
  -> 模型网关
  -> 路由决策
  -> ReAct / Multi-Agent 执行循环
  -> 工具系统
  -> Step Controller
  -> prediction.csv 写出
  -> 验证链
  -> 确定性 repair
  -> failure analysis
  -> 本地 evaluator
```

## 核心模块

### 配置系统

位置：

```text
src/data_agent_baseline/config.py
```

能力：

- 加载 YAML 配置
- 自动加载本地 `.env`
- 支持 `${ENV_VAR}` 和 `${ENV_VAR:-default}`
- 支持模型后端选择
- 避免把真实 key 写进配置文件

### 模型网关

位置：

```text
src/data_agent_baseline/agents/model.py
```

当前支持：

- OpenAI-compatible API
- DeepSeek
- Ollama 本地模型

Ollama 适配器已经做了本地模型兼容：

- `think: false`
- `format: json`
- `num_predict: 2048`
- 空 `content` fallback

### Agent 执行循环

位置：

```text
src/data_agent_baseline/agents/
```

包含：

- ReAct Agent
- Multi-Agent planner / executor
- prompt 构造
- JSON 输出解析和修复
- step controller
- debugger
- verifier

Agent 不是直接猜答案，而是按照 JSON action 协议一步步调用工具。

### 工具系统

位置：

```text
src/data_agent_baseline/tools/
```

当前工具支持：

- CSV profile / read
- JSON profile / read / query
- SQLite schema inspection
- SQLite SQL 执行
- DuckDB 跨 CSV / JSON / SQLite 查询
- Python 执行
- Markdown 文档读取和搜索
- PDF 读取、搜索和表格抽取
- 最终 `answer` 提交

目前最稳定的是结构化数据分析路径：

```text
CSV / JSON / SQLite -> DuckDB SQL -> Python 辅助计算 -> answer
```

PDF 和长文档工具已经可用，但复杂 PDF 表格和长叙事文档仍需要继续通过 benchmark 样本打磨。

### 运行、验证和修复

位置：

```text
src/data_agent_baseline/run/
```

包含：

- `runner.py`：任务运行主流程
- `route_decision.py`：SQL / Python / 文档 / hybrid 路由决策
- `verification_chain.py`：5 层验证链
- `semantic_verifier.py`：问题语义风险检查
- `repair.py`：确定性修复
- `guided_retry.py`：失败后的重试提示
- `failure_analysis.py`：失败归因

当前 repair 能力包括：

- 裁剪冗余证据列
- full name 拆分为 `first_name` / `last_name`
- average monthly 候选公式重算
- event expense type 总额重算
- literal rank finish time 重算

这些 repair 都是保守规则，并且有测试覆盖。

## 当前项目进度

当前本地测试状态：

```text
28 passed, 1 skipped
```

已经完成：

- 本地环境跑通
- demo 数据集跑通
- DeepSeek 后端跑通
- Ollama 后端跑通
- CSV / JSON / SQLite 工具闭环
- Markdown 文档工具
- PDF 工具
- 本地 evaluator
- trace / summary / failure_analysis 产物
- Step Controller
- verification chain
- deterministic repair
- regression tests
- GitHub 仓库初始化

近期 benchmark 进展：

- DeepSeek `limit 10` 跑通，`overall_score = 0.985`
- `task_25` / `task_38` 的冗余列问题已修复
- `limit 20` 部分运行完成 18 个任务，partial `overall_score = 0.8889`
- `task_89` rank 语义错误已修复
- `task_163` expense type 语义错误已修复

当前下一步重点：

```text
task_169 / task_180 等 medium 大表任务的性能和超时控制
```

后续方向是：

- SQL-first 策略更强
- 避免 pandas 直接读超大表
- 加强大表采样、预聚合和超时控制
- 继续把失败样本抽象成通用 verifier / repair

## 安装环境

如果没有 `uv`：

```powershell
python -m pip install --user uv
```

安装依赖：

```powershell
python -m uv sync
```

开发和测试依赖：

```powershell
python -m uv sync --extra dev
```

## 配置模型 Key

复制 `.env.example`：

```powershell
Copy-Item .env.example .env
```

DeepSeek 示例：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_BASE=https://api.deepseek.com
```

Ollama 示例：

```env
OLLAMA_MODEL=qwen3:8b
OLLAMA_API_BASE=http://localhost:11434
```

注意：

- 真实 key 只放 `.env`
- 不要提交 `.env`
- 不要把真实 key 写进 YAML、README 或截图

## 数据目录

数据不进入 git。

本地期望结构：

```text
data/public/input/
data/public/output/
```

其中：

- `input/` 是任务输入
- `output/` 是 public demo 的 gold answer

隐藏测试环境通常只有 `input/`，不能假设一定有 `output/`。

## 常用命令

检查项目和数据状态：

```powershell
python -m uv run dabench status --config configs/react_baseline.local.yaml
```

查看一个任务：

```powershell
python -m uv run dabench inspect-task task_11 --config configs/react_baseline.local.yaml
```

运行一个任务：

```powershell
python -m uv run dabench run-task task_11 --config configs/react_baseline.local.yaml
```

运行小批量 benchmark：

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 10
```

运行全部任务：

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml
```

评估一个 run：

```powershell
python -m uv run python evaluate.py batch `
  --input-dir data/public/input `
  --output-dir data/public/output `
  --predictions-dir artifacts/runs/<run_id> `
  --only-existing
```

如果是完整 50 题 benchmark，可以去掉 `--only-existing`，让缺失 prediction 也计入失败。

## 两种模型后端怎么跑

### DeepSeek / OpenAI-compatible

配置：

```text
configs/react_baseline.local.yaml
```

命令：

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.local.yaml
```

### Ollama

配置：

```text
configs/react_baseline.ollama.example.yaml
```

先启动 Ollama：

```powershell
ollama serve
```

小批量测试：

```powershell
python -m uv run dabench run-benchmark --config configs/react_baseline.ollama.example.yaml --limit 5
```

Ollama 当前更适合作为本地连通性和 prompt regression 后端。真正追求 benchmark 分数时，优先使用 DeepSeek / OpenAI-compatible 后端。

## 查看 trace

trace 文件位置：

```text
artifacts/runs/<run_id>/<task_id>/trace.json
```

查看最近一次 run：

```powershell
Get-ChildItem artifacts\runs -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
```

查看某个 trace：

```powershell
Get-Content artifacts\runs\<run_id>\task_89\trace.json
```

只看 action 流：

```powershell
python -m uv run python -c "import json; d=json.load(open('artifacts/runs/<run_id>/task_89/trace.json', encoding='utf-8')); print([s.get('action') for s in d.get('steps', [])])"
```

## 测试

Windows PowerShell 下建议先禁用全局 pytest 插件，避免用户环境里的插件版本冲突：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m uv run --extra dev python -m pytest
```

当前期望结果：

```text
28 passed, 1 skipped
```

真实模型 smoke test：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:RUN_REAL_MODEL_TESTS = "1"
python -m uv run --extra dev python -m pytest tests/test_real_model_smoke.py -q
```

这个测试会调用真实模型 API。

## Docker / 提交入口

`main.py` 是 submission-style 入口，读取：

```text
/input
```

写出：

```text
/output
/logs/runtime.log
```

模型配置由环境变量提供：

```text
MODEL_NAME
MODEL_API_URL
MODEL_API_KEY
```

本地构建示例：

```powershell
docker build -t datapilot .
```

运行示例：

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

## 仓库提交注意事项

以下内容不应提交：

- `.env`
- `.venv/`
- `data/`
- `artifacts/runs/`
- `.pytest_cache/`
- `__pycache__/`
- 本地 `configs/react_baseline.local.yaml`

仓库保留：

- 代码
- example config
- 文档
- 测试
- `uv.lock`
- `artifacts/.gitkeep`

## 项目结构

```text
.
  configs/                  示例 YAML 配置
  docs/                     RUNBOOK 和实验记录
  scripts/                  本地辅助脚本
  src/data_agent_baseline/  主包
    agents/                 模型适配器和 Agent runtime
    benchmark/              数据集 schema 和 loader
    run/                    runner、verification、repair、analysis
    tools/                  工具系统
  tests/                    回归测试
  evaluate.py               本地 evaluator
  main.py                   submission-style 入口
  pyproject.toml            项目依赖和配置
  uv.lock                   锁定依赖
```

## 当前一句话总结

DataPilot 现在已经具备完整 DataAgent 架构，能够稳定完成本地任务运行、工具调用、答案生成、评估、trace 复盘和确定性修复。下一阶段的重点不是继续搭骨架，而是围绕 medium / hard 任务提升性能、语义稳定性和 benchmark 分数。
