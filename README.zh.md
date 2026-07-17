<div align="center">

# DataAgent-Bench Starter Kit

[English](README.md) | 中文

[![官方网站](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo 数据集](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

面向 KDD Cup 2026 DataAgent-Bench 挑战的官方 starter kit。本项目提供一个 ReAct 风格的数据智能体 baseline：读取任务上下文、调用结构化分析工具，并输出可评测的 `prediction.csv`。

## 项目亮点

- 基于 JSON action 协议的 ReAct agent runtime，并完整记录工具调用轨迹。
- 自动判断任务路线，覆盖 SQL-first、Python-first、document-first 和混合型任务。
- 用 DuckDB 在同一个 SQL 接口中分析 CSV、JSON、SQLite 等数据源。
- 内置上下文画像、文档搜索/抽取、SQLite 查询和带超时的 Python 执行工具。
- 运行后自动做结果校验、轻量修复和预测文件标准化。
- 同时支持本地 CLI 和 Docker/评测环境中的 `main.py` 提交入口。

## 目录

- [快速开始](#快速开始)
- [数据集结构](#数据集结构)
- [配置说明](#配置说明)
- [CLI 命令](#cli-命令)
- [Agent 工具](#agent-工具)
- [输出文件](#输出文件)
- [提交运行时](#提交运行时)
- [开发与测试](#开发与测试)
- [项目结构](#项目结构)
- [相关资源](#相关资源)

## 快速开始

安装 [`uv`](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装项目依赖：

```bash
uv sync
```

从上方 badge 链接下载公开 demo 数据集，并解压到以下位置：

```text
data/public/input/
```

检查项目路径和数据集是否可见：

```bash
uv run dabench status --config configs/react_baseline.example.yaml
```

如果还没有本地配置，先创建一份：

```bash
cp -n configs/react_baseline.deepseek.example.yaml configs/react_baseline.local.yaml
```

本地 DeepSeek 测试可以直接读取 Claude 配置里的 DeepSeek token，不需要把 key 写进 yaml：

```bash
uv run python scripts/with_claude_deepseek.py --show
uv run python scripts/with_claude_deepseek.py -- \
  uv run dabench run-task task_1 --config configs/react_baseline.local.yaml
```

`configs/react_baseline.local.yaml` 仍然可以手工配置。`api_key` 支持完整环境变量占位，例如 `${OPENAI_API_KEY}`；也支持带默认值的 `${ENV_VAR:-default}`。

运行单个任务：

```bash
uv run dabench run-task task_1 --config configs/react_baseline.local.yaml
```

运行少量任务做 smoke test：

```bash
uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 5
```

## 数据集结构

公开 demo 数据集默认放在 `data/public/input/`。每个任务目录包含任务元信息和上下文目录：

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

如果公开 demo 提供标准答案，答案文件位于：

```text
data/public/output/task_<id>/gold.csv
```

Hidden test 只提供 `input/`，不要假设存在 `output/`。

`task.json` 包含：

- `task_id`
- `difficulty`
- `question`

`context/` 中可能包含 CSV、JSON、SQLite 数据库、DB 文件、Markdown/文本文件以及其他任务资源。

## 配置说明

仓库自带示例配置 `configs/react_baseline.example.yaml`。本地运行时推荐使用类似下面的配置：

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: ${OPENAI_API_KEY}
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

| 字段 | 含义 |
| --- | --- |
| `dataset.root_path` | 任务输入根目录。相对路径会按项目根目录解析。 |
| `agent.model` | 发送给 OpenAI-compatible API 的模型名称。 |
| `agent.api_base` | OpenAI-compatible API 根地址。 |
| `agent.api_key` | API key。形如 `${ENV_VAR}` 或 `${ENV_VAR:-default}` 的完整值会从环境变量读取。 |
| `agent.max_steps` | 默认 ReAct 最大步数，运行时可能按任务路线自适应调整。 |
| `agent.temperature` | 模型采样温度。 |
| `run.output_dir` | 运行产物根目录。 |
| `run.run_id` | 可选运行目录名。留空时使用 UTC 时间戳；目录已存在会报错。 |
| `run.max_workers` | `run-benchmark` 的并行 worker 数。 |
| `run.task_timeout_seconds` | 单任务墙钟超时。设为 `0` 或负数可关闭任务级超时。 |

## CLI 命令

```bash
uv run dabench <command> --config PATH [options]
```

| 命令 | 作用 | 示例 |
| --- | --- | --- |
| `status` | 查看项目路径、配置路径、数据集根目录和公开任务数量。 | `uv run dabench status --config configs/react_baseline.example.yaml` |
| `inspect-task` | 查看任务元信息，并列出 `context/` 下可访问文件。 | `uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml` |
| `run-task` | 对单个任务运行 baseline，并写出任务产物。 | `uv run dabench run-task task_1 --config configs/react_baseline.local.yaml` |
| `run-benchmark` | 批量运行配置中的数据集。 | `uv run dabench run-benchmark --config configs/react_baseline.local.yaml --limit 5` |

`run-benchmark` 支持 `--limit N`，适合快速验证。

## Agent 工具

所有传给工具的文件路径都必须是相对于当前任务 `context/` 目录的相对路径。

| 工具 | 作用 |
| --- | --- |
| `profile_context` | 扫描上下文，返回 schema、预览、统计信息、JSON key path、SQLite 元信息和文档线索。 |
| `list_context` | 列出 `context/` 下的文件和目录。 |
| `read_csv` | 读取 CSV 预览。 |
| `read_json` | 读取 JSON 预览。 |
| `query_json` | 用点分隔 key path 查询 JSON 文件。 |
| `read_doc` | 读取文本或 Markdown 预览。 |
| `search_doc` | 在文档段落中搜索关键词或短语。 |
| `extract_doc_records` | 从长叙述文档中抽取重复出现的结构化实体记录。 |
| `inspect_sqlite_schema` | 查看 SQLite/DB 文件中的表和字段。 |
| `execute_context_sql` | 对单个 SQLite/DB 文件执行只读 SQL。 |
| `execute_data_sql` | 用 DuckDB 对 `context/` 内 CSV、JSON、SQLite 等数据源执行只读 SQL。 |
| `execute_python` | 在任务 `context/` 目录内执行 Python 代码，带固定超时。 |
| `answer` | 提交最终答案表格并结束当前任务。 |

## 输出文件

每个任务会写出：

```text
artifacts/runs/<run_id>/<task_id>/
├── prediction.csv
└── trace.json
```

批量运行还会生成：

```text
artifacts/runs/<run_id>/summary.json
```

`trace.json` 中包含 agent 步骤、路线判断、校验结果、修复元信息、耗时，以及失败时的原因。

## 提交运行时

`main.py` 是 Docker/评测环境入口。它从 `/input` 读取所有任务，将预测结果和汇总写入 `/output`，并把运行日志写入 `/logs/runtime.log`。

这里是提交通道，只使用主办方注入的 OpenAI-compatible Chat Completions API 配置；不会读取 `DEEPSEEK_API_KEY`、Claude 配置或本地 yaml。

评测环境需要提供：

| 环境变量 | 含义 |
| --- | --- |
| `MODEL_NAME` | 模型名称。 |
| `MODEL_API_URL` | OpenAI-compatible API 根地址。 |
| `MODEL_API_KEY` | API key。 |

提交运行时还支持以下可选参数：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `SUBMISSION_MAX_WORKERS` | `2` | 并发处理任务数。 |
| `SUBMISSION_MAX_STEPS` | `36` | 长文档/混合路线的 ReAct 最大步数；简单 SQL/Python 路线仍会自适应使用较小步数。 |
| `SUBMISSION_TASK_TIMEOUT_SECONDS` | `600` | 单题墙钟超时。 |
| `SUBMISSION_TOTAL_TIMEOUT_SECONDS` | `42600` | 总运行预算，默认约 11h50m。 |
| `SUBMISSION_SHUTDOWN_BUFFER_SECONDS` | `300` | 12h 限制前的收尾缓冲。 |

本地构建并运行容器：

```bash
docker build -t dabench-starter .
docker run --rm \
  -e MODEL_NAME="$MODEL_NAME" \
  -e MODEL_API_URL="$MODEL_API_URL" \
  -e MODEL_API_KEY="$MODEL_API_KEY" \
  -v "$PWD/data/public/input:/input:ro" \
  -v "$PWD/artifacts/submission-output:/output" \
  -v "$PWD/artifacts/logs:/logs" \
  dabench-starter
```

## 开发与测试

运行测试：

```bash
uv run pytest
```

运行 lint：

```bash
uv run ruff check src tests main.py
```

## 项目结构

```text
.
├── configs/                  # 本地运行配置
├── data/                     # 下载后的公开 demo 数据
├── docs/                     # 实验记录和指南
├── src/data_agent_baseline/  # baseline Python 包
│   ├── agents/               # 模型适配、prompt、ReAct runtime、critic
│   ├── benchmark/            # 数据集加载和 schema
│   ├── run/                  # 路线判断、执行、校验、修复
│   └── tools/                # 上下文读取、SQL、画像、Python、answer 工具
├── tests/                    # 回归测试
├── main.py                   # 提交入口
├── Dockerfile                # 容器运行环境
└── GNN/                      # 独立的图学习实验工作区
```

## 相关资源

- 官方网站：https://dataagent.top
- Demo 数据集：https://drive.google.com/file/d/1c6u5WlFw4KV7CBRyXh5BvFYbKqxhBSbL/view
- 问题反馈：https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- Discord：https://discord.com/invite/7eFwJQN3Fx
- 微信公众号：`数据智能与分析实验室 DIAL`
