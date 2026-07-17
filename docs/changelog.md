# DAG-DataAgent 改进记录

## 技术栈

- **模型**: deepseek-chat (DeepSeek API, OpenAI 兼容协议)
- **框架**: 基于 HKUSTDial 官方 starter kit 二次开发
- **环境**: macOS, Python 3.13, uv 包管理

---

## Round 1: 厚壳 (Outer Shell)

### 新增模块

| 模块 | 文件 | 功能 |
|------|------|------|
| Context Profiler | `tools/profiler.py` | 自动扫描 CSV/JSON/SQLite/文档，输出列名、类型、行数、缺失率、表结构 |
| Answer Verifier | `tools/verifier.py` | 输出前检查：冗余列、数值精度、日期格式、空值处理 |
| Local Evaluator | `evaluate.py` | Jaccard 列签名匹配 + 多维细粒度评分 |
| 信号路由 | `run/route_decision.py` | 关键词匹配 + 文件类型打分，自动选 SQL/Python/Doc/Hybrid 路径 |
| 5 层验证链 | `run/verification_chain.py` | readable → contract → task_contract → sanity → shape |
| 规则修复 | `run/repair.py` | 确定性修复动作：normalize CSV、round 数值 |
| Trace Analyzer | `tools/trace_analyzer.py` | 批量 trace 分析、失败归类、案例生成 |
| 新工具 | `tools/registry.py` | +profile_context、+query_json、+search_doc |

### Prompt 改进
- 4 阶段 Plan-before-Act 工作流
- 证据溯源要求
- `${ENV_VAR}` 环境变量注入（不再硬编码 API key）

### 效果
- Easy: 4/5 (80%)
- Medium: 1/3 (33%)
- Hard: 0/2 (0%)

### 失败根因分析
1. **LLM 畸形 JSON** (task_249): 15/16 步输出错误
2. **文档搜索死循环** (task_344): 搜不到信息但不会放弃
3. **API 超时** (task_379): subprocess 卡死导致 0 步
4. **语义歧义** (task_200): "total atoms" 有两种解读

---

## Round 2: 内壳 (Inner Shell)

### 修改内容

| 改动 | 文件 | 效果 |
|------|------|------|
| JSON 修复 | `agents/react.py` | 自动修尾逗号、去除 JSON 后多余文本 |
| 建议式 Prompt | `agents/prompt.py` | 4 阶段改为条件触发，加 escape hatches |
| 智能 Tool | `tools/registry.py` | 同一动作重复 3 次注入 repetition_warning |
| Critic 抽查 | `agents/critic.py` | 每步后检查数值范围、空输出、错误文本 |
| 步数预警 | `agents/react.py` | 剩余 3 步时注入 "answer NOW" |

### 效果
- task_249: 0.0 → 1.0 (JSON 修复 + advisory prompt)
- task_344: fail → 收敛 (不再死循环，但语义偏差)
- task_200: 仍 0.0 (语义歧义，需其他修复)

---

## Round 3: Verify-First + 论文驱动的改进

### 来自 2026 年论文的启发

| 论文 | 关键发现 | 我们的改动 |
|------|---------|-----------|
| Liu & Meng 2026 (arXiv:2604.22273) | 自我修正只在 EIR≤0.5% 时有效；Verify-First prompt 将 EIR 降至 0% | Verify-First Protocol |
| Google Research 2026 | 多 Agent 在顺序任务上退化 39-70%；架构应匹配任务类型 | 路线感知 max_steps |
| Cyclic Subtask Graphs 2026 | Finance-Agent 瓶颈在检索和证据合成 | 数值范围预提取 |

### 修改内容

| 改动 | 文件 | 效果 |
|------|------|------|
| Verify-First Protocol | `agents/prompt.py` | answer 前强制逐值溯源，禁用未确认的外部知识 |
| 数值范围预提取 | `tools/profiler.py` | 扫描文档中所有 "between X and Y"、"normal: A-B" 等范围模式 |
| 路线感知步数 | `run/runner.py` | sql/python=8步、hybrid=12步、doc=16步 |

### 效果
- task_200: 0.0 → 1.0 (Verify-First 修正了 "total atoms" 的理解)
- task_249: 保持 1.0
- task_344: 仍 0.0 (医学领域知识盲区，非代码问题)

---

## Round 4: Subprocess Bug 修复

### 问题
macOS Python 3.8+ 默认 `multiprocessing` 用 `spawn` 模式，子进程重新导入模块时阻塞，导致 ~30% 任务 0 步超时。

### 修复
```python
# Before: multiprocessing.Process (macOS spawn 卡死)
process = multiprocessing.Process(target=..., args=...)

# After: threading.Thread (共享内存, 无 fork)
thread = threading.Thread(target=..., daemon=True)
```

### 文件
`run/runner.py` — `_run_single_task_with_timeout()`

### 效果
task_379: 0 步超时 → 7 步完成，score=0.95

---

## 当前稳定成绩

| 难度 | 数量 | Score≥0.9 | 率 |
|------|------|----------|-----|
| Easy | 7 | 6 | 86% |
| Medium | 3 | 3 | 100% |
| Hard | 2 | 1 | 50% |
| **Total** | **12** | **10** | **83%** |

### 剩余失败

| 任务 | Score | 根因 | 解决方向 |
|------|-------|------|---------|
| task_19 | 0.0 | full_name vs first+last 格式不匹配 | Output contract 推断 |
| task_344 | 0.0 | 医学阈值用外部知识而非数据分布 | 数据驱动阈值 or self-consistency |

---

## 架构总览

```
                    ┌──────────────────────┐
                    │  Task Input            │
                    └──────────┬───────────┘
                               │
               ┌───────────────▼───────────────┐
               │  信号路由 (route_decision)      │
               │  SQL/Python/Doc/Hybrid 自动选   │
               └───────────────┬───────────────┘
                               │
               ┌───────────────▼───────────────┐
               │  ReAct 循环 (Inner Shell)      │
               │  - 建议式 prompt + escape hatch │
               │  - 智能 Tool (去重记忆)         │
               │  - Critic 每步抽查              │
               │  - JSON repair fallback        │
               │  - 步数预警                     │
               └───────────────┬───────────────┘
                               │ prediction.csv
               ┌───────────────▼───────────────┐
               │  Outer Shell                   │
               │  5 层验证 → 规则修复 → 再验证   │
               └──────────────────────────────┘
```
