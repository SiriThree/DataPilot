# DAG-DataAgent 完整实验记录

## 比赛信息

- **比赛**: KDD Cup 2026 — Data Agents for Complex Data Analysis
- **官方模型**: Qwen3.5-35B-A3B（评测系统注入）
- **开发模型**: deepseek-chat（本地调试）
- **Phase 1**: 4/28 – 5/20（A-board）, 5/20 – 5/25（B-board）
- **最终提交截止**: 5/20/2026（EoD AoE）
- **最终成绩公布**: 5/25/2026
- **提交频率**: 每天最多 1 次，Phase 1 最多 30 次
- **提交方式**: Docker 镜像 → tar.gz → Google Drive → 邮件

### 官方评分规则

- **列名不参与评分**，只看列内容（column signature）
- **列顺序无关、行顺序无关**
- **数值标准化到 2 位小数**（四舍五入）
- **空值归一为空字符串**（null/nan/NaN → ""）
- **姓名可合可分**（full_name 和 first+last 都接受）
- **冗余列惩罚**: Score = Recall - λ × (Extra/Predicted)，下限 0
- **总分**: 所有任务分数的平均值
- **评测集**: A-board ~60 tasks + B-board ~320 tasks

### 评测环境

| 资源 | 限制 |
|------|------|
| CPU | 16 vCPU, x86-64 |
| 内存 | 64 GB RAM |
| GPU | 不允许 |
| 总时限 | 12 小时（所有任务） |
| A-board 时限 | 2 小时 |
| 外网 | 完全阻断 |
| 模型 | Qwen3.5-35B-A3B（官方注入）|

---

## 架构演进

### Round 1: 厚壳 (Outer Shell)

**新增模块**:
| 模块 | 文件 | 功能 |
|------|------|------|
| Context Profiler | `tools/profiler.py` | 自动扫描 CSV/JSON/SQLite/文档 |
| Answer Verifier | `tools/verifier.py` | 输出前检查格式 |
| Local Evaluator | `evaluate.py` | Jaccard 列匹配 + 多维评分 |
| 信号路由 | `run/route_decision.py` | 关键词+文件类型决定 SQL/Python/Doc 路径 |
| 5 层验证链 | `run/verification_chain.py` | readable→contract→task_contract→sanity→shape |
| 规则修复 | `run/repair.py` | 确定性修复 CSV 格式 |
| Trace Analyzer | `tools/trace_analyzer.py` | 批量分析、失败归类 |
| 新工具 | `tools/registry.py` | +profile_context, +query_json, +search_doc |

**Prompt 改进**: 4 阶段 Plan-before-Act + 证据溯源 + ${ENV_VAR} 配置

**初始成绩**: Easy 4/5 (80%), Medium 1/3 (33%), Hard 0/2 (0%)

**失败根因**:
1. LLM 畸形 JSON (task_249): 15/16 步 error
2. 文档搜索死循环 (task_344): 找不到信息但不会停
3. API 超时 (task_379): macOS subprocess 卡死
4. 语义歧义 (task_200): "total atoms" 双重解读

---

### Round 2: 内壳 (Inner Shell)

**修改**:
| 改动 | 效果 |
|------|------|
| JSON 修复 (`_repair_json`) | 自动修尾逗号、去除多余文本 |
| 建议式 Prompt | 4 阶段改为条件触发 + escape hatches |
| 智能 Tool | 重复 3 次注入 repetition_warning |
| Critic 抽查 | 每步检查数值范围、空输出、错误文本 |
| 步数预警 | 剩 3 步时注入 "answer NOW" |

**效果**: task_249: 0→1.0, task_344: 死循环→收敛

---

### Round 3: Verify-First + 论文驱动

**来自 2026 论文**:
| 论文 | 发现 | 我们的应用 |
|------|------|-----------|
| Liu & Meng (arXiv:2604.22273) | 自我修正只在 EIR≤0.5% 时有效 | Verify-First Protocol |
| Google Research 2026 | 多 Agent 在顺序任务上退化 39-70% | 路线感知 max_steps |
| Cyclic Subtask Graphs 2026 | 数据 Agent 瓶颈在检索和证据合成 | 数值范围预提取 |

**修改**:
- Verify-First Protocol: answer 前逐值溯源，禁用外部知识
- 数值范围预提取: 扫描文档中所有阈值范围
- 路线感知步数: sql/python=8, hybrid=12, doc=24

**效果**: task_200: 0→1.0

---

### Round 4: Subprocess Bug 修复

**问题**: macOS Python `multiprocessing.Process` 用 `spawn` 模式，子进程重导模块阻塞，30% 任务 0 步超时。

**修复**: `multiprocessing.Process` → `threading.Thread`

**效果**: task_379: 0步超时→7步完成 (0.95), task_25: 0步→11步完成

---

## 最终成绩 (50/50 任务，best historical)

### 按难度

| 难度 | 数量 | Score≥0.9 | 率 | Avg Score |
|------|------|----------|-----|-----------|
| Easy | 15 | 11 | 73% | 0.81 |
| Medium | 23 | 20 | 87% | 0.90 |
| Hard | 11 | 6 | 55% | 0.57 |
| Extreme | 1 | 0 | 0% | 0.00 |
| **Total** | **50** | **37** | **74%** | **0.78** |

### 全部任务

| Task | Diff | Score | Composite | Status |
|------|------|-------|-----------|--------|
| task_11 | easy | 1.00 | 0.62 | PASS |
| task_19 | easy | 0.00 | 0.30 | FAIL — 姓名格式（官方接受） |
| task_22 | easy | 1.00 | 0.65 | PASS |
| task_24 | easy | 1.00 | 0.85 | PASS |
| task_25 | easy | 1.00 | 0.57 | PASS |
| task_26 | easy | 1.00 | 0.85 | PASS |
| task_27 | easy | 0.28 | 0.46 | FAIL |
| task_38 | easy | 0.85 | 0.94 | MARGINAL |
| task_64 | easy | 1.00 | 0.07 | PASS |
| task_67 | easy | 1.00 | 0.85 | PASS |
| task_74 | easy | 1.00 | 0.85 | PASS |
| task_75 | easy | 1.00 | 1.00 | PASS |
| task_80 | easy | 1.00 | 0.75 | PASS |
| task_86 | easy | 1.00 | 0.85 | PASS |
| task_89 | easy | 0.00 | 0.30 | FAIL |
| task_145 | medium | 1.00 | 0.85 | PASS |
| task_163 | medium | 0.00 | 0.00 | FAIL |
| task_169 | medium | 0.00 | 0.30 | FAIL |
| task_173 | medium | 1.00 | 0.85 | PASS |
| task_180 | medium | 1.00 | 0.85 | PASS |
| task_194 | medium | 1.00 | 0.85 | PASS |
| task_196 | medium | 1.00 | 0.85 | PASS |
| task_199 | medium | 1.00 | 0.85 | PASS |
| task_200 | medium | 1.00 | 0.85 | PASS (Round 3 修复) |
| task_214 | medium | 1.00 | 0.85 | PASS |
| task_218 | medium | 1.00 | 0.85 | PASS |
| task_243 | medium | 1.00 | 0.85 | PASS |
| task_249 | medium | 1.00 | 0.85 | PASS (Round 2 修复) |
| task_250 | medium | 1.00 | 0.85 | PASS |
| task_257 | medium | 1.00 | 0.85 | PASS |
| task_259 | medium | 0.85 | 0.79 | MARGINAL |
| task_261 | medium | 1.00 | 0.85 | PASS |
| task_269 | medium | 1.00 | 0.85 | PASS |
| task_283 | medium | 1.00 | 0.85 | PASS |
| task_287 | medium | 1.00 | 0.85 | PASS |
| task_292 | medium | 1.00 | 0.85 | PASS |
| task_303 | medium | 1.00 | 0.85 | PASS |
| task_305 | medium | 1.00 | 0.85 | PASS |
| task_330 | hard | 0.00 | 0.30 | FAIL — 格式 |
| task_344 | hard | 0.00 | 0.30 | FAIL — 医学知识 |
| task_349 | hard | 1.00 | 0.85 | PASS |
| task_350 | hard | 1.00 | 0.85 | PASS |
| task_352 | hard | 1.00 | 0.85 | PASS |
| task_355 | hard | 0.28 | 0.51 | FAIL — 姓名格式（官方接受） |
| task_379 | hard | 0.95 | 0.98 | PASS (Round 4 修复) |
| task_396 | hard | 0.00 | 0.30 | FAIL — 计算 |
| task_408 | hard | 0.00 | 0.30 | FAIL — 计算 |
| task_415 | hard | 1.00 | 0.85 | PASS |
| task_420 | hard | 1.00 | 0.85 | PASS (24步修复) |
| task_418 | extreme | 0.00 | 0.30 | FAIL — 步数耗尽 |

### 失败分类

| 类型 | 数量 | 任务 | 官方是否接受 |
|------|------|------|------------|
| 格式（姓名拼接） | 3 | task_19, 330, 355 | **官方接受！** |
| 计算偏差 | 4 | task_27, 38, 396, 408 | 否 |
| 语义/知识 | 3 | task_89, 344, 259 | 否 |
| 空输出 | 2 | task_163, 169 | 否 |
| 步数耗尽 | 1 | task_418 | 否 |

**关键发现：** 官方规则明确接受姓名合并格式。task_19/330/355 可能不是真正的失败。调整后实际满分率可能更高。

### 改进历程关键指标

| Round | Task | Before | After | 机制 |
|-------|------|--------|-------|------|
| Inner Shell | task_249 | 0.0 | 1.0 | JSON repair + advisory prompt |
| Verify-First | task_200 | 0.0 | 1.0 | 语义歧义修正 |
| Thread Fix | task_379 | fail | 0.95 | macOS subprocess 修复 |
| 24 Steps | task_420 | 0.0 | 1.0 | 步数增加 |

---

## 评分指标说明

| 指标 | 含义 | 竞赛是否使用 |
|------|------|------------|
| **score** | 列 Jaccard 匹配（竞赛评分） | **是** |
| value_overlap | 忽略列结构的纯值重叠率 | 否（调试用） |
| row_match | 行数匹配度 | 否（调试用） |
| **composite** | 0.4×score + 0.3×val_ov + 0.3×row_m | 否（调试用） |

composite > 0.5 通常说明推理过程正确，只是输出格式问题。

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
