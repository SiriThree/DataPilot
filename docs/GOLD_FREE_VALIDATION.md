# Gold-free 验证与纠错边界

DataPilot 的运行时验证和纠错必须是 gold-free：`run-task` 和 `run-benchmark` 只能使用任务输入、上下文文件、模型输出、工具执行结果和 trace，不允许读取 `gold.csv` 或调用 `evaluate.py`。

## 运行时允许使用的信息

- `task.json` 中的问题、难度和上下文路径。
- `context/` 下的 CSV、JSON、SQLite、Markdown、PDF 等输入数据。
- 模型生成的 `prediction.csv`。
- `trace.json` 中的工具调用、SQL/Python 执行结果和中间观察。
- verifier 产生的格式、形状、数值、语义风险和证据一致性信号。

## 运行时不允许使用的信息

- `data/public/output/**/gold.csv`。
- `evaluate.py` 的 gold-based score。
- 任何基于 task id 直接返回答案的规则。
- 任何从离线评测结果反向写入运行时答案的逻辑。

## 分层机制

1. **格式验证**：检查 CSV 是否可读、列数/行数是否合理、空值和非法 token 是否可接受。
2. **执行验证**：检查 SQL/Python 是否成功执行、是否超时、是否有空结果或异常结果。
3. **语义契约验证**：根据题目描述推断输出类型、单位、比例/百分比、Top-K、阈值、分母等约束。
4. **证据一致性验证**：比对最终答案与 trace 中的查询结果、文档抽取结果和中间计算是否一致。
5. **纠错与重试**：只根据上述 gold-free 信号选择 guided retry 或 deterministic repair。

## 离线层允许使用 gold

以下入口属于开发评测层，可以在任务跑完后使用 gold：

- `evaluate.py`
- `dabench mine-failures`
- `tests/test_evaluate.py`
- 人工分析公开 benchmark 的 `evaluation.json`

这些结果只能用于发现模式、设计通用 verifier/solver 和做回归评测，不能让运行时直接依赖 gold 答案。
