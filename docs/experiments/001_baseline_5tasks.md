# Experiment Log

## Run: 20260502T153612Z (May 2, 2026)
- Config: configs/react_baseline.local.yaml
- Model: deepseek-chat @ api.deepseek.com/v1
- Limit: 5 tasks (first 5 from dataset)
- Duration: ~7.5 min (89.5s + others)
- Result: 5/5 succeeded, 4/5 full score (1.0)

### Task-level Results

| Task | Difficulty | Score | Steps | Verif 5/5 | Repair | Notes |
|------|-----------|-------|-------|-----------|--------|-------|
| task_11 | easy | 1.0 | 10 | PASS | No | 血栓病人筛选, hybrid_doc_table route |
| task_19 | easy | 0.0 | 4 | PASS | No | 姓名格式不匹配: 输出 full_name 列, gold 要 first_name+last_name |
| task_22 | easy | 1.0 | 10 | PASS | No | |
| task_24 | easy | 1.0 | 4 | PASS | No | |
| task_25 | easy | 1.0 | 9 | PASS | No | |

### Summary
- Success rate: 5/5 (100%)
- Score >= 1.0: 4/5 (80%)
- Avg steps: 7.4
- Avg time: ~90s/task
- Route decisions: hybrid_doc_table (task_11), python_first (task_19), sql_first (task_22), sql_first (task_24), sql_first (task_25)

### Key Findings
1. All tasks completed without timeout/crash — baseline is stable
2. 5-layer verification passed all 5 tasks — output formats are clean
3. 0 repair actions triggered — format quality is good out of the box
4. task_19 failure: semantic format mismatch (concatenated full_name vs separate first_name+last_name). Not caught by verification chain because task.json has no output contract.
5. Route decision correctly identified data types in all 5 cases
6. Signal matching worked: task_11 had JSON+doc → hybrid_doc_table; task_19 had CSV → python_first; tasks 22/24/25 had DB → sql_first

### Next Steps
- Run full 50 tasks to get baseline score distribution
- Investigate task_19: add output contract inference from question text pattern matching
- Compare easy/medium/hard/extreme score breakdown
- Profile which steps consume most time (API vs tool execution)
