# Experiment Log 003: Inner Shell Validation

## Date: May 3, 2026
- Config: configs/react_baseline.local.yaml
- Model: deepseek-chat
- Changes: Smart Tool (repetition detection), Advisory Prompt (escape hatches), JSON Repair, Critic (tier-1 rule checks)

---

## Before/After Comparison

### task_249 (medium): Fixed — 0.0 → 1.0

| Metric | Before | After |
|--------|--------|-------|
| Steps | 16 | 4 |
| __error__ steps | 15 | 0 |
| Repetition warnings | — | 0 |
| Score | N/A (no answer) | **1.0** |

Root cause was malformed JSON output from deepseek-chat. Fixed by:
1. `_repair_json()` — strips trailing commas, removes text after JSON closing brace
2. Advisory prompt — simpler output format, less structural overhead

### task_344 (hard): Loop broken, answer produced

| Metric | Before | After |
|--------|--------|-------|
| Steps | 16 | 16 |
| search_doc calls | 9 | 7 |
| Repetition warnings | — | 0 |
| Answer produced | No (max_steps) | **Yes** |
| Score | N/A (no answer) | 0.0 |

Agent no longer gets stuck in infinite search. After 7 searches without finding explicit normal ranges, it switched to execute_python to compute with available data and common knowledge. Answer was 2 (gold says 4), but the structural fix is: agent now self-converges instead of looping forever.

Score was 0.0 because:
- Agent used "common medical knowledge" for WBC normal range (4.0-11.0)
- Gold uses a dataset-implicit threshold that yields different results
- This is a domain knowledge / semantic gap, not a code issue

### task_200 (medium): Already worked

| Metric | Before | After |
|--------|--------|-------|
| Score | 0.0 | 0.0 |
| Steps | 5 | — |

Semantic mismatch: "total atoms" interpreted differently. Not affected by inner shell changes.

---

## What Inner Shell Fixed

| Problem | Mechanism | Result |
|---------|-----------|--------|
| Malformed LLM JSON | `_repair_json()` in `parse_model_step` | task_249: 15 errors → 0 |
| Prompt over-structuring | Advisory workflow with conditional stages | Agent skips unnecessary planning |
| No escape hatch | "3 failures → change approach", step budget warning | task_344: agent self-converges |
| No error recovery | `repetition_warning` in tool output | Detection ready, not yet triggered in tests |

## What Inner Shell Can't Fix (yet)

| Problem | Why | Next Step |
|--------|-----|-----------|
| Semantic ambiguity | Single agent can't self-review reasoning | Multi-agent Critic |
| Domain knowledge gaps | Agent uses external knowledge when docs are silent | Grounding check by Critic |
| Cross-source interpretation | "total atoms" = molecule atoms vs atom entries | Contract inference from question patterns |

---

## Net Improvement

```
               Before          After
task_249:      0.0      →     1.0    (+1.0)
task_344:      fail     →     0.0    (converges, semantic gap)
task_200:      0.0      →     0.0    (unchanged, different root cause)
task_305:      1.0      →     1.0    (unchanged)

Overall:       1/3      →     2/3 score-able, 1/3 full score
```
