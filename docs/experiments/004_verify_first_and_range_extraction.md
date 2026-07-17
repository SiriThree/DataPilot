# Experiment Log 004: Verify-First + Range Extraction + Adaptive Steps

## Date: May 3, 2026
- Config: configs/react_baseline.local.yaml
- Model: deepseek-chat
- Changes: Verify-First Prompt, Numeric Range Pre-extraction, Route-based Adaptive max_steps

---

## Before/After Comparison

| Task | Difficulty | Before | After | Delta | Root Cause |
|------|-----------|--------|-------|-------|-----------|
| task_11 | easy | 1.0 | 1.0 | = | Stable |
| task_19 | easy | 0.0 | 0.0 | = | Format: full_name vs first+last |
| task_22 | easy | 1.0 | 1.0 | = | Stable |
| task_24 | easy | 1.0 | 1.0 | = | Stable |
| task_25 | easy | 1.0 | 0.0 | ⬇ | API issue (0 steps, model error) |
| task_64 | easy | — | 0.0 | NEW | Format: single cell vs rows |
| task_67 | easy | — | 1.0 | NEW | Good |
| **task_200** | **medium** | **0.0** | **1.0** | **⬆ +1.0** | Verify-first fixed semantic interpretation |
| **task_249** | **medium** | **0.0** | **1.0** | **⬆ +1.0** | JSON repair + advisory prompt |
| task_305 | medium | 1.0 | 1.0 | = | Stable |
| task_344 | hard | 0.0 | 0.0 | = | API timeout (0 steps) |
| task_379 | hard | 0.0 | — | — | API timeout (600s, unreachable) |

### Summary by Difficulty

| Difficulty | Before (score ≥ 0.9) | After (score ≥ 0.9) | Change |
|-----------|---------------------|--------------------|--------|
| Easy | 4/5 (80%) | 4/7 (57%) | API flakiness caused 1 regression |
| Medium | 1/3 (33%) | 3/3 (100%) | **All medium tasks now score 1.0** |
| Hard | 0/2 (0%) | 0/2 (0%) | API timeouts — not a code issue |

---

## What Worked

### 1. Verify-First Protocol (task_200: 0.0 → 1.0)
The agent originally interpreted "total atoms" as "count all atoms in matching molecules" (answer: 4).
After verify-first: agent re-read the question, traced each value to its source, and corrected to
"count matching atom entries" (answer: 1). Gold is 1.

### 2. JSON Repair + Advisory Prompt (task_249: 0.0 → 1.0)
Previously: 15/16 `__error__` steps from malformed LLM JSON output.
After: 4 clean steps, 1.0 score. Both the JSON repair fallback and the simpler advisory prompt
contributed to this fix.

### 3. Route-Based Adaptive max_steps
No negative impact observed. Easy tasks routed to 8 steps still complete in 4-6 steps.
Hard tasks routed to 16 steps. No regressions from the step count change.

## What Didn't Work

### 1. Numeric Range Pre-extraction (task_344)
Didn't help because task_344 never got past step 0 — the initial API call timed out.
The range extraction code itself works correctly (verified with test text).
Needs a model that doesn't time out to validate.

### 2. API Reliability (multiple tasks)
Deepseek-chat drops requests intermittently:
- task_25: 0 steps (model error)
- task_344: 0 steps (timed out)
- task_379: 600s timeout (unreachable)

This is the single biggest bottleneck. The code improvements are working, but they can't
help if the model API never responds. Solution: add retry logic + shorter per-call timeout.

## Remaining Format Issues

### task_19: Concatenated vs split name columns
Question: "List the full name of members"
Agent outputs: `full_name` column with "Trent Smith"
Gold expects: `first_name,last_name` columns with "Trent,Smith"
Fix: Output contract inference from question patterns

### task_64: Single cell list vs row-per-item
Question: "List the superpowers"
Agent outputs: single cell "Agility, Super Strength, Stamina"
Gold expects: 3 rows, one per power
Fix: Output contract inference from "list" keyword in question

---

## Architecture Bottleneck Assessment

After three rounds of improvements (thick shell → inner shell → verify-first), the
bottleneck has shifted:

```
Before:  JSON malformation > search loops > semantic errors > API timeouts
After:   API timeouts > format mismatches > semantic errors
```

The code-level improvements have saturated. The remaining issues are:
1. **Model API reliability** (needs retry + shorter timeout)
2. **Output format inference** (needs contract from question patterns)
3. **Doc-heavy tasks** (need chunking-based retrieval, not keyword search)

## Next Priority

Based on 2026 paper findings (Liu & Meng, arXiv:2604.22273):
- Self-consistency (3 runs + voting) for hard tasks could help task_344 when API is responsive
- But the Markov model says this only helps when EIR ≈ 0% — our verify-first prompt should achieve this
- Output contract inference is the highest-ROI next step (fixes task_19, task_64 pattern)
