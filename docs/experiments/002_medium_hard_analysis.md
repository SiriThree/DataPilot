# Experiment Log 002: Medium & Hard Task Analysis

## Run: May 2, 2026
- Config: configs/react_baseline.local.yaml
- Model: deepseek-chat
- Tasks: 3 medium + 3 hard (task_420 interrupted)

---

## Results

| Task | Difficulty | Score | Steps | Status | Root Cause |
|------|-----------|-------|-------|--------|-----------|
| task_200 | medium | 0.0 | 5 | succeeded | Semantic mismatch: agent counted 4 atoms (all atoms in molecule), gold expects 1 (matching atom entries) |
| task_249 | medium | N/A | 16 | FAIL | 15/16 steps = __error__ (malformed LLM JSON output). Only profile_context succeeded. |
| task_305 | medium | 1.0 | ? | succeeded | Pure success |
| task_344 | hard | N/A | 16 | FAIL | Document search loop: searched knowledge.md 9 times for "fibrinogen normal range", never found, ran out of steps |
| task_379 | hard | N/A | 0 | FAIL | Timed out (600s). Zero steps recorded — initial LLM call likely never completed. |
| task_420 | hard | — | — | interrupted | Not completed |

**Success rate: 1/5 (20%)**, compared to 4/5 (80%) on easy tasks.

---

## Failure Pattern Analysis

### Pattern 1: LLM Malformed JSON (task_249)
- 15 consecutive `__error__` steps — the model kept producing responses that couldn't be parsed as valid JSON
- Only step 1 (profile_context) succeeded
- This is a model robustness issue with deepseek-chat; stronger models (GPT-4, Claude) handle this better
- **Fix**: Add JSON format retry, or add a `repair_json` fallback (e.g., strip malformed content, re-request)

### Pattern 2: Document Search Loop (task_344)
- Agent searched documents 9 times, read_doc 2 times, but couldn't find "fibrinogen normal range"
- The information was either not in the truncated document preview (4000 chars) or deeply buried
- Agent exhausted all 16 steps without giving up gracefully
- **Fix**: (1) Increase doc preview/chunking, (2) Add early termination if stuck in loop, (3) Fallback to "best guess + explain uncertainty" after N failed searches

### Pattern 3: API Timeout (task_379)
- 0 steps — the initial model call never returned within 600s
- Likely a model API congestion issue, not a code issue
- **Fix**: Add API-level timeout + retry, or reduce prompt length for large-context tasks

### Pattern 4: Semantic Interpretation (task_200)
- Agent correctly executed CSV+SQLite cross-source query (5 steps, efficient)
- But "total atoms" interpreted as "count of atoms in matching molecule" (4) vs gold's "count of matching atom entries" (1)
- The pipeline worked perfectly — verification passed, repair not needed — but the answer was semantically wrong
- **Fix**: Harder to fix with rules; needs better prompt disambiguation or multi-pass verification

---

## Immediate Action Items (Priority Order)

### P0: Fix Malformed JSON (target: task_249 → succeed)
1. Add JSON repair fallback in `parse_model_step()`: if parse fails, try stripping markdown, fixing common errors
2. Reduce max_steps from 16 to 12 to fail faster on hopeless tasks (saves time+ tokens)

### P0: Fix Document Search (target: task_344 → succeed)
1. Increase `read_doc` max_chars default from 4000 to 8000
2. Add `search_doc` with regex support for numeric ranges
3. After 3 failed searches, trigger "insufficient information" fallback

### P1: Defense Against Model Failures
1. Add JSON repair in `_strip_json_fence`
2. Add retry_with_backoff for API calls (task_379)
3. Route decision should flag "high_risk" tasks that need extra tool guidance

### P2: Semantic Review
1. For questions with ambiguous phrasing ("total atoms"), ask the agent to output its interpretation before computing
2. Post-answer: compare prediction shape against question keywords

---

## Key Insight

The "thick shell" works perfectly when the LLM produces valid output (task_200: verification all passed, repair not needed). But when the LLM itself fails (malformed JSON, loop, timeout), the shell can't help because it operates on the prediction.csv level. We need a **middle layer**: defense against LLM-level failures before they reach the output stage.
