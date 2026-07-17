from __future__ import annotations

from data_agent_baseline.agents.react import parse_model_step


def test_parse_model_step_strips_qwen_thinking_tags() -> None:
    raw_response = """
<think>
I should inspect the files first.
</think>
```json
{"thought":"Scout: inspect context.","action":"profile_context","action_input":{}}
```
"""

    step = parse_model_step(raw_response)

    assert step.thought == "Scout: inspect context."
    assert step.action == "profile_context"
    assert step.action_input == {}


def test_parse_model_step_extracts_json_after_preface() -> None:
    raw_response = (
        "Here is the next action:\n"
        '{"thought":"Formatter: final answer is ready.",'
        '"action":"answer",'
        '"action_input":{"columns":["x"],"rows":[["1"]]}}\n'
        "Done."
    )

    step = parse_model_step(raw_response)

    assert step.action == "answer"
    assert step.action_input == {"columns": ["x"], "rows": [["1"]]}
