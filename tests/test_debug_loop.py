"""Tests for multi-turn debug loop controller."""

import pytest
from src.debugging.debug_loop import DebugLoop
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus


def test_debug_loop_success():
    sandbox = PythonSandbox()
    problem = "Write a function add(a, b) that adds two numbers."
    test_cases = [{"fn_name": "add", "input": [2, 3], "expected": 5}]

    # Mock model: turn 1 returns buggy code, turn 2 returns fixed code
    def mock_model(prompt: str) -> str:
        if "previous solution failed" in prompt.lower():
            return "```python\ndef add(a, b):\n    return a + b\n```"
        return "```python\ndef add(a, b):\n    return a - b\n```"

    loop = DebugLoop(sandbox=sandbox, max_turns=3)
    session = loop.run_session(problem, test_cases, model_fn=mock_model)

    assert session["solved"] is True
    assert session["solved_turn"] == 2
    assert session["total_turns"] == 2
    assert session["final_status"] == ExecutionStatus.AC
