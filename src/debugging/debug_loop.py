"""Iterative execution-feedback debugging loop.
Builds the agentic loop: problem -> buggy code -> sandbox execution -> feedback -> model fix -> re-execute.
Implementation owned by Junior B.
"""

import re
from typing import Any, Callable, Dict, List, Optional
from src.execution.executor import ExecutionResult, PythonSandbox
from src.execution.status import ExecutionStatus
from src.rewards.execution_reward import compute_partial_reward


def extract_code_block(text: str) -> str:
    """Extracts python code from markdown code blocks or returns raw string."""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    pattern_generic = r"```\s*(.*?)\s*```"
    match_gen = re.search(pattern_generic, text, re.DOTALL)
    if match_gen:
        return match_gen.group(1).strip()
    return text.strip()


def format_execution_feedback(result: ExecutionResult) -> str:
    """Formats execution outcome and traceback for LLM prompt context."""
    if result.status == ExecutionStatus.AC:
        return "All test cases PASSED successfully."

    feedback = [f"Execution Status: {result.status}"]

    if result.traceback:
        feedback.append(f"Traceback / Error Details:\n{result.traceback}")
    elif result.stderr:
        feedback.append(f"Error Output (stderr):\n{result.stderr}")
    elif result.stdout:
        feedback.append(f"Output (stdout):\n{result.stdout}")

    if result.total_tests > 0:
        feedback.append(f"Passed {result.passed_tests} / {result.total_tests} test cases.")

    return "\n".join(feedback)


class DebugLoop:
    """Controls multi-turn agentic debugging sessions up to max_turns K."""

    def __init__(
        self,
        sandbox: Optional[PythonSandbox] = None,
        max_turns: int = 5,
        traceback_token_cap: int = 500,
    ):
        self.sandbox = sandbox or PythonSandbox()
        self.max_turns = max_turns
        self.traceback_token_cap = traceback_token_cap

    def build_initial_prompt(self, problem_description: str) -> str:
        return (
            f"Solve the following programming problem in Python.\n"
            f"Problem Statement:\n{problem_description}\n\n"
            f"Write clean, self-contained Python code wrapped in ```python ... ```."
        )

    def build_turn_prompt(
        self,
        problem_description: str,
        previous_code: str,
        execution_result: ExecutionResult,
    ) -> str:
        feedback = format_execution_feedback(execution_result)
        return (
            f"Problem Statement:\n{problem_description}\n\n"
            f"Previous solution failed execution.\n"
            f"Previous Code:\n```python\n{previous_code}\n```\n\n"
            f"Execution Feedback:\n{feedback}\n\n"
            f"Identify the bug, correct the code, and provide the fixed Python solution wrapped in ```python ... ```."
        )

    def run_session(
        self,
        problem_description: str,
        test_cases: List[Dict[str, Any]],
        model_fn: Callable[[str], str],
        initial_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs multi-turn debugging loop for up to max_turns."""
        history = []
        solved = False
        solved_turn = None

        current_code = initial_code

        for turn_idx in range(1, self.max_turns + 1):
            if current_code is None:
                prompt = self.build_initial_prompt(problem_description)
                raw_response = model_fn(prompt)
                current_code = extract_code_block(raw_response)
            else:
                # If we received initial code in turn 1, we execute it directly
                prompt = ""

            exec_res = self.sandbox.run_tests(current_code, test_cases)
            reward = compute_partial_reward(exec_res)

            turn_info = {
                "turn": turn_idx,
                "prompt": prompt,
                "code": current_code,
                "execution_result": exec_res.to_dict(),
                "reward": reward,
                "status": exec_res.status,
            }
            history.append(turn_info)

            if exec_res.status == ExecutionStatus.AC:
                solved = True
                solved_turn = turn_idx
                break

            # Build next turn prompt with feedback if not solved
            if turn_idx < self.max_turns:
                turn_prompt = self.build_turn_prompt(problem_description, current_code, exec_res)
                raw_response = model_fn(turn_prompt)
                current_code = extract_code_block(raw_response)

        return {
            "solved": solved,
            "solved_turn": solved_turn,
            "total_turns": len(history),
            "max_turns": self.max_turns,
            "history": history,
            "final_status": history[-1]["status"],
            "final_reward": history[-1]["reward"],
        }
