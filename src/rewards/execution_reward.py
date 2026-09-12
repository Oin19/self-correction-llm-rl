"""Execution-guided reward functions."""


def compute_reward(status: str, tests_passed: int = 0, total_tests: int = 0) -> float:
    """Returns a reward score based on execution result.

    - AC -> +1.0 (all tests passed)
    - Partial pass (WA/TLE/MLE/PE) -> fraction of tests passed (0.0 to 1.0)
    - CE / RE -> -0.2 penalty for broken / non-compiling code
    """
    if status == "AC":
        return 1.0
    elif status in ("WA", "TLE", "MLE", "PE") and total_tests > 0:
        return float(tests_passed / total_tests)
    elif status in ("CE", "RE"):
        return -0.2
    return 0.0


def compute_reward_binary(status: str, tests_passed: int = 0, total_tests: int = 0) -> float:
    """Binary pass/fail reward for RQ4 ablation study (+1.0 if AC, 0.0 otherwise)."""
    return 1.0 if status == "AC" else 0.0
