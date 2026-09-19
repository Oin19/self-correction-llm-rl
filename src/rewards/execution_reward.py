"""Execution-guided reward functions."""

from typing import Any, Dict, Union
from src.execution.executor import ExecutionResult
from src.execution.status import ExecutionStatus


def compute_reward(status: str, tests_passed: int = 0, total_tests: int = 0) -> float:
    """Returns a reward score based on execution result.

    - AC -> +1.0 (all tests passed)
    - Partial pass (WA/TLE/MLE/PE) -> fraction of tests passed (0.0 to 1.0)
    - CE / RE -> -0.2 penalty for broken / non-compiling code
    """
    if status == "AC":
        return 1.0
    elif status in ("WA", "TLE", "MLE", "PE") and total_tests > 0 and tests_passed > 0:
        return float(tests_passed / total_tests)
    elif status in ("CE", "RE") and tests_passed == 0:
        return -0.2
    elif total_tests > 0 and tests_passed > 0:
        return float(tests_passed / total_tests)
    return -0.2


def compute_reward_binary(status: str, tests_passed: int = 0, total_tests: int = 0) -> float:
    """Binary pass/fail reward for RQ4 ablation study (+1.0 if AC, 0.0 otherwise)."""
    return 1.0 if status == "AC" else 0.0


def compute_binary_reward(
    result: Union[ExecutionResult, Dict[str, Any]],
    success_reward: float = 1.0,
    penalty: float = -0.2,
) -> float:
    """Computes binary execution reward (+1.0 for AC, -0.2 for any failure)."""
    status = result.status if isinstance(result, ExecutionResult) else result.get("status", "")
    if status == ExecutionStatus.AC:
        return success_reward
    return penalty


def compute_partial_reward(
    result: Union[ExecutionResult, Dict[str, Any]],
    success_reward: float = 1.0,
    penalty: float = -0.2,
) -> float:
    """Computes partial test execution reward (passed_fraction if no fatal error, else penalty)."""
    if isinstance(result, ExecutionResult):
        status = result.status
        passed = result.passed_tests
        total = result.total_tests
    else:
        status = result.get("status", "")
        passed = result.get("passed_tests", 0)
        total = result.get("total_tests", 1)

    if status == ExecutionStatus.AC:
        return success_reward

    if status in (ExecutionStatus.CE, ExecutionStatus.RE, ExecutionStatus.TLE, ExecutionStatus.MLE):
        return penalty

    if total > 0 and passed > 0:
        return float(passed / total)

    return penalty


def compute_status_aware_reward(
    result: Union[ExecutionResult, Dict[str, Any]]
) -> float:
    """Computes status-aware reward based on error severity order.
    AC (+1.0) > PE (+0.5) > WA (+0.0) > TLE/MLE (-0.1) > CE/RE (-0.2)
    """
    status = result.status if isinstance(result, ExecutionResult) else result.get("status", "")

    status_weights = {
        ExecutionStatus.AC: 1.0,
        ExecutionStatus.PE: 0.5,
        ExecutionStatus.WA: 0.0,
        ExecutionStatus.TLE: -0.1,
        ExecutionStatus.MLE: -0.1,
        ExecutionStatus.CE: -0.2,
        ExecutionStatus.RE: -0.2,
    }

    return status_weights.get(status, -0.2)

