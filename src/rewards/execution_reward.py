"""Execution-guided reward functions for RL (PPO/DPO) self-correction.
Implementation owned by Junior B.
"""

from typing import Dict, Any, Union
from src.execution.executor import ExecutionResult
from src.execution.status import ExecutionStatus


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

    if total > 0:
        return float(passed / total)

    return penalty


def compute_status_aware_reward(
    result: Union[ExecutionResult, Dict[str, Any]]
) -> float:
    """Computes status-aware reward based on error severity order (Table 3).
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
