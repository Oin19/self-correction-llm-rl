"""Tests for execution reward functions."""

import pytest
from src.execution.executor import ExecutionResult
from src.execution.status import ExecutionStatus
from src.rewards.execution_reward import (
    compute_binary_reward,
    compute_partial_reward,
    compute_status_aware_reward,
)


def test_binary_reward():
    ac_res = ExecutionResult(status=ExecutionStatus.AC, passed_tests=5, total_tests=5)
    wa_res = ExecutionResult(status=ExecutionStatus.WA, passed_tests=2, total_tests=5)

    assert compute_binary_reward(ac_res) == 1.0
    assert compute_binary_reward(wa_res) == -0.2


def test_partial_reward():
    ac_res = ExecutionResult(status=ExecutionStatus.AC, passed_tests=5, total_tests=5)
    partial_res = ExecutionResult(status=ExecutionStatus.WA, passed_tests=3, total_tests=5)
    ce_res = ExecutionResult(status=ExecutionStatus.CE, passed_tests=0, total_tests=5)

    assert compute_partial_reward(ac_res) == 1.0
    assert compute_partial_reward(partial_res) == 0.6
    assert compute_partial_reward(ce_res) == -0.2


def test_status_aware_reward():
    assert compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.AC)) == 1.0
    assert compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.PE)) == 0.5
    assert compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.WA)) == 0.0
    assert compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.TLE)) == -0.1
    assert compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.CE)) == -0.2
    assert compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.RE)) == -0.2
