"""Tests for synthetic error injection engine."""

import pytest
from src.error_injection import BugInjector, BugType
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus


def test_syntax_bug_injection():
    injector = BugInjector(seed=42)
    code = """
def is_even(n):
    if n % 2 == 0:
        return True
    return False
"""
    result = injector.inject_bug(code, category=BugType.SYNTAX)
    assert result["bug_type"] == "syntax"
    sandbox = PythonSandbox()
    res = sandbox.run_single(result["buggy_code"])
    assert res.status == ExecutionStatus.CE


def test_logic_bug_injection():
    injector = BugInjector(seed=42)
    code = """
def add(a, b):
    return a + b
"""
    result = injector.inject_bug(code, category=BugType.LOGIC)
    assert result["bug_type"] == "logic"
    assert "-" in result["buggy_code"] or "return 0" in result["buggy_code"]


def test_runtime_bug_injection():
    injector = BugInjector(seed=42)
    code = """
def get_item(items, idx):
    return items[idx]
"""
    result = injector.inject_bug(code, category=BugType.RUNTIME)
    assert result["bug_type"] == "runtime"
    sandbox = PythonSandbox()
    test_cases = [{"fn_name": "get_item", "input": [[1, 2], 0], "expected": 1}]
    res = sandbox.run_tests(result["buggy_code"], test_cases)
    assert res.status in (ExecutionStatus.RE, ExecutionStatus.CE)


def test_infinite_loop_injection():
    injector = BugInjector(seed=42)
    code = """
def compute_sum(n):
    total = 0
    while n > 0:
        total += n
        n -= 1
    return total
"""
    result = injector.inject_bug(code, category=BugType.INFINITE_LOOP)
    assert result["bug_type"] == "infinite_loop"
    sandbox = PythonSandbox(default_timeout=1.0)
    test_cases = [{"fn_name": "compute_sum", "input": [5], "expected": 15}]
    res = sandbox.run_tests(result["buggy_code"], test_cases)
    assert res.status == ExecutionStatus.TLE


def test_dataset_augmentation():
    injector = BugInjector(seed=42)
    dataset = [
        {"id": 1, "solution": "def f(): return 1"},
        {"id": 2, "solution": "def g(x): return x * 2"},
    ]
    augmented = injector.create_buggy_dataset(dataset)
    assert len(augmented) == 2
    assert "buggy_code" in augmented[0]
    assert "bug_type" in augmented[0]
