"""Tests for sandboxed Python executor and status classification."""

import pytest
from src.execution.executor import PythonSandbox, ExecutionResult
from src.execution.status import ExecutionStatus, STATUSES


def test_status_definitions():
    assert "AC" in STATUSES
    assert "WA" in STATUSES
    assert "TLE" in STATUSES
    assert "CE" in STATUSES
    assert "RE" in STATUSES


def test_executor_syntax_error():
    sandbox = PythonSandbox()
    invalid_code = "def foo():\n    return 42 missing_colon"
    res = sandbox.run_single(invalid_code)
    assert res.status == ExecutionStatus.CE
    assert res.passed_tests == 0
    assert "SyntaxError" in res.traceback


def test_executor_accepted_code():
    sandbox = PythonSandbox()
    code = """
def add(a, b):
    return a + b
"""
    test_cases = [
        {"fn_name": "add", "input": [2, 3], "expected": 5},
        {"fn_name": "add", "input": [-1, 1], "expected": 0},
    ]
    res = sandbox.run_tests(code, test_cases)
    assert res.status == ExecutionStatus.AC
    assert res.passed_tests == 2
    assert res.total_tests == 2


def test_executor_wrong_answer():
    sandbox = PythonSandbox()
    code = """
def add(a, b):
    return a - b  # Bug: subtraction instead of addition
"""
    test_cases = [
        {"fn_name": "add", "input": [2, 3], "expected": 5},
    ]
    res = sandbox.run_tests(code, test_cases)
    assert res.status == ExecutionStatus.WA
    assert res.passed_tests == 0
    assert "AssertionError" in res.traceback or "AssertionError" in res.stderr


def test_executor_runtime_error():
    sandbox = PythonSandbox()
    code = """
def divide(a, b):
    return a / b
"""
    test_cases = [
        {"fn_name": "divide", "input": [10, 0], "expected": 5},
    ]
    res = sandbox.run_tests(code, test_cases)
    assert res.status == ExecutionStatus.RE
    assert res.passed_tests == 0
    assert "ZeroDivisionError" in res.traceback or "ZeroDivisionError" in res.stderr


def test_executor_timeout():
    sandbox = PythonSandbox(default_timeout=1.0)
    code = """
def infinite_loop():
    while True:
        pass
infinite_loop()
"""
    res = sandbox.run_single(code)
    assert res.status == ExecutionStatus.TLE
    assert res.passed_tests == 0
    assert "TimeoutError" in res.traceback or "timed out" in res.stderr
