"""Tests for sandboxed Python executor and status classification."""

import unittest
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus, STATUSES


class TestExecutor(unittest.TestCase):
    def test_status_definitions(self):
        self.assertIn("AC", STATUSES)
        self.assertIn("WA", STATUSES)
        self.assertIn("TLE", STATUSES)
        self.assertIn("CE", STATUSES)
        self.assertIn("RE", STATUSES)

    def test_executor_syntax_error(self):
        sandbox = PythonSandbox()
        invalid_code = "def foo():\n    return 42 missing_colon"
        res = sandbox.run_single(invalid_code)
        self.assertEqual(res.status, ExecutionStatus.CE)
        self.assertEqual(res.passed_tests, 0)
        self.assertIn("SyntaxError", res.traceback)

    def test_executor_accepted_code(self):
        sandbox = PythonSandbox()
        code = "def add(a, b):\n    return a + b\n"
        test_cases = [
            {"fn_name": "add", "input": [2, 3], "expected": 5},
            {"fn_name": "add", "input": [-1, 1], "expected": 0},
        ]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.AC)
        self.assertEqual(res.passed_tests, 2)
        self.assertEqual(res.total_tests, 2)

    def test_executor_wrong_answer(self):
        sandbox = PythonSandbox()
        code = "def add(a, b):\n    return a - b\n"
        test_cases = [
            {"fn_name": "add", "input": [2, 3], "expected": 5},
        ]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.WA)
        self.assertEqual(res.passed_tests, 0)

    def test_executor_runtime_error(self):
        sandbox = PythonSandbox()
        code = "def divide(a, b):\n    return a / b\n"
        test_cases = [
            {"fn_name": "divide", "input": [10, 0], "expected": 5},
        ]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.RE)
        self.assertEqual(res.passed_tests, 0)

    def test_executor_timeout(self):
        sandbox = PythonSandbox(default_timeout=1.0)
        code = "def infinite_loop():\n    while True:\n        pass\ninfinite_loop()\n"
        res = sandbox.run_single(code)
        self.assertEqual(res.status, ExecutionStatus.TLE)
        self.assertEqual(res.passed_tests, 0)


if __name__ == "__main__":
    unittest.main()
