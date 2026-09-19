import unittest

from src.execution.executor import PythonSandbox
from src.rewards.execution_reward import compute_partial_reward


class ExecutionRewardRegressionTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = PythonSandbox(default_timeout=2.0, max_memory_mb=1024.0)
        self.tests = [{"assertion": "assert add(1,2) == 3"}] * 5 + [
            {"assertion": "assert add(2,3) == 5"}
        ] * 5

    def test_dense_reward_full_partial_and_compile_failure(self):
        cases = {
            "full": "def add(a,b): return a+b",
            "partial": "def add(a,b): return a+b if a==1 else 0",
            "broken": "def add(a,b): return a+",
        }
        expected = {"full": 1.0, "partial": 0.5, "broken": -0.2}
        for name, code in cases.items():
            result = self.sandbox.run_tests(code, self.tests)
            reward = compute_partial_reward(result)
            self.assertAlmostEqual(reward, expected[name], places=6, msg=name)

    def test_zero_pass_nonfatal_result_is_zero(self):
        result = {"status": "WA", "passed_tests": 0, "total_tests": 10}
        self.assertEqual(compute_partial_reward(result), 0.0)

    def test_empty_suite_is_not_success(self):
        result = self.sandbox.run_tests("print('hello')", [])
        self.assertNotEqual(result.status, "AC")
        self.assertEqual(compute_partial_reward(result), -0.2)


if __name__ == "__main__":
    unittest.main()
