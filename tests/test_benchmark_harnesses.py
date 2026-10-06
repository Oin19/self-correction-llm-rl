import unittest

from src.execution.executor import PythonSandbox
from src.utils.test_cases import normalize_tests


class TestBenchmarkHarnesses(unittest.TestCase):
    def setUp(self):
        self.sandbox = PythonSandbox(default_timeout=2.0)

    def test_humaneval_normalization(self):
        example = {
            "prompt": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n",
            "entry_point": "add",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(1, 2) == 3\n"
                "    assert candidate(-1, 5) == 4\n"
            ),
        }
        tests = normalize_tests(example)
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0]["entry_point"], "add")
        self.assertIn("check(candidate)", tests[0]["human_eval_test"])

    def test_humaneval_passes(self):
        example = {
            "prompt": "def add(a, b):\n    return 0\n",
            "entry_point": "add",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(1, 2) == 3\n"
                "    assert candidate(-1, 5) == 4\n"
            ),
        }
        tests = normalize_tests(example)
        result = self.sandbox.run_tests(
            "def add(a, b):\n    return a + b\n",
            tests,
        )
        self.assertEqual(result.status, "AC")
        self.assertEqual(result.passed_tests, 1)
        self.assertEqual(result.total_tests, 1)

    def test_humaneval_wrong_answer(self):
        example = {
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        }
        tests = normalize_tests(example)
        result = self.sandbox.run_tests(
            "def add(a, b):\n    return 0\n",
            tests,
        )
        self.assertEqual(result.status, "WA")
        self.assertEqual(result.passed_tests, 0)
        self.assertEqual(result.total_tests, 1)

    def test_mbpp_test_list_still_works(self):
        example = {
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(5, 7) == 12",
            ]
        }
        tests = normalize_tests(example)
        result = self.sandbox.run_tests(
            "def add(a, b):\n    return a + b\n",
            tests,
        )
        self.assertEqual(result.status, "AC")
        self.assertEqual(result.passed_tests, 2)
        self.assertEqual(result.total_tests, 2)


if __name__ == "__main__":
    unittest.main()
