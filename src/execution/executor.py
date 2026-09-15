"""Sandboxed execution interface for running Python code safely."""

import ast
import os
import sys
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.execution.status import ExecutionStatus

try:
    import resource
except ImportError:
    resource = None


def run_code(code: str, timeout: int = 3) -> dict:
    """Safely run Python code and return status + output.

    Status codes:
    - AC: All Correct (exit code 0)
    - WA: Wrong Answer (handled when comparing expected output)
    - TLE: Time Limit Exceeded
    - MLE: Memory Limit Exceeded
    - CE: Compilation / Syntax Error
    - RE: Runtime Error (crash / exception)
    """
    python_cmd = sys.executable if sys.executable else ("python3" if os.name != "nt" else "python")

    def preexec_limit():
        if resource is not None and os.name != "nt":
            try:
                # 1 GB memory limit
                resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))
            except Exception:
                pass

    try:
        result = subprocess.run(
            [python_cmd, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec_limit if (resource is not None and os.name != "nt") else None,
        )

        if result.returncode == 0:
            return {"status": "AC", "output": result.stdout, "traceback": ""}
        else:
            tb = result.stderr
            if "SyntaxError" in tb or "IndentationError" in tb:
                return {"status": "CE", "output": "", "traceback": tb}
            elif "MemoryError" in tb:
                return {"status": "MLE", "output": "", "traceback": "Memory limit exceeded"}
            else:
                return {"status": "RE", "output": "", "traceback": tb}
    except subprocess.TimeoutExpired:
        return {"status": "TLE", "output": "", "traceback": "Execution timed out"}
    except MemoryError:
        return {"status": "MLE", "output": "", "traceback": "Memory limit exceeded"}
    except Exception as e:
        return {"status": "RE", "output": "", "traceback": str(e)}


@dataclass
class ExecutionResult:
    status: str  # AC, WA, TLE, MLE, CE, RE, PE
    passed_tests: int = 0
    total_tests: int = 0
    stdout: str = ""
    stderr: str = ""
    traceback: str = ""
    execution_time: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 1.0 if self.status == ExecutionStatus.AC else 0.0
        return self.passed_tests / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "pass_rate": self.pass_rate,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "traceback": self.traceback,
            "execution_time": self.execution_time,
            "details": self.details,
        }


class PythonSandbox:
    """Safe execution sandbox for Python code snippets against test suites."""

    def __init__(self, default_timeout: float = 5.0, max_memory_mb: float = 1024.0):
        self.default_timeout = default_timeout
        self.max_memory_mb = max_memory_mb

    def check_syntax(self, code: str) -> Optional[ExecutionResult]:
        """Validates python code syntax before execution."""
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            tb_str = f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}\n{e.text or ''}"
            return ExecutionResult(
                status=ExecutionStatus.CE,
                passed_tests=0,
                total_tests=0,
                stderr=str(e),
                traceback=tb_str,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.CE,
                passed_tests=0,
                total_tests=0,
                stderr=str(e),
                traceback=traceback.format_exc(),
            )

    def run_single(
        self,
        code: str,
        test_case: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Runs python code in an isolated subprocess with timeout."""
        timeout = timeout or self.default_timeout

        syntax_err = self.check_syntax(code)
        if syntax_err:
            return syntax_err

        # Prepare harness runner script
        harness_code = self._build_runner_script(code, test_case)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(harness_code)
            tmp_path = tmp_file.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = proc.stdout
            stderr = proc.stderr

            if proc.returncode == 0:
                if "PASSED_TEST_MARKER" in stdout:
                    return ExecutionResult(
                        status=ExecutionStatus.AC,
                        passed_tests=1,
                        total_tests=1,
                        stdout=stdout.replace("PASSED_TEST_MARKER", "").strip(),
                        stderr=stderr,
                    )
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.AC,
                        passed_tests=1,
                        total_tests=1,
                        stdout=stdout.strip(),
                        stderr=stderr,
                    )
            elif proc.returncode == 2:  # Assertion / Wrong Answer error code
                return ExecutionResult(
                    status=ExecutionStatus.WA,
                    passed_tests=0,
                    total_tests=1,
                    stdout=stdout,
                    stderr=stderr,
                    traceback=stderr.strip(),
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.RE,
                    passed_tests=0,
                    total_tests=1,
                    stdout=stdout,
                    stderr=stderr,
                    traceback=stderr.strip(),
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TLE,
                passed_tests=0,
                total_tests=1,
                stderr=f"Execution timed out after {timeout} seconds.",
                traceback=f"TimeoutError: Code execution exceeded time limit of {timeout}s.",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RE,
                passed_tests=0,
                total_tests=1,
                stderr=str(e),
                traceback=traceback.format_exc(),
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def run_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        timeout_per_test: Optional[float] = None,
    ) -> ExecutionResult:
        """Runs python code against a suite of test cases."""
        syntax_err = self.check_syntax(code)
        if syntax_err:
            syntax_err.total_tests = len(test_cases)
            return syntax_err

        if not test_cases:
            res = self.run_single(code, timeout=timeout_per_test)
            res.total_tests = 1
            return res

        passed = 0
        total = len(test_cases)
        details = []
        overall_status = ExecutionStatus.AC
        first_traceback = ""
        combined_stdout = []
        combined_stderr = []

        for i, test in enumerate(test_cases):
            res = self.run_single(code, test_case=test, timeout=timeout_per_test)
            test_detail = {
                "test_index": i,
                "status": res.status,
                "passed": res.status == ExecutionStatus.AC,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "traceback": res.traceback,
            }
            details.append(test_detail)

            if res.stdout:
                combined_stdout.append(f"[Test {i+1}] {res.stdout}")
            if res.stderr:
                combined_stderr.append(f"[Test {i+1}] {res.stderr}")

            if res.status == ExecutionStatus.AC:
                passed += 1
            else:
                if not first_traceback:
                    first_traceback = res.traceback
                # Pick highest severity error for overall status
                if res.status in (ExecutionStatus.TLE, ExecutionStatus.MLE):
                    overall_status = ExecutionStatus.TLE
                elif res.status == ExecutionStatus.RE and overall_status != ExecutionStatus.TLE:
                    overall_status = ExecutionStatus.RE
                elif res.status == ExecutionStatus.WA and overall_status not in (ExecutionStatus.TLE, ExecutionStatus.RE):
                    overall_status = ExecutionStatus.WA

        return ExecutionResult(
            status=ExecutionStatus.AC if passed == total else overall_status,
            passed_tests=passed,
            total_tests=total,
            stdout="\n".join(combined_stdout),
            stderr="\n".join(combined_stderr),
            traceback=first_traceback,
            details=details,
        )

    def _build_runner_script(self, code: str, test_case: Optional[Dict[str, Any]]) -> str:
        """Constructs executable python script containing solution and test runner logic."""
        script = f"{code}\n\n"

        if not test_case:
            script += "print('PASSED_TEST_MARKER')\n"
            return script

        # Assertion case
        if "assertion" in test_case:
            script += "try:\n"
            script += f"    {test_case['assertion']}\n"
            script += "    print('PASSED_TEST_MARKER')\n"
            script += "except AssertionError as e:\n"
            script += "    import sys, traceback\n"
            script += "    traceback.print_exc()\n"
            script += "    sys.exit(2)\n"

        # Input/Output standard I/O test case
        elif "input" in test_case and "output" in test_case and isinstance(test_case["input"], str):
            inp = repr(test_case["input"])
            expected = repr(test_case["output"].strip())
            script += "import io, sys, traceback\n"
            script += f"sys.stdin = io.StringIO({inp})\n"
            script += "out_buf = io.StringIO()\n"
            script += "sys.stdout = out_buf\n"
            script += "try:\n"
            script += f"    exec({repr(code)})\n"
            script += "    actual = out_buf.getvalue().strip()\n"
            script += f"    if actual == {expected}:\n"
            script += "        sys.stderr.write('PASSED_TEST_MARKER\\n')\n"
            script += "    else:\n"
            script += "        sys.stderr.write(f'AssertionError: Expected {expected}, got {actual}\\n')\n"
            script += "        sys.exit(2)\n"
            script += "except SystemExit as e:\n"
            script += "    sys.exit(e.code)\n"
            script += "except Exception as e:\n"
            script += "    traceback.print_exc()\n"
            script += "    sys.exit(1)\n"


        # Function call case: fn_name, input, expected
        elif "fn_name" in test_case and "input" in test_case and "expected" in test_case:
            fn_name = test_case["fn_name"]
            inputs = test_case["input"]
            expected = repr(test_case["expected"])

            if isinstance(inputs, list):
                args_str = ", ".join(repr(arg) for arg in inputs)
            elif isinstance(inputs, dict):
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in inputs.items())
            else:
                args_str = repr(inputs)

            script += "import sys, traceback\n"
            script += "try:\n"
            script += f"    actual = {fn_name}({args_str})\n"
            script += f"    if actual == {expected}:\n"
            script += "        print('PASSED_TEST_MARKER')\n"
            script += "    else:\n"
            script += f"        sys.stderr.write('AssertionError: Expected ' + repr({expected}) + ', got ' + repr(actual) + '\\n')\n"
            script += "        sys.exit(2)\n"
            script += "except SystemExit as e:\n"
            script += "    sys.exit(e.code)\n"
            script += "except Exception as e:\n"
            script += "    traceback.print_exc()\n"
            script += "    sys.exit(1)\n"
        else:
            script += "print('PASSED_TEST_MARKER')\n"

        return script


