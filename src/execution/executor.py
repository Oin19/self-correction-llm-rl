"""Sandboxed execution interface for running Python code safely."""

import os
import sys
import subprocess

try:
    import resource
except ImportError:
    resource = None


def run_code(code: str, timeout: int = 8) -> dict:
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
