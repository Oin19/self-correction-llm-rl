"""Execution status definitions for code sandbox execution."""

STATUSES = ("AC", "PE", "WA", "TLE", "MLE", "CE", "RE")

STATUS_DESCRIPTIONS = {
    "AC": "All Correct (all test cases passed)",
    "PE": "Partial Execution (some test cases passed)",
    "WA": "Wrong Answer (output mismatch)",
    "TLE": "Time Limit Exceeded (execution timed out)",
    "MLE": "Memory Limit Exceeded (memory cap hit)",
    "CE": "Compilation / Syntax Error",
    "RE": "Runtime Error (exception / crash)",
}

def is_valid_status(status: str) -> bool:
    """Check if status code is valid."""
    return status in STATUSES
