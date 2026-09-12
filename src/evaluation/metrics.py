"""Evaluation metrics for Pass@1, Fix@K, error distribution, and recovery rates.
Implementation owned by Junior B.
"""

from typing import Any, Dict, List, Tuple
import numpy as np


def calculate_pass_at_1(results: List[Dict[str, Any]]) -> float:
    """Calculates Pass@1 accuracy across single-attempt evaluation results."""
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("status") == "AC" or r.get("solved", False))
    return float(passed / len(results))


def calculate_fix_at_k(sessions: List[Dict[str, Any]], k: int) -> float:
    """Calculates Fix@K: cumulative success rate within K debugging turns."""
    if not sessions:
        return 0.0
    success_count = 0
    for s in sessions:
        turns = s.get("history", [])
        # Check if solved at or before turn K
        if s.get("solved") and s.get("solved_turn") is not None and s["solved_turn"] <= k:
            success_count += 1
        else:
            # Fallback check history directly
            for t in turns[:k]:
                if t.get("status") == "AC":
                    success_count += 1
                    break
    return float(success_count / len(sessions))


def calculate_error_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates status percentage distribution (AC, WA, TLE, MLE, CE, RE, PE)."""
    if not results:
        return {}
    counts: Dict[str, int] = {}
    for r in results:
        st = r.get("status") or r.get("final_status", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    total = len(results)
    return {st: float(cnt / total) for st, cnt in counts.items()}


def calculate_recovery_rate(sessions: List[Dict[str, Any]]) -> float:
    """Calculates percentage of initially failing (buggy) solutions successfully fixed."""
    buggy_sessions = [
        s for s in sessions
        if s.get("history") and s["history"][0].get("status") != "AC"
    ]
    if not buggy_sessions:
        return 1.0  # 100% if no initial bugs
    recovered = sum(1 for s in buggy_sessions if s.get("solved", False))
    return float(recovered / len(buggy_sessions))


def summarize_eval_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates complete summary dictionary for paper tables."""
    return {
        "pass_at_1": calculate_pass_at_1(sessions),
        "fix_at_1": calculate_fix_at_k(sessions, k=1),
        "fix_at_3": calculate_fix_at_k(sessions, k=3),
        "fix_at_5": calculate_fix_at_k(sessions, k=5),
        "recovery_rate": calculate_recovery_rate(sessions),
        "error_distribution": calculate_error_distribution(sessions),
        "total_eval_samples": len(sessions),
    }
