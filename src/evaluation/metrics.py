"""Evaluation metrics and benchmark runner."""

from typing import Any, Dict, List
from src.debugging.debug_loop import agentic_debug_loop


def extract_eval_test_cases(example: dict) -> list:
    """Extract executable test case list from evaluation dataset item (HumanEval, MBPP, APPS)."""
    if "test" in example and isinstance(example["test"], str):
        return [example["test"]]
    
    test_cases = example.get("test_cases", example.get("input_output", []))
    if isinstance(test_cases, str):
        try:
            parsed = eval(test_cases)
            if isinstance(parsed, dict):
                return parsed.get("inputs", [])
            elif isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    elif isinstance(test_cases, list):
        return test_cases
    return []


def evaluate(model, tokenizer, dataset, K: int = 3, label: str = "", debug_loop_fn=None) -> dict:
    """Compute Pass@1 and Fix@K metrics on an evaluation dataset."""
    if debug_loop_fn is None:
        debug_loop_fn = agentic_debug_loop

    pass1, fixk, total = 0, 0, len(dataset)
    for i, example in enumerate(dataset):
        problem = example.get("prompt", example.get("question", ""))
        test_cases = extract_eval_test_cases(example)

        history = debug_loop_fn(model, tokenizer, problem, test_cases, K=K)
        statuses = [h["result"]["status"] for h in history]

        if statuses and statuses[0] == "AC":
            pass1 += 1
        if "AC" in statuses:
            fixk += 1

        if (i + 1) % 5 == 0 or (i + 1) == total:
            current_pass1 = pass1 / (i + 1)
            current_fixk = fixk / (i + 1)
            print(f"{label} [{i+1}/{total}] Pass@1={current_pass1:.2%} Fix@{K}={current_fixk:.2%}")

    results = {
        "label": label,
        "pass_at_1": pass1 / total if total > 0 else 0.0,
        f"fix_at_{K}": fixk / total if total > 0 else 0.0,
        "total": total,
    }
    return results


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
        if s.get("solved") and s.get("solved_turn") is not None and s["solved_turn"] <= k:
            success_count += 1
        else:
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
        return 1.0
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
