"""Evaluation metrics and benchmark runner."""

import json
from typing import Any, Dict, List
from src.debugging.debug_loop import agentic_debug_loop


def extract_eval_test_cases(example: dict) -> list:
    """Normalize HumanEval, MBPP and APPS test fields."""
    if isinstance(example.get("test"), str) and example["test"].strip():
        return [example["test"]]
    if isinstance(example.get("test_list"), list) and example["test_list"]:
        return [{"assertion": x} for x in example["test_list"] if isinstance(x, str) and x.strip()]
    raw = example.get("input_output", example.get("test_cases", []))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []
    inputs, outputs, fn = raw.get("inputs", []), raw.get("outputs", []), raw.get("fn_name")
    cases = []
    for inp, out in zip(inputs, outputs):
        if fn:
            if isinstance(inp, list):
                args = ", ".join(repr(x) for x in inp)
            elif isinstance(inp, dict):
                args = ", ".join(f"{k}={v!r}" for k, v in inp.items())
            else:
                args = repr(inp)
            cases.append({"assertion": f"assert {fn}({args}) == {out!r}"})
        else:
            cases.append({"input": "\n".join(inp) if isinstance(inp, list) else str(inp), "output": "\n".join(out) if isinstance(out, list) else str(out)})
    return cases


def evaluate(model, tokenizer, dataset, K: int = 3, label: str = "", debug_loop_fn=None) -> dict:
    """Compute Pass@1 and Fix@K using explicit executable tests."""
    debug_loop_fn = debug_loop_fn or agentic_debug_loop
    total = len(dataset)
    pass1 = fixk = 0
    for i, example in enumerate(dataset):
        tests = extract_eval_test_cases(example)
        if not tests:
            raise ValueError(f"No executable tests for evaluation example {i}; refusing to score it.")
        problem = example.get("prompt", example.get("question", example.get("text", "")))
        history = debug_loop_fn(model, tokenizer, problem, tests, K=K)
        statuses = [h["result"]["status"] for h in history]
        pass1 += int(bool(statuses) and statuses[0] == "AC")
        fixk += int("AC" in statuses)
        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"{label} [{i+1}/{total}] Pass@1={pass1/(i+1):.2%} Fix@{K}={fixk/(i+1):.2%}")
    return {"label": label, "pass_at_1": pass1/total if total else 0.0, f"fix_at_{K}": fixk/total if total else 0.0, "total": total}


def calculate_pass_at_1(results: List[Dict[str, Any]]) -> float:
    if not results: return 0.0
    return sum(1 for r in results if r.get("status") == "AC" or r.get("solved", False)) / len(results)


def calculate_fix_at_k(sessions: List[Dict[str, Any]], k: int) -> float:
    if not sessions: return 0.0
    return sum(1 for s in sessions if (s.get("solved") and s.get("solved_turn", 10**9) <= k) or any(t.get("status") == "AC" for t in s.get("history", [])[:k])) / len(sessions)


def calculate_error_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results: return {}
    counts: Dict[str, int] = {}
    for r in results:
        st = r.get("status") or r.get("final_status", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return {st: n / len(results) for st, n in counts.items()}


def calculate_recovery_rate(sessions: List[Dict[str, Any]]) -> float:
    buggy = [s for s in sessions if s.get("history") and s["history"][0].get("status") != "AC"]
    return sum(1 for s in buggy if s.get("solved", False)) / len(buggy) if buggy else 0.0


def summarize_eval_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"pass_at_1": calculate_pass_at_1(sessions), "fix_at_1": calculate_fix_at_k(sessions, 1), "fix_at_3": calculate_fix_at_k(sessions, 3), "fix_at_5": calculate_fix_at_k(sessions, 5), "recovery_rate": calculate_recovery_rate(sessions), "error_distribution": calculate_error_distribution(sessions), "total_eval_samples": len(sessions)}
