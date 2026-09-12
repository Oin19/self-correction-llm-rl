"""Evaluation metrics and benchmark runner."""

from src.debugging.debug_loop import agentic_debug_loop


def evaluate(model, tokenizer, dataset, K: int = 3, label: str = "", debug_loop_fn=None) -> dict:
    """Compute Pass@1 and Fix@K metrics on a evaluation dataset."""
    if debug_loop_fn is None:
        debug_loop_fn = agentic_debug_loop

    pass1, fixk, total = 0, 0, len(dataset)
    for i, example in enumerate(dataset):
        problem = example.get("prompt", example.get("question", ""))
        test_cases = example.get("test_cases", example.get("input_output", []))

        if isinstance(test_cases, str):
            try:
                parsed = eval(test_cases)
                test_cases = parsed.get("inputs", []) if isinstance(parsed, dict) else []
            except Exception:
                test_cases = []

        history = debug_loop_fn(model, tokenizer, problem, test_cases, K=K)
        statuses = [h["result"]["status"] for h in history]

        if statuses and statuses[0] == "AC":
            pass1 += 1
        if "AC" in statuses:
            fixk += 1

        if (i + 1) % 20 == 0 or (i + 1) == total:
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


# TODO: implement resource and recovery metrics.
