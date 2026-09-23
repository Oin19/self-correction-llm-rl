"""Benchmark test-case normalization (no heavy ML deps)."""
import json


def normalize_tests(ex: dict) -> list:
    """Normalize APPS/HumanEval/MBPP test fields into executor-ready cases.

    Returns distinct test cases only — never duplicates cases to pad the count,
    so partial rewards (passed/total) reflect real unique tests.
    """
    if isinstance(ex.get("test"), str) and ex["test"].strip():
        return [ex["test"]]
    if isinstance(ex.get("test_list"), list) and ex["test_list"]:
        return [{"assertion": x} for x in ex["test_list"] if isinstance(x, str) and x.strip()]
    raw = ex.get("input_output", ex.get("test_cases", []))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []
    ins, outs, fn = raw.get("inputs", []), raw.get("outputs", []), raw.get("fn_name")
    cases = []
    for inp, out in zip(ins, outs):
        if fn:
            if isinstance(inp, list):
                args = ", ".join(repr(x) for x in inp)
            elif isinstance(inp, dict):
                args = ", ".join(f"{k}={v!r}" for k, v in inp.items())
            else:
                args = repr(inp)
            cases.append({"assertion": f"assert {fn}({args}) == {out!r}"})
        else:
            cases.append({
                "input": "\n".join(inp) if isinstance(inp, list) else str(inp),
                "output": "\n".join(out) if isinstance(out, list) else str(out),
            })
    return cases
