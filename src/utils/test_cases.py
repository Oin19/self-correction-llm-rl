"""Benchmark test-case normalization (no heavy ML deps)."""
import json


def _cases_from_input_output(raw) -> list:
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


def normalize_tests(ex: dict) -> list:
    """Normalize APPS/HumanEval/MBPP test fields into executor-ready cases.

    Prefers multi-case sources (input_output / test_list) over a monolithic
    ``test`` script so passed/total can be > 1 and partial reward works.
    Never duplicates cases to pad the count.
    """
    sources = []

    raw = ex.get("input_output", ex.get("test_cases", []))
    io_cases = _cases_from_input_output(raw)
    if io_cases:
        sources.append(io_cases)

    if isinstance(ex.get("test_list"), list) and ex["test_list"]:
        tl = [{"assertion": x} for x in ex["test_list"] if isinstance(x, str) and x.strip()]
        if tl:
            sources.append(tl)

    if isinstance(ex.get("test"), str) and ex["test"].strip():
        sources.append([ex["test"]])

    if not sources:
        return []
    # Most cases first so partial credit has room; ties keep preference order above.
    return max(sources, key=len)
