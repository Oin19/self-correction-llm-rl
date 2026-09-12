"""Iterative execution-feedback debugging loop."""

import random
import torch

from src.execution.executor import run_code
from src.models.generation import extract_code_block


def build_prompt(problem: str, prev_code: str = None, traceback: str = None) -> str:
    """Build the model prompt — adds error context on retry turns."""
    if prev_code is None:
        # Turn 1: just problem
        return f"### Problem:\n{problem}\n\n### Write a Python solution:\n```python"
    else:
        # Turn 2+: problem + previous attempt + error
        tb_trimmed = (traceback or "")[:512]  # cap traceback length
        return (
            f"### Problem:\n{problem}\n\n"
            f"### Your previous code:\n```python\n{prev_code}\n```\n\n"
            f"### Error you received:\n{tb_trimmed}\n\n"
            f"### Fixed version:\n```python"
        )


def agentic_debug_loop(model, tokenizer, problem: str, test_cases: list = None, K: int = 3) -> list:
    """Run up to K debugging turns with execution feedback.

    Returns a list of dicts: [{'turn': t, 'code': c, 'result': r}, ...]
    """
    history = []
    code, traceback = None, ""
    device = next(model.parameters()).device
    test_cases = test_cases or []

    for turn in range(K):
        prompt = build_prompt(problem, code, traceback)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2 if turn == 0 else 1.0,
                do_sample=(turn > 0),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        full = tokenizer.decode(output[0], skip_special_tokens=True)
        code = extract_code_block(full, prompt)

        full_code = code + "\n" + "\n".join(test_cases) if test_cases else code
        result = run_code(full_code)

        history.append({"turn": turn + 1, "code": code, "result": result})

        if result["status"] == "AC":
            break
        traceback = result.get("traceback", "")

    return history


def agentic_loop_no_feedback(model, tokenizer, problem: str, test_cases: list = None, K: int = 3) -> list:
    """RQ2 ablation study: same debugging loop but injects RANDOM fake traceback instead of real execution feedback."""
    history = []
    code = None
    fake_errors = [
        'NameError: name "x" is not defined',
        "IndexError: list index out of range",
        "TypeError: unsupported operand type(s)",
    ]
    device = next(model.parameters()).device
    test_cases = test_cases or []

    for turn in range(K):
        fake_tb = random.choice(fake_errors)
        prompt = build_prompt(problem, code, fake_tb)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2 if turn == 0 else 1.0,
                do_sample=(turn > 0),
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        full = tokenizer.decode(output[0], skip_special_tokens=True)
        code = extract_code_block(full, prompt)

        full_code = code + "\n" + "\n".join(test_cases) if test_cases else code
        result = run_code(full_code)

        history.append({"turn": turn + 1, "code": code, "result": result})

        if result["status"] == "AC":
            break

    return history
