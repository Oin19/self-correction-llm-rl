"""DPO training entry points."""

import json
from datasets import Dataset

from src.debugging.debug_loop import agentic_debug_loop


def parse_apps_test_cases(prob: dict) -> list:
    """Extract test case verification code from APPS dataset item (input_output field)."""
    test_cases = prob.get("test_cases", [])
    if test_cases:
        return test_cases if isinstance(test_cases, list) else [str(test_cases)]

    io_data = prob.get("input_output", None)
    if not io_data:
        return []

    if isinstance(io_data, str):
        try:
            io_data = json.loads(io_data)
        except Exception:
            return []

    if not isinstance(io_data, dict):
        return []

    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    fn_name = io_data.get("fn_name", None)

    if not inputs or not outputs:
        return []

    test_code_snippets = []
    if fn_name:
        for inp, outp in zip(inputs, outputs):
            test_code_snippets.append(f"assert {fn_name}(*{repr(inp)}) == {repr(outp)}")
    else:
        for inp, outp in zip(inputs, outputs):
            inp_str = "\n".join(inp) if isinstance(inp, list) else str(inp)
            outp_str = "\n".join(outp) if isinstance(outp, list) else str(outp)
            test_code_snippets.append({"input": inp_str, "output": outp_str})


    return test_code_snippets


def make_preference_pairs(problems, model, tokenizer, K: int = 3) -> Dataset:
    """Collect (chosen, rejected) preference pairs from execution debug loop rollouts."""
    pairs = []
    total = len(problems)
    print(f"Starting DPO preference pair generation across {total} APPS problems...", flush=True)

    for idx, prob in enumerate(problems):
        question = prob.get("question", prob.get("prompt", ""))
        test_cases = parse_apps_test_cases(prob)
        history = agentic_debug_loop(model, tokenizer, question, test_cases, K=K)

        ac_turns = [h for h in history if h["result"]["status"] == "AC"]
        bad_turns = [h for h in history if h["result"]["status"] in ("CE", "RE", "WA", "TLE", "MLE")]

        if ac_turns and bad_turns:
            pairs.append({
                "prompt": question,
                "chosen": ac_turns[0]["code"],
                "rejected": bad_turns[0]["code"],
            })
        # No reference-solution fallback: DPO pairs must come from the model's
        # own execution-grounded rollouts so the comparison remains methodologically clean.

        if (idx + 1) % 5 == 0 or (idx + 1) == total:
            print(f"   [Progress: {idx + 1}/{total}] Generated {len(pairs)} preference pairs...", flush=True)

    return Dataset.from_list(pairs)


def run_dpo_training(
    model,
    tokenizer,
    preference_data: Dataset,
    output_dir: str = "./checkpoints/dpo",
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
):
    """Run Direct Preference Optimization (DPO) training."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise ImportError(f"TRL library is required for DPO training. Install with `pip install trl`: {e}")

    try:
        dpo_config = DPOConfig(
            beta=beta,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            fp16=True,
            report_to="none",
        )
    except Exception:
        dpo_config = DPOConfig(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            fp16=True,
            report_to="none",
        )

    # Compatible with TRL versions using processing_class or tokenizer
    try:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=preference_data,
            processing_class=tokenizer,
        )
    except (TypeError, Exception):
        try:
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_config,
                train_dataset=preference_data,
                tokenizer=tokenizer,
            )
        except Exception:
            dpo_trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=preference_data,
                processing_class=tokenizer,
            )

    dpo_trainer.train()
    dpo_trainer.save_model(f"{output_dir}/final")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(f"{output_dir}/final")
    return dpo_trainer
