"""DPO training entry points."""

import json
from typing import TYPE_CHECKING, List

from src.debugging.debug_loop import agentic_debug_loop
from src.utils.repro import git_commit_sha, set_global_seed
from src.utils.test_cases import normalize_tests

if TYPE_CHECKING:
    from datasets import Dataset


def parse_apps_test_cases(prob: dict) -> list:
    """Extract executor-ready tests via the same normalizer PPO/eval use."""
    return normalize_tests(prob)


def make_preference_pairs(problems, model, tokenizer, K: int = 3, seed: int = 42):
    """Collect (chosen, rejected) preference pairs from execution debug loop rollouts."""
    from datasets import Dataset

    set_global_seed(seed)
    pairs = []
    total = len(problems)
    print(f"Starting DPO preference pair generation across {total} APPS problems (seed={seed})...", flush=True)

    for idx, prob in enumerate(problems):
        question = prob.get("question", prob.get("prompt", ""))
        test_cases = parse_apps_test_cases(prob)
        if not test_cases:
            continue
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
    preference_data,
    output_dir: str = "./checkpoints/dpo",
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    seed: int = 42,
    commit_sha: str = "",
):
    """Run Direct Preference Optimization (DPO) training."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise ImportError(f"TRL library is required for DPO training. Install with `pip install trl`: {e}")

    set_global_seed(seed)
    print(f"DPO seed={seed} commit={commit_sha or git_commit_sha() or 'unknown'}", flush=True)

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
    final_dir = f"{output_dir}/final"
    dpo_trainer.save_model(final_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(final_dir)
    with open(f"{final_dir}/dpo_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint_type": "dpo_execution_grounded",
            "preference_source": "model_generated_execution_rollouts",
            "reference_solution_fallback": False,
            "test_harness": "normalize_tests",
            "seed": seed,
            "commit_sha": commit_sha or git_commit_sha(),
            "beta": beta,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "num_preference_pairs": len(preference_data),
        }, f, indent=2)
    return dpo_trainer
