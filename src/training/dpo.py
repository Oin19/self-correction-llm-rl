"""DPO training entry points."""

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from src.debugging.debug_loop import agentic_debug_loop


def make_preference_pairs(problems, model, tokenizer, K: int = 3) -> Dataset:
    """Collect (chosen, rejected) preference pairs from execution debug loop rollouts.

    'chosen': code solution that passed (AC)
    'rejected': code solution that failed (CE / RE / WA / TLE / MLE)
    """
    pairs = []
    for prob in problems:
        question = prob.get("question", prob.get("prompt", ""))
        test_cases = prob.get("test_cases", [])
        history = agentic_debug_loop(model, tokenizer, question, test_cases, K=K)

        ac_turns = [h for h in history if h["result"]["status"] == "AC"]
        bad_turns = [h for h in history if h["result"]["status"] in ("CE", "RE", "WA", "TLE", "MLE")]

        if ac_turns and bad_turns:
            pairs.append({
                "prompt": question,
                "chosen": ac_turns[0]["code"],  # winning solution
                "rejected": bad_turns[0]["code"],  # losing solution
            })

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
    dpo_config = DPOConfig(
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        output_dir=output_dir,
        fp16=True,
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT reference trick
        args=dpo_config,
        train_dataset=preference_data,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    return dpo_trainer
