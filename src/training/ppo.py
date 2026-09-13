"""PPO training entry points with execution-guided rewards."""

import torch
try:
    from trl import AutoModelForCausalLMWithValueHead
except ImportError:
    try:
        from trl.models import AutoModelForCausalLMWithValueHead
    except ImportError:
        from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead

try:
    from trl import PPOConfig, PPOTrainer
except ImportError:
    from trl.trainer import PPOConfig, PPOTrainer

from src.execution.executor import run_code
from src.rewards.execution_reward import compute_reward


def run_ppo_training(
    sft_model_path: str,
    tokenizer,
    dataset,
    output_dir: str = "./checkpoints/ppo",
    num_epochs: int = 10,
    learning_rate: float = 1e-6,
    batch_size: int = 16,
    mini_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    init_kl_coef: float = 0.02,
    target_kl: float = 6.0,
):
    def tokenize_ppo_prompt(example):
        problem = example.get("question", example.get("prompt", ""))
        prompt_text = f"### Problem:\n{problem}\n\n### Solution:\n```python\n"
        tokens = tokenizer(prompt_text, truncation=True, max_length=512, padding="max_length")
        return {"input_ids": tokens["input_ids"]}

    if "input_ids" not in dataset.column_names:
        dataset = dataset.map(tokenize_ppo_prompt, remove_columns=dataset.column_names)

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    ppo_config = PPOConfig(
        model_name=sft_model_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kl_penalty="kl",
        init_kl_coef=init_kl_coef,
        target_kl=target_kl,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=None,  # TRL handles reference policy for PEFT
        tokenizer=tokenizer,
        dataset=dataset,
    )

    for epoch in range(num_epochs):
        for batch in ppo_trainer.dataloader:
            queries = batch["input_ids"]
            responses = ppo_trainer.generate(
                queries,
                max_new_tokens=512,
                temperature=1.0,
                top_p=0.95,
            )

            rewards = []
            for q, r in zip(queries, responses):
                code = tokenizer.decode(r, skip_special_tokens=True)
                result = run_code(code)
                reward_val = compute_reward(result["status"], 0, 1)
                rewards.append(torch.tensor(reward_val, dtype=torch.float32))

            stats = ppo_trainer.step(queries, responses, rewards)
            mean_score = stats.get("ppo/mean_scores", 0.0)
            kl_val = stats.get("objective/kl", 0.0)
            print(f"PPO Epoch {epoch} | mean_reward={mean_score:.3f} | kl={kl_val:.3f}")

    ppo_model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    return ppo_trainer
