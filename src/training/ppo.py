"""PPO training entry points with execution-guided rewards."""

import gc
import os
import sys
import torch

try:
    from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
except Exception:
    try:
        from trl import AutoModelForCausalLMWithValueHead
    except Exception:
        try:
            from trl.models import AutoModelForCausalLMWithValueHead
        except Exception:
            AutoModelForCausalLMWithValueHead = None

try:
    from trl.trainer.ppo_trainer import PPOTrainer
except Exception:
    try:
        from trl import PPOTrainer
    except Exception:
        PPOTrainer = None

try:
    from trl.trainer.ppo_config import PPOConfig
except Exception:
    try:
        from trl import PPOConfig
    except Exception:
        PPOConfig = None

from src.execution.executor import run_code
from src.rewards.execution_reward import compute_reward


def run_ppo_training(
    sft_model_path: str,
    tokenizer,
    dataset,
    output_dir: str = "./checkpoints/ppo",
    num_epochs: int = 1,
    learning_rate: float = 1e-6,
    batch_size: int = 2,
    mini_batch_size: int = 1,
    gradient_accumulation_steps: int = 2,
    init_kl_coef: float = 0.02,
    target_kl: float = 6.0,
    max_steps: int = 10,
):
    print("-> [1/4] Preparing PPO dataset and tokenizer...", flush=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_ppo_prompt(example):
        problem = example.get("question", example.get("prompt", ""))
        prompt_text = f"### Problem:\n{problem}\n\n### Solution:\n```python\n"
        tokens = tokenizer(prompt_text, truncation=True, max_length=256)
        return {"input_ids": tokens["input_ids"]}

    if "input_ids" not in dataset.column_names:
        dataset = dataset.map(tokenize_ppo_prompt, remove_columns=dataset.column_names)

    # Clear CUDA memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"-> [2/4] Available GPUs: {num_gpus}. Loading PPO model with Value Head...", flush=True)

    # Multi-GPU (T4 x2) distribution strategy
    if num_gpus >= 2:
        print("   Multi-GPU detected! Distributing Policy Model across GPU 0 & GPU 1 using device_map='auto'", flush=True)
        device_map = "auto"
    elif num_gpus == 1:
        print("   Single GPU detected! Using cuda:0 with memory-efficient precision", flush=True)
        device_map = {"": 0}
    else:
        device_map = None

    try:
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except Exception:
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )

    if hasattr(ppo_model, "config"):
        ppo_model.config.pad_token_id = tokenizer.pad_token_id
        ppo_model.config.use_cache = True
    if hasattr(ppo_model, "generation_config") and ppo_model.generation_config is not None:
        ppo_model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Enable Gradient Checkpointing to save VRAM
    if hasattr(ppo_model, "pretrained_model") and hasattr(ppo_model.pretrained_model, "gradient_checkpointing_enable"):
        try:
            ppo_model.pretrained_model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Freeze base model parameters so only LoRA + Value Head are optimized
    if hasattr(ppo_model, "pretrained_model"):
        for param in ppo_model.pretrained_model.parameters():
            param.requires_grad = False

    for name, param in ppo_model.named_parameters():
        if "lora_" in name or "v_head" in name or "summary" in name or "score" in name:
            param.requires_grad = True

    trainable_params = [p for p in ppo_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    print("-> [3/4] Initializing TRL PPOTrainer...", flush=True)
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

    def ppo_collate_fn(data):
        return {key: [d[key] for d in data] for key in data[0]}

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        ref_model=None,  # TRL manages ref_model using PEFT shared weights
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        data_collator=ppo_collate_fn,
    )

    generation_kwargs = {
        "max_new_tokens": 64,
        "do_sample": True,
        "top_p": 0.95,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    step_count = 0
    total_batches = min(len(ppo_trainer.dataloader), max_steps) if max_steps else len(ppo_trainer.dataloader)

    print(f"-> [4/4] Starting PPO Rollout Optimization ({total_batches} steps max)...", flush=True)
    for epoch in range(num_epochs):
        for batch in ppo_trainer.dataloader:
            step_count += 1
            print(f"   [Step {step_count}/{total_batches}] Generating code & executing in sandbox...", flush=True)

            query_tensors = [
                q.squeeze() if isinstance(q, torch.Tensor) and q.dim() > 1 else (torch.tensor(q, dtype=torch.long) if not isinstance(q, torch.Tensor) else q)
                for q in batch["input_ids"]
            ]
            with torch.no_grad():
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    **generation_kwargs,
                )

            rewards = []
            for q, r in zip(query_tensors, response_tensors):
                code = tokenizer.decode(r, skip_special_tokens=True)
                result = run_code(code, timeout=3)
                reward_val = compute_reward(result["status"], 0, 1)
                rewards.append(torch.tensor(reward_val, dtype=torch.float32))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            mean_score = stats.get("ppo/mean_scores", 0.0)
            kl_val = stats.get("objective/kl", 0.0)
            print(f"   ✓ Completed Step {step_count}/{total_batches} | mean_reward={mean_score:.3f} | kl={kl_val:.3f}", flush=True)

            if max_steps and step_count >= max_steps:
                break
        if max_steps and step_count >= max_steps:
            break

    print(f"-> Saving final PPO adapter checkpoint to {output_dir}/final...", flush=True)
    ppo_model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    return ppo_trainer
