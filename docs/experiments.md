# Experiment Log

Every completed experiment should record:

| Field | Required |
|---|---|
| Experiment ID | Yes |
| Research Question | Yes |
| Model + checkpoint | Yes |
| Dataset + split | Yes |
| Method | Yes |
| Debugging turns K | Yes |
| Sampling settings | Yes |
| Feedback type | Yes |
| Reward version | Yes for RL |
| Seed | Yes |
| GPU / VRAM | Yes |
| Wall-clock time | Yes |
| Metrics | Yes |
| Notes / anomalies | Yes |

Do not report a result without enough metadata to reproduce the run.

## Entries

### EXP-PPO-2026-09-24 (Verified PPO Training Rerun)

| Field | Value |
|---|---|
| Experiment ID | `EXP-PPO-2026-09-24` |
| Research Question | RQ3 / RQ4 (does RL with dense partial execution reward train stably?) |
| Model + checkpoint | `deepseek-ai/deepseek-coder-1.3b-instruct` + SFT LoRA (`./checkpoints/sft/final`) → PPO (`./checkpoints/ppo/final`) |
| Dataset + split | `codeparrot/apps` `train[:1000]`, solutions non-empty |
| Method | PPO (`trl`), `kl_penalty=abs`, frozen SFT `ref_model`, `is_peft_model=False` |
| Debugging turns K | N/A (single-turn rollout) |
| Sampling settings | `temperature=1.0`, `top_p=0.95`, `max_new_tokens=384`, prompt truncation 350 |
| Feedback type | Sandbox execution (partial packed-suite credit) |
| Reward version | Dense partial: AC→1.0; packed segment/line `passed/total`; WA/PE 0→0.0; CE/RE/TLE/MLE 0→−0.2 |
| Seed | Not fixed (sampling); dataset order fixed by `select()` |
| GPU / VRAM | Kaggle T4 |
| Wall-clock time | ~10 PPO steps (smoke-scale; not full 50-step paper run) |
| Metrics | See `outputs.md` NB05 2026-09-24: rewards include `0.25/0.33/1.0`; `kl` 1.1–2.2 all positive; `kl_coef` 0.050→0.048; `grad_norm` 0.16–0.32 all non-zero |
| Notes / anomalies | Fixes supervisor issues (no partial reward, high/negative KL, `grad_norm=0`). Empty/truncated model outputs still common on hard APPS (1.3B quality, not a bug). Git: `56ce85a` (packed partial credit), `d3cada9` (outputs). Tests: 49 pass. |

**Status**: Validation / diagnostic run — not a final paper training run. Superseded for RQ3/RQ4 claims by the paper-scale arms below.

### EXP-PPO-DENSE (Completed — Paper RQ3 Dense Arm)

| Field | Value |
|---|---|
| Experiment ID | `EXP-PPO-DENSE` |
| Research Question | RQ3 (does RL with dense partial execution reward beat SFT / zero-shot?) |
| Model + checkpoint | `deepseek-ai/deepseek-coder-1.3b-instruct` + SFT LoRA → `./checkpoints/ppo_dense/final` |
| Dataset + split | `codeparrot/apps` `train[:1000]`, tests non-empty |
| Method | PPO, `reward_mode=dense`, frozen SFT ref, `kl_penalty=abs`; LoRA trainable `3,147,777 / 1,349,619,713` (`0.2332%`) |
| Debugging turns K | N/A (single-turn rollout) |
| Sampling settings | `temperature=1.0`, `top_p=0.95`, `max_new_tokens=384`, prompt truncation 350 |
| Feedback type | Sandbox execution (partial packed-suite credit) |
| Reward version | Dense partial: AC→1.0; `passed/total`; WA/PE 0→0.0; CE/RE/TLE/MLE 0→−0.2 |
| Seed | `42` (logged in `ppo_metadata.json`) |
| Hyperparameters | `batch_size=2`, `mini_batch_size=1`, `gradient_accumulation_steps=2`, `learning_rate=2e-7`, `init_kl_coef=0.30`, `target_kl=4.0`, `max_steps=100` |
| GPU / VRAM | Kaggle T4 |
| Wall-clock time | Not captured in run log (100 steps × 2 rollouts, Kaggle session) |
| Metrics | 100/100 steps. Mean step reward `0.039` (sum `3.888`, range `−0.200…1.000`), positive on `33/100` steps; AC `21/200` rollouts (`10.5%`). KL: `0 → 4.822` max @ step 99 (`1.21×` target 4.0), mean `2.53`, final `4.201` — bounded. `kl_coef` `0.3000 → 0.2223` (min `0.2200` @ step 95, upticks when KL > target). `grad_norm` `0.191–0.692`, non-zero every step. Mean reward first-10 `0.089` → last-10 `0.085` (stable). |
| Notes / anomalies | Run 2026-09-25 on Kaggle; Cell 2 cloned `junior-A`, metadata verified: `reward_mode=dense`, `seed=42`, `commit_sha=36c8055…`, `learning_rate=2e-7`, `init_kl_coef=0.3`, `target_kl=4`, `batch_size=2`. Empty/truncated rollouts on hard APPS persist (1.3B model quality, not a bug). |

**Status**: Completed 2026-09-25 — paper-ready RQ3 arm (metadata verified).

### EXP-PPO-BINARY (Completed — Paper RQ4 Binary Arm)

| Field | Value |
|---|---|
| Experiment ID | `EXP-PPO-BINARY` |
| Research Question | RQ4 (does binary AC-only reward match dense partial credit?) |
| Model + checkpoint | `deepseek-ai/deepseek-coder-1.3b-instruct` + SFT LoRA → `./checkpoints/ppo_binary/final` |
| Dataset + split | Same as `EXP-PPO-DENSE` (`train[:1000]`, same order/seed) |
| Method | PPO, `reward_mode=binary`, frozen SFT ref, **identical HPs/seed to dense arm** (only `reward_mode` differs) |
| Debugging turns K | N/A |
| Sampling settings | Same as dense arm |
| Feedback type | Sandbox execution |
| Reward version | Binary: AC→1.0; all non-AC→0.0 (sample rewards strictly ∈ {0.0, 1.0}) |
| Seed | `42` |
| Hyperparameters | Same as dense: `batch_size=2`, `mini_batch_size=1`, `gradient_accumulation_steps=2`, `learning_rate=2e-7`, `init_kl_coef=0.30`, `target_kl=4.0`, `max_steps=100` |
| GPU / VRAM | Kaggle T4 |
| Wall-clock time | 41.0 min total (log timestamps); training loop 149.7s → 2449.0s = 38.3 min |
| Metrics | 100/100 steps. Mean step reward `0.115` (sum `11.500`, range `0.000…1.000`), positive on `22/100` steps; **AC `23/200` rollouts (`11.5%`)**. Sample rewards strictly `0.0`/`1.0` — partial passes (e.g. `passed=2/8`, `1/7`) correctly score `0.000`. Rollout status: WA 82, RE 71, AC 23, CE 21, TLE 2, MLE 1. KL: max `4.787` @ step 97 (`1.20×` target 4.0), mean `2.534`, final `4.181` — bounded, no divergence. `kl_coef` `0.3000 → 0.2247` (min `0.2216` @ step 93; dense arm `→0.2223`). `grad_norm` `0.194–0.588`, non-zero every step. |
| Notes / anomalies | Run 2026-09-25 on Kaggle (4th attempt; earlier attempts had `target_kl=6` or old HPs). Metadata verified: `reward_mode=binary`, `reward_type=execution_binary_ac_only`, `seed=42`, `commit_sha=36c8055…`, `target_kl=4` — only `reward_mode`/`reward_type` differ from dense metadata. |

**Status**: Completed 2026-09-25 — paper-ready RQ4 arm (metadata verified).

### EXP-DPO-RETRAIN (Planned — Paper RQ5)

| Field | Value |
|---|---|
| Experiment ID | `EXP-DPO-RETRAIN` |
| Research Question | RQ5 (is DPO competitive with PPO?) |
| Model + checkpoint | SFT → `./checkpoints/dpo/final` |
| Dataset + split | `codeparrot/apps` preference rollouts on `train[:500]` (K=3) |
| Method | DPO β=0.1, LR 5e-5, 3 epochs; tests via `normalize_tests` (aligned with PPO) |
| Seed | `42` |
| Notes / anomalies | Supersedes NB06 2026-09-15 (100 pairs, pre-fix harness). |

**Status**: Planned.

### EXP-EVAL-RQ3RQ4 (Planned — Full 5-Arm Evaluation)

| Field | Value |
|---|---|
| Experiment ID | `EXP-EVAL-RQ3RQ4` |
| Research Question | RQ3 / RQ4 / RQ5 / RQ7 |
| Arms | Zero-Shot · SFT · PPO-dense · PPO-binary · DPO |
| Dataset + split | HumanEval 164 + MBPP 500 (`EVAL_N=None`) |
| Method | NB07 corrected; fixed seed 42; same decoding/tests for all arms |
| Metrics | Pass@1, Fix@3, Fix@5 per arm/benchmark |
| Notes / anomalies | Do not cite historical NB07 smoke tables. Requires all trained checkpoints present. |

**Status**: Planned (after paper training runs).
