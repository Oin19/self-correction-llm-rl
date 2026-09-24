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

**Status**: Validation / diagnostic run — not a final paper training run. Next: NB07 4-variant evaluation with this checkpoint.
