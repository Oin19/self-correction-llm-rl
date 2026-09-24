# Experiment Validation Status

## Current status

The September 2026 notebook runs are preserved as execution evidence, but the numerical results from the original Notebook 7 must not be treated as final paper results until they are reproduced with the corrected execution pipeline.

### Corrected requirements

1. Every scored example must have explicit executable benchmark tests.
2. Empty test suites must never receive an AC score.
3. MBPP `test_list`, HumanEval `test`, and APPS `input_output` must be recognized.
4. PPO must load the trained SFT adapter; it must not silently fall back to the base model.
5. PPO reward must be computed from benchmark execution, not merely process exit status.
6. The evaluation must not silently replace a missing SFT/PPO/DPO checkpoint with the base model.
7. Smoke-test results (N<=20 per benchmark) are validation evidence only, not final paper results.
8. Final comparisons must use the same fixed evaluation set, decoding settings, and evaluation code for all available variants.
9. Packed APPS multi-test stdin suites must be scored with segment/line partial credit (`packed_tests=T`), not binary 0/1.
10. PPO must use a frozen SFT `ref_model` (`is_peft_model=False`, `kl_penalty=abs`, `horizon=100`) so KL is measured against the SFT policy, not the base model.

## Corrected notebooks

- `notebooks/05_ppo_training_corrected.ipynb`: small PPO smoke test using the repository implementation and explicit benchmark rewards.
- `notebooks/07_evaluation_and_ablations_corrected.ipynb`: evaluation with explicit checkpoint handling and recorded sample counts.

## PPO validation

- **Smoke Test Status**: **PASSED (2026-09-18)** — pipeline only; metrics from that run are superseded.
- **Verified Training Status**: **PASSED (2026-09-24)** on `junior-A` @ `56ce85a` / `d3cada9`
  - SFT adapter (`./checkpoints/sft/final`) loaded into `AutoModelForCausalLMWithValueHead` with frozen `ref_model`.
  - Supervisor issues **all fixed and verified against real Kaggle logs**:
    1. **Partial/positive rewards**: packed multi-test suites yield `passed=2/8`, `1/4`, `1/3` with `reward=0.25–0.33`; full AC samples at `1.0`.
    2. **KL sign/magnitude**: step 1 `kl=0.000`, steps 2–10 `kl` in `[1.10, 2.17]`, all positive; adaptive `kl_coef` decays `0.0500→0.0482` (correct when KL < target).
    3. **`grad_norm` logging**: non-zero every step (`0.16–0.32`); `param_norm_delta` and `sample_lora_delta` positive throughout.
  - Full step-by-step metrics: see `outputs.md` → “Notebook 05: Reinforcement Learning: PPO Training” (2026-09-24 verified section).
  - Test suite: **49 tests pass** locally (`python -m unittest discover -s tests`).

## Reporting rule

Do not report the previous Notebook 7 percentages as empirical conclusions. Replace them only with results from the corrected, reproducible evaluation runs.

Do not cite the pre-fix PPO smoke/50-step logs (negative KL, `grad_norm=0`, binary-only rewards) as current results — they are retained in `outputs.md` only as historical superseded evidence.
