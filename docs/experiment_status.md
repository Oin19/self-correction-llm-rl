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

## Corrected notebooks

- `notebooks/05_ppo_training_corrected.ipynb`: small PPO smoke test using the repository implementation and explicit benchmark rewards.
- `notebooks/07_evaluation_and_ablations_corrected.ipynb`: evaluation with explicit checkpoint handling and recorded sample counts.

## PPO validation

- **Smoke Test Status**: **PASSED (2026-09-18)**
  - SFT adapter (`./checkpoints/sft/final`) loaded into `AutoModelForCausalLMWithValueHead`.
  - APPS benchmark execution test cases evaluated for code rewards.
  - 2-step PPO rollout completed cleanly without errors (`PPO step 1 reward=-0.200 kl=0.000`, `PPO step 2 reward=-0.200 kl=-0.048`).

## Reporting rule

Do not report the previous Notebook 7 percentages as empirical conclusions. Replace them only with results from the corrected, reproducible evaluation runs.
