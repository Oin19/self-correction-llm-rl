# Experiment Validation Status

## Current Status

The September 2026 notebook runs are preserved as execution evidence. The current proposal is based on a **four-arm preliminary pilot**:

1. Zero-Shot
2. SFT
3. PPO-Dense
4. PPO-Binary

The pilot is intended to motivate the proposed research rather than establish final benchmark conclusions.

## Corrected Requirements

1. Every scored example must have explicit executable benchmark tests.
2. Empty test suites must never receive an AC score.
3. MBPP `test_list`, HumanEval `test`, and APPS `input_output` must be recognized correctly.
4. PPO must load the trained SFT adapter and must not silently fall back to the base model.
5. PPO reward must be computed from benchmark execution, not merely process exit status.
6. Evaluation must not silently replace a missing checkpoint with the base model.
7. Smoke-test results (N <= 20 per benchmark) are validation evidence only, not final paper results.
8. Final comparisons must use the same fixed evaluation set, decoding settings, and evaluation code for all compared variants.
9. Packed APPS multi-test stdin suites must be scored with segment/line partial credit rather than binary 0/1.
10. PPO must use a frozen SFT reference model so that KL is measured against the intended SFT policy.
11. Paper PPO runs must record reward mode, seed, checkpoint directory, and commit SHA in metadata.
12. DPO preference collection must use the same normalized execution-test harness as PPO/evaluation.
13. Any final multi-arm comparison must use matched initialization, identical evaluation settings, and verified benchmark tests.

## Corrected Notebooks

- `notebooks/05_ppo_training_corrected.ipynb`: PPO with dense/binary reward modes, separate checkpoint directories, fixed seed, and paper-run metadata.
- `notebooks/06_dpo_training.ipynb`: DPO preference collection using the normalized execution-test harness.
- `notebooks/07_evaluation_and_ablations_corrected.ipynb`: evaluation and ablation pipeline with fixed evaluation settings.

## PPO Validation

- **Smoke test:** PASSED (2026-09-18) as pipeline validation; its metrics are superseded.
- **Verified training validation:** PASSED (2026-09-24) on the corrected PPO pipeline.
  - SFT adapter loaded into the PPO policy.
  - Frozen reference policy used for KL measurement.
  - Partial execution rewards were observed on multi-test suites.
  - Positive KL values and non-zero gradient norms were observed during the diagnostic run.
- The diagnostic run used for validation is retained as execution evidence; it is not itself the final paper comparison.
- Test suite status recorded in the repository: 60 tests pass.

## Paper-Run Status

| Component | Status | Notes |
|---|---|---|
| PPO-Dense training | Complete | 100-step pilot run with recorded seed/checkpoint metadata |
| PPO-Binary training | Complete | 100-step pilot run with recorded seed/checkpoint metadata |
| Four-arm preliminary evaluation | Complete, preliminary | Zero-Shot / SFT / PPO-Dense / PPO-Binary; proposal-basis evidence only |
| DPO | Preliminary / not matched | Existing DPO checkpoint was initialized from the base model rather than the matched SFT checkpoint; excluded from PPO-DPO conclusions |
| Full benchmark evaluation | Planned | HumanEval (164) + MBPP (500), after clean execution-harness validation |
| Matched DPO retraining | Planned | Same SFT initialization and normalized execution harness as PPO |
| Feedback ablations | Planned | No feedback, status-only, sanitized traceback/error, and richer structured feedback |
| Turn-budget analysis | Planned | Vary K and analyze cumulative success, conditional repair, marginal gain per turn, latency, and compute |
| APPS transfer evaluation | Planned | Distinguish in-domain behavior from transfer to HumanEval and MBPP |
| Model-scale analysis | Planned if resources permit | Extend protocol to 3B and 7B backbones |

## Preliminary Evaluation: Reproducibility Gate

A preliminary four-arm evaluation on the first 100 HumanEval and first 100 MBPP problems had previously been recorded in the repository.

**Current reporting status:** these numerical results are **not treated as final or reproducible evidence until the HumanEval execution harness has been independently validated and the evaluation is rerun from a clean checkout.**

The repository therefore intentionally does **not** reproduce the previous N=100 percentage table here.

Before reusing those numbers in the proposal or paper:

1. Validate the HumanEval harness on a small manually inspected set.
2. Confirm that correct, wrong, syntax-error, and runtime-error programs receive the expected execution statuses.
3. Run a small smoke evaluation (for example, 5–20 problems).
4. Record the exact repository commit, notebook commit, dataset revision, checkpoint provenance, and evaluation configuration.
5. Rerun the preliminary evaluation from the clean validated pipeline.
6. Only then promote the resulting numbers to proposal/paper evidence.

## DPO Caveat

A preliminary DPO checkpoint exists, but it was initialized from the base model rather than the matched SFT checkpoint and was trained on a limited preference set.

Therefore:

- It is retained as development evidence.
- It is **not** used for a clean PPO-vs-DPO conclusion.
- Matched DPO retraining from the SFT initialization is part of the proposed research.

## Reporting Rule

The repository must clearly distinguish:

- implemented pipeline components,
- validated execution/training behavior,
- preliminary pilot evidence,
- final empirical results,
- proposed future experiments.

Do not report preliminary or smoke-test percentages as final conclusions.

Do not use the old unmatched DPO checkpoint to make a PPO-vs-DPO claim.

Do not promote the previous HumanEval/MBPP N=100 percentages until they have passed the reproducibility gate described above.
