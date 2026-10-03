# Self-Correction in Small Code Language Models via Execution-Guided Reinforcement Learning

Research project investigating when small Code LLMs can learn useful self-correction from execution feedback, and how reward design, feedback quality, optimization method, and debugging-turn budget affect repair behavior.

## Research Questions

1. Can small language models perform agentic debugging via execution feedback?
2. How does the quality and type of execution feedback affect self-correction?
3. How does reinforcement learning compare with zero-shot prompting and SFT for multi-turn debugging?
4. How does reward design, particularly dense partial-credit versus binary execution reward, affect learning?
5. How does matched DPO compare with PPO for execution-guided self-correction?
6. How does the debugging-turn budget affect cumulative and conditional repair?
7. Does behavior learned on APPS transfer to HumanEval and MBPP?

## Primary Model

- `deepseek-ai/deepseek-coder-1.3b-instruct` — the primary model used in the completed pilot and current training pipeline.
- LoRA is used for parameter-efficient adaptation.

The proposed study may extend the protocol to 3B and 7B backbones if computational resources permit.

## Methods

- Zero-shot prompting
- Supervised Fine-Tuning (SFT)
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)

The completed pilot uses four matched evaluation arms: Zero-Shot, SFT, PPO-Dense, and PPO-Binary. DPO is part of the proposed research, but the preliminary DPO checkpoint was initialized from the base model rather than the matched SFT checkpoint and is therefore excluded from PPO-DPO conclusions.

## Execution-Guided Debugging

The core research loop is:

`problem → initial code → execute → feedback → correction → execute → ...`

Generated Python programs are executed in a sandbox with execution controls. The pipeline records execution outcomes and uses feedback from failed attempts to condition subsequent corrections. The proposed study considers multiple debugging budgets, including K = 1, 3, and 5, with additional turn-budget analysis planned.

## Reward Design

The pilot compares two PPO reward formulations under matched training conditions:

- **PPO-Dense:** accepted programs receive 1.0; non-accepted programs can receive partial credit based on tests passed, with penalties for zero-pass execution failures.
- **PPO-Binary:** accepted programs receive 1.0 and all other outcomes receive 0.

The purpose is to study whether richer execution-based reward information improves learning rather than to assume that one reward formulation is superior.

## Preliminary Pilot

A completed DeepSeek-Coder 1.3B pilot evaluates:

- Zero-Shot
- SFT
- PPO-Dense
- PPO-Binary

on the first 100 problems from HumanEval and the first 100 problems from MBPP.

Reported metrics are:

- Pass@1 — first-attempt success
- Fix@3 — cumulative success by the third correction attempt
- Fix@5 — cumulative success by the fifth correction attempt
- Conditional repair rate — planned for the full analysis
- Execution status and error recovery
- Tokens, latency, and compute cost — planned for expanded evaluation

The pilot is preliminary evidence and is used to motivate the proposed controlled experiments. It does not establish that RL is universally better than zero-shot prompting or SFT.

## Proposed Research

The next stages are:

1. **Full benchmark evaluation:** HumanEval (164) and MBPP (500) under fixed evaluation settings and gate-verified checkpoints.
2. **Matched DPO retraining:** train DPO from the same SFT initialization and use the same normalized execution harness as PPO.
3. **Feedback ablations:** compare no feedback, status-only feedback, sanitized traceback/error feedback, and richer structured execution feedback.
4. **Turn-budget analysis:** vary K and report cumulative success, conditional repair, marginal gain per turn, latency, and compute.
5. **Transfer evaluation:** use held-out APPS evaluation to distinguish in-domain behavior from transfer to HumanEval and MBPP.
6. **Model-scale analysis:** if resources permit, repeat the protocol on 3B and 7B backbones.

## Evaluation and Analysis

The study is designed to separate first-attempt generation quality from actual repair ability. In addition to Pass@1 and cumulative Fix@K, the planned analysis includes:

- Conditional repair rate among first-attempt failures
- Failure-category transitions
- Execution-harness failures
- Tokens generated
- Wall-clock latency
- Marginal compute cost per repair turn
- Bootstrap confidence intervals
- Paired statistical tests for full-benchmark comparisons
- Analysis by failure type and feedback condition

A no-feedback control will use the same attempt and decoding budget to distinguish genuine use of execution evidence from gains caused only by repeated sampling.

## Repository Structure

```text
src/          Core implementation
notebooks/    Ordered experiment notebooks
data/         Data preparation instructions and local data placeholders
configs/      Model, training, and experiment configurations
tests/        Unit tests
docs/         Research and workflow documentation
results/      Evaluation metrics, figures, and selected result artifacts
checkpoints/  Checkpoint instructions; model weights are not committed
```

## Status

- **Pipeline implemented:** execution-guided debugging, sandboxed execution, reward computation, evaluation utilities, checkpoint-provenance checks, and reproducibility tooling are in place.
- **PPO training complete:** dense-reward and binary-reward PPO pilot arms were trained for 100 steps with matched hyperparameters/seed and verified metadata.
- **Four-arm preliminary evaluation complete:** Zero-Shot / SFT / PPO-Dense / PPO-Binary were evaluated on the first 100 HumanEval and first 100 MBPP problems. These results are preliminary pilot evidence, not final full-benchmark results.
- **DPO:** a preliminary 100-pair DPO checkpoint exists, but it was initialized from the base model rather than the matched SFT checkpoint. It is excluded from PPO-DPO conclusions; matched DPO retraining is planned.
- **Next major evaluation:** full HumanEval (164) + MBPP (500), followed by matched DPO, feedback ablations, turn-budget analysis, and APPS transfer evaluation.
- **Model scaling:** 3B/7B experiments are planned if computational resources permit.
- **Tests:** 60 pass (`python -m unittest discover -s tests`).

## Research Principle

The project does not assume that reinforcement learning, dense rewards, DPO, or additional debugging turns are inherently better. The goal is to identify the conditions under which execution-guided self-correction helps small code models, and when it remains unreliable.
