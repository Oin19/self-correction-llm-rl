# Self-Correction in Small Code Language Models via Execution-Guided Reinforcement Learning

Research project investigating how small Code LLMs can learn self-correction from execution feedback, and how reward design, feedback quality, optimization method, and debugging-turn budget affect repair behavior.

## Research Questions

1. Can small language models perform agentic debugging via execution feedback?
2. How does the quality and type of execution feedback affect self-correction?
3. How does reinforcement learning compare with zero-shot prompting and SFT for multi-turn debugging?
4. How does reward design, particularly dense partial-credit versus binary execution reward, affect learning?
5. How does matched DPO compare with PPO for execution-guided self-correction?
6. How does the debugging-turn budget affect cumulative and conditional repair?
7. Does behavior learned on APPS transfer to HumanEval and MBPP?

## Primary Model

- `deepseek-ai/deepseek-coder-1.3b-instruct`
- LoRA is used for parameter-efficient adaptation.

The study may extend the protocol to 3B and 7B backbones if computational resources permit.

## Methods

- Zero-shot prompting
- Supervised Fine-Tuning (SFT)
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)

The research compares these approaches under controlled execution-guided debugging settings.

## Execution-Guided Debugging

The core research loop is:

`problem → initial code → execute → feedback → correction → execute → ...`

Generated Python programs are executed in a sandbox with execution controls. Execution outcomes are used to condition subsequent correction attempts. The study considers multiple debugging budgets, including K = 1, 3, and 5.

## Reward Design

The project implements two execution-based PPO reward formulations:

- **Dense reward:** combines acceptance with partial credit based on tests passed, with penalties for zero-pass execution failures.
- **Binary reward:** accepted programs receive a positive reward while other outcomes receive zero reward.

The goal is to study how reward granularity affects learning and self-correction.

## Proposed Research

The planned study includes:

1. **Full benchmark evaluation:** HumanEval and MBPP under fixed evaluation settings.
2. **Matched DPO training:** train DPO from the same SFT initialization used for PPO comparisons.
3. **Feedback ablations:** compare no feedback, status-only feedback, sanitized error feedback, and richer structured execution feedback.
4. **Turn-budget analysis:** vary the number of debugging turns and measure cumulative success, conditional repair, latency, and compute.
5. **Transfer evaluation:** evaluate behavior across APPS, HumanEval, and MBPP.
6. **Model-scale analysis:** if resources permit, repeat the protocol on larger small-model backbones.

## Evaluation and Analysis

The evaluation is designed to distinguish first-attempt generation quality from actual repair ability. Planned metrics and analyses include:

- Pass@1
- Fix@K
- Conditional repair rate
- Failure-category transitions
- Execution outcomes
- Tokens generated
- Wall-clock latency
- Marginal compute cost per repair turn
- Statistical confidence intervals and paired comparisons
- Analysis by failure type and feedback condition

A no-feedback control uses the same attempt and decoding budget to distinguish genuine use of execution evidence from gains caused by repeated sampling.

## Repository Structure

```text
src/          Core implementation
notebooks/    Ordered experiment notebooks
data/         Data preparation instructions and local data placeholders
configs/      Model, training, and experiment configurations
tests/        Unit tests
docs/         Research and workflow documentation
checkpoints/  Checkpoint instructions; model weights are not committed
```

## Implementation Status

- Execution-guided debugging pipeline implemented
- Sandboxed execution and reward computation implemented
- Zero-shot, SFT, PPO, and DPO training/evaluation components implemented
- Checkpoint provenance and reproducibility tooling implemented
- Unit tests included in `tests/`
- Full benchmark evaluation, matched DPO comparison, feedback ablations, turn-budget analysis, and model-scaling experiments are part of the planned study

## Research Principle

The project does not assume that reinforcement learning, dense rewards, DPO, or additional debugging turns are inherently better. The goal is to identify the conditions under which execution-guided self-correction helps small code models, and when it remains unreliable.
