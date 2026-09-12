# Methodology

## End-to-End Loop

`problem → initial code → execute → feedback → correction → execute → ...`

The model receives the programming problem and generates an initial solution. The generated code is executed in a sandbox. Execution feedback may include traceback, failed-test summary, and execution status. The model then generates a correction for the next turn.

## Execution Status

Track at minimum:
- AC — Accepted
- PE — Presentation/format issue where applicable
- WA — Wrong Answer
- TLE — Time Limit Exceeded
- MLE — Memory Limit Exceeded
- CE — Compilation/Syntax Error
- RE — Runtime Error

## Reward Variants

- Binary final execution reward
- Partial test-pass reward
- Execution-status-aware reward
- Invalid/syntax output penalty

## Baselines

- Zero-shot prompting
- No-feedback debugging
- True execution feedback
- Random/incorrect feedback ablation
- SFT
- PPO
- DPO

## Controlled Variables

Record model, dataset split, prompt format, sampling temperature, top-p, debugging turns, test budget, reward version, random seed, GPU, wall-clock time, and VRAM.

## Evaluation

Primary metrics are Pass@1, Fix@3, Fix@5, execution success, wall-clock time, VRAM usage, token usage where available, and reward stability.
