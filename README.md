# Self-Correction in LLMs via Execution-Guided Reinforcement Learning

Research project investigating whether small Code LLMs can learn iterative program debugging through execution-guided feedback.

## Research Questions

1. Can small language models perform agentic debugging?
2. How does the quality and type of execution feedback affect self-correction?
3. Does RL outperform prompting and SFT?
4. How does reward design affect learning?
5. Is DPO a viable alternative to PPO?
6. How many debugging turns are useful?
7. Does learned behavior generalize across benchmarks?

## Primary Model Family

- Qwen2.5-Coder 1.5B Instruct
- Qwen2.5-Coder 3B Instruct
- Qwen2.5-Coder 7B Instruct

Additional small Code LLMs may be evaluated for cross-family analysis.

## Methods

- Zero-shot prompting
- Supervised Fine-Tuning (SFT)
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)

## Evaluation

- Pass@1
- Fix@3
- Fix@5
- Execution status and error recovery
- Wall-clock time
- VRAM usage
- Reward stability

## Experimental Design

The core debugging loop is:

`problem → initial code → execute → feedback → correction → execute → ...`

Planned execution statuses include AC, WA, TLE, MLE, CE, and RE. Experiments use multiple debugging turns and controlled comparisons across feedback, reward, model-size, and training-method conditions.

## Repository Structure

```text
src/          Core implementation
notebooks/    Ordered experiment notebooks
data/         Data preparation instructions and local data placeholders
configs/      Model, training, and experiment configurations
tests/        Unit tests
docs/         Research and workflow documentation
results/      Metrics, logs, and figures (large outputs kept out of Git)
checkpoints/  Checkpoint instructions; model weights are not committed
```


## Status

Pipeline validation is in progress.

- **PPO training fixes verified (2026-09-24)**: dense partial rewards on packed APPS multi-test suites, healthy positive KL against a frozen SFT reference, and non-zero `grad_norm` logging. See `docs/experiment_status.md` and `outputs.md` (NB05 verified section).
- **Next**: Notebook 07 four-variant evaluation (Zero-Shot / SFT / DPO / PPO); quantitative research claims only after controlled, reproducible runs.
