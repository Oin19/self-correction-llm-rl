# Self-Correction in Small Code Language Models via Execution-Guided Reinforcement Learning

Research project investigating whether small Code LLMs can learn iterative program debugging from execution-guided feedback, and which reward, feedback, and optimization choices enable reliable self-correction.

## Research Questions

1. Can small language models perform agentic debugging?
2. How does the quality and type of execution feedback affect self-correction?
3. Does RL outperform prompting and SFT?
4. How does reward design affect learning?
5. Is DPO a viable alternative to PPO?
6. How many debugging turns are useful?
7. Does learned behavior generalize across benchmarks?

## Primary Model

- `deepseek-ai/deepseek-coder-1.3b-instruct` — used for all runs to date (SFT, PPO-dense, PPO-binary, DPO) and the N=100 evaluation.

Other small Code LLMs (e.g., the Qwen2.5-Coder family) may be evaluated later for cross-family analysis.

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
results/      Evaluation metrics, figures, and selected result artifacts
checkpoints/  Checkpoint instructions; model weights are not committed
```

## Status

- **PPO dense + binary training complete (2026-09-25)**: both paper arms trained 100 steps each with identical HPs/seed (`36c8055`), metadata verified (dense reward vs AC-only binary reward, KL-bounded). See `docs/experiments.md` and `outputs.md`.
- **N=100 five-arm preliminary evaluation complete (2026-09-28)**: Zero-Shot / SFT / PPO-dense / PPO-binary / DPO on the first 100 HumanEval + 100 MBPP problems — identical problems, generation settings, and sample budget for all arms; Exec Success 1.00, Infra Errors 0. **Proposal-basis preliminary results, not final paper results.** Table + figure: `docs/experiment_status.md`, `results/preliminary_eval_n100.png`. The current DPO checkpoint is a caveated preliminary run (base-initialized, 100 pairs) — not a clean RQ5 conclusion.
- **Reserved for final paper**: full five-arm evaluation on HumanEval (164) + MBPP (500), and a clean `train[:500]` DPO retrain (future work — not running now).
- Tests: 60 pass (`python -m unittest discover -s tests`).
