# Results

Keep generated logs, metrics, and figures out of Git when they are large.

Each result should be traceable to:
- Git commit SHA
- experiment ID
- model/checkpoint
- dataset split
- training/evaluation configuration
- random seed

Final paper-ready tables and figures should be copied here only when they are reasonably sized and reproducible.

## Current experiment pointers

| Experiment | Status | Where |
|---|---|---|
| `EXP-PPO-2026-09-24` (verified PPO training) | Validation PASSED — supervisor issues fixed | `docs/experiments.md`, `outputs.md` NB05 2026-09-24 |
| `EXP-PPO-DENSE` (RQ3 paper arm) | **Completed 2026-09-25** — 100/100 steps, KL max 4.82, AC 10.5%, ckpt `ppo_dense/final` | `docs/experiments.md`, `outputs.md` NB05 paper arms |
| `EXP-PPO-BINARY` (RQ4 paper arm) | **Completed 2026-09-25** — 100/100 steps, AC 23/200 (11.5%), KL max 4.79, ckpt `ppo_binary/final` | `docs/experiments.md`, `outputs.md` NB05 paper arms |
| `EXP-DPO-RETRAIN` (RQ5) | **Checkpoint exists** — lead-approved NB06 run (base-init, 100 pairs caveat in `outputs.md` NB07) | `docs/experiments.md`, `outputs.md` |
| `EXP-EVAL-RQ3RQ4` (5-arm eval) | **Smoke N=20 + large N=100 completed 2026-09-28** (`0c4da7f`, all arms, Exec 100%, infra 0) — N=100 = proposal-basis preliminary eval; full 164+500 planned for final paper | `outputs.md` NB07; do not use historical NB07 tables |

