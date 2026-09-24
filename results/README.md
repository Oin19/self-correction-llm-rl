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
| `EXP-PPO-DENSE` (RQ3 paper arm) | **Planned** — NB05 `REWARD_MODE=dense`, `MAX_STEPS≥100` | `docs/experiments.md` |
| `EXP-PPO-BINARY` (RQ4 paper arm) | **Planned** — NB05 `REWARD_MODE=binary`, same HPs/seed | `docs/experiments.md` |
| `EXP-DPO-RETRAIN` (RQ5) | **Planned** — NB06 `train[:500]`, `normalize_tests` | `docs/experiments.md` |
| `EXP-EVAL-RQ3RQ4` (5-arm full eval) | **Pending** all training checkpoints | NB07 corrected; do not use historical NB07 tables |

