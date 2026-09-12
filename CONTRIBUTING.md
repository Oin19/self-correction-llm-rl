# Contribution & Team Workflow

## Roles

### Junior 1 — Model & Training
- Model loading and generation
- SFT
- PPO
- DPO
- Training configuration and checkpoints

### Junior 2 — Data, Execution & Evaluation
- APPS preprocessing
- Dataset splits
- Sandboxed code execution
- Traceback / feedback extraction
- Debugging loop
- Reward computation
- Evaluation and metrics

### Co-Lead
- Research direction and experiment design
- Scientific validation
- Pull-request review
- Reproducibility and leakage checks
- Results interpretation and paper integration

## Git Workflow

1. Never push implementation directly to `main`.
2. Start from an up-to-date `main` branch.
3. Create a focused feature branch.
4. Make small, descriptive commits.
5. Push the feature branch.
6. Open a Pull Request into `main`.
7. Co-lead reviews the PR.
8. Address requested changes.
9. Merge only after approval.

### Branch naming

- `junior1/feature-model-loader`
- `junior1/feature-sft`
- `junior1/feature-ppo`
- `junior2/feature-sandbox`
- `junior2/feature-debug-loop`
- `junior2/feature-evaluation`

## Pull Request Checklist

- [ ] The change has one clear purpose.
- [ ] Code runs on the intended environment.
- [ ] Relevant tests were run.
- [ ] No credentials or model weights are included.
- [ ] No raw/private dataset is committed.
- [ ] Experiment configuration is documented.
- [ ] Random seeds and sampling settings are recorded where applicable.
- [ ] Results are reproducible from the committed code/config.
- [ ] The PR explains which research question or milestone it supports.
