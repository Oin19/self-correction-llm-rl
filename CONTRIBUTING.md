# Contribution & Research Workflow

## Team Roles

### Co-Lead — Oindrila Banerjee
- Research direction and experiment design
- Scientific validation
- Experiment review
- Reproducibility and leakage checks
- Results interpretation
- Research proposal and paper integration

### Research Contributor — Rajdeep Bhowmick
- Model and training implementation
- SFT, PPO, and DPO experiments
- Training configurations
- Checkpoint management
- Training logs and experiment artifacts

### Research Contributor — Aihik Basu
- Dataset preparation
- Sandboxed code execution
- Debugging loop
- Evaluation harness
- Metrics and analysis
- Evaluation artifacts

## Git Workflow

1. Work on a feature branch rather than directly on main.
2. Keep commits focused and descriptive.
3. Push experimental or implementation changes to the appropriate branch.
4. Open a Pull Request into main for substantial changes.
5. Co-Lead reviews research-critical changes before merging.
6. Merge only after the relevant code, tests, and experiment provenance have been checked.

## Research Reproducibility

Every reported experiment should record, where applicable:

- Model and checkpoint
- Dataset and split
- Random seed
- Training configuration
- Decoding settings
- Reward configuration
- Debugging-turn budget
- Evaluation settings
- Relevant Git commit SHA

Research results should be traceable to the code, configuration, checkpoint, and evaluation procedure that produced them.

## Experimental Integrity

- Do not report preliminary or smoke-test results as final results.
- Do not compare checkpoints with different initialization or training conditions as if they were matched experiments.
- Record known limitations and experimental caveats.
- Keep evaluation and training data clearly separated where required.
- Do not silently replace missing checkpoints, datasets, or evaluation components with substitutes.
- Verify benchmark tests before using execution results as evidence.

## What Not to Commit

- Model weights or large checkpoint files
- API keys, credentials, or secrets
- Private or restricted datasets
- Large raw experiment artifacts unless specifically required
- Unverified benchmark results presented as final findings

## Pull Request Checklist

- [ ] Code runs in the intended environment.
- [ ] Relevant tests pass.
- [ ] Configuration is documented.
- [ ] Random seed and sampling settings are recorded where applicable.
- [ ] Checkpoint provenance is clear.
- [ ] No data leakage or unintended test-set use is introduced.
- [ ] Research claims are supported by the corresponding experiment.
- [ ] Important limitations or caveats are documented.
- [ ] The change is reproducible from the committed code/configuration.

## Research Principle

The repository should distinguish clearly between what has been implemented, what has been experimentally validated, what is preliminary evidence, and what remains proposed future work.
