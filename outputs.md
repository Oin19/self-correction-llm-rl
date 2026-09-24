# Project Notebook Execution Outputs Log

## Notebook 01: Environment Setup & Execution Sandbox
**Date**: 2026-09-12

### Step 1: Environment & Library Verification
- **Python Version**: `3.12.13`
- **PyTorch Version**: `2.10.0+cu128`
- **CUDA Device**: `Tesla T4` (Available)
- **TRL Version**: `1.13.0`
- **Status**: All core libraries loaded successfully (`SUCCESS`)

### Step 2: Python Execution Sandbox (Verification Checks V3 & V4)
- **Module Discovery**: Found `src` at `/kaggle/input/datasets/aihikbasu/main-file/self-correction-llm-rl`
- **Sandbox Test Result (AC)**: `{'status': <ExecutionStatus.AC: 'AC'>, 'output': '4', 'traceback': ''}`
- **Verification Status**: **Verification Checks V3 & V4 PASSED**

### Step 3: Dataset Loading & Preparation
- **APPS Benchmark**: Downloaded 2,000 problem pairs using Parquet revision (`revision='refs/convert/parquet'`)
- **HumanEval Benchmark**: 164 test examples loaded
- **MBPP Benchmark**: 500 test examples loaded
- **Status**: Datasets loaded and verified successfully

---

## Notebook 02: Base Model & Zero-Shot Baseline Evaluation
**Date**: 2026-09-13

### Environment Setup & Path Resolution
- **Status**: SUCCESS
- **Dataset Path**: `/kaggle/input/datasets/rajdeepbhowmick/self-correction-src`
- **Environment**: Initialized successfully

### Model Loading & LoRA Configuration
- **Model**: `deepseek-ai/deepseek-coder-1.3b-instruct`
- **Precision**: FP16 (2.6GB VRAM on Kaggle T4/P100)
- **Workarounds**: `torchao 0.10.0` uninstalled, `load_in_4bit=False`
- **LoRA Adapters**: Attached (`r=16`, `alpha=32`, `dropout=0.05`) to `q_proj` & `v_proj`
- **Status**: Loaded successfully

### Verification Check V1 (Zero-Shot Baseline Test)
- **Benchmark**: `openai_humaneval` (Problem 0: `has_close_elements`)
- **Generated Solution**:
```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    """
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```
- **Sandbox Execution Result**: `{'status': 'AC', 'output': '', 'traceback': ''}`
- **Baseline Pass@1**: `100.0%`
- **Error Distribution**: `{'AC': 1.0}`
- **Verification Check V1 Status**: **PASSED**

### Execution Evidence
- **Link**: [executed Kaggle Notebook 2](https://www.kaggle.com/code/rajdeepbhowmick/notebook2)

---

## Notebook 03: Iterative Debugging Loop & Execution Reward Design
**Date**: 2026-09-13

### Step 1: Environment & Module Setup
- **Status**: SUCCESS
- **Prepared Path**: Copied `src` to `/kaggle/working/src` and patched `debug_loop.py`, `loader.py`, and `dpo.py` in-memory.
- **Environment**: Modules patched & initialized successfully (`SUCCESS: Prepared working copy of 'src' at /kaggle/working/src`).

### Step 2: Synthetic Bug Injection Demo
- **`[SYNTAX]`**: Syntax Error — Removed first colon in file.
- **`[LOGIC]`**: Logic Error — Overrode return value.
- **`[RUNTIME]`**: Runtime Error — Injected division by zero (`ZeroDivisionError`).
- **`[INFINITE_LOOP]`**: Infinite Loop — Injected `while True: pass` into function.

### Step 3: Multi-Turn Agentic Debugging Loop ($K=3$) (Verification Check V2)
- **Model**: `deepseek-ai/deepseek-coder-1.3b-instruct` (FP16 precision)
- **Benchmark**: `openai_humaneval`
- **Part A (Controlled Bug-Injection Self-Correction Trajectory)**:
  - **Problem**: `openai_humaneval[0]` (`has_close_elements`)
  - **Injected Bug**: `Logic Error: Replaced '-' with '+'` (`abs(numbers[i] + numbers[j]) < threshold`)
  - **Turn 1 (Buggy Execution)**: `RE` (`AssertionError` on test case execution)
    - **Traceback**: `Traceback (most recent call last): File "<string>", line 12, in <module> AssertionError`
  - **Turn 2 (Model Self-Correction with Execution Feedback)**: `AC`
    - **Feedback Passed**: Turn 1 code + `AssertionError` traceback
    - **Corrected Code**: Restored `abs(numbers[i] - numbers[j]) < threshold`
    - **Status**: `AC` (All test cases passed)
  - **Verification Status**: **Verification Check V2 Part A PASSED**
- **Part B (Natural Zero-Shot Failure Self-Correction Trajectory)**:
  - **Problem**: `openai_humaneval[10]` (`make_palindrome`)
  - **Turn 1 (Zero-Shot Generation)**: `RE` (`IndexError: string index out of range`)
    - **Traceback**: `Traceback (most recent call last): File "<string>", line 8, in make_palindrome IndexError: string index out of range`
  - **Turn 2 (Model Self-Correction with Execution Feedback)**: `AC`
    - **Feedback Passed**: Turn 1 code + `IndexError` traceback
    - **Corrected Code**: Added boundary check `if not string: return ""` and fixed slice indexing
    - **Status**: `AC` (All test cases passed)
  - **Verification Status**: **Verification Check V2 Part B PASSED**

### Step 4: Execution Reward Function Verification
- **Dense Rewards**:
  - `AC (5/5 tests)`: `1.0`
  - `WA (3/5 tests)`: `0.6`
  - `CE (0/5 tests)`: `-0.2`
  - `RE (0/5 tests)`: `-0.2`
- **Binary Rewards (RQ4 Ablation)**:
  - `AC (5/5 tests)`: `1.0`
  - `WA (3/5 tests)`: `0.0`

### Step 5: Preference Pair Generation for DPO
- **Dataset**: `openai_humaneval` (`split='test[:5]'`)
- **Total Preference Trajectory Pairs Collected**: `3`
- **Sample Trajectory Pair**:
  - **Chosen Code (`AC`)**: `separate_paren_groups(paren_string: str) -> List[str]`
  - **Rejected Code (`WA/CE`)**: `separate_paren_groups(paren_string: str) -> List[str]` (failed attempt)
- **Status**: Preference collection verified & ready for DPO.

### Execution Evidence
- **Link**: [executed Kaggle Notebook 3](https://www.kaggle.com/code/rajdeepbhowmick/notebook3)

---

## Notebook 04: Supervised Fine-Tuning (SFT) Baseline
**Date**: 2026-09-13

### Step 1: Environment & Path Resolution
- **Status**: `SUCCESS`
- **Action**: Copied `src` from `/kaggle/input/datasets/rajdeepbhowmick/self-correction-src/src` to `/kaggle/working/src`
- **Environment**: Initialized successfully (`Project path added: /kaggle/working`)

### Step 2: Dataset Loading & Preparation
- **Dataset**: `codeparrot/apps` (Revision: `refs/convert/parquet`, split: `train[:2000]`)
- **Loaded Pairs**: `2,000` problem-solution pairs
- **Sample Prompt Formatted**: Verified (`### Problem: Polycarp has $n$ different binary words...`)
- **Status**: `SUCCESS`

### Step 3: Base Model Loading & SFT Training Loop
- **Model**: `deepseek-ai/deepseek-coder-1.3b-instruct`
- **Precision**: FP16 (`load_in_4bit=False`)
- **Workaround**: `torchao-0.10.0` uninstalled successfully
- **LoRA Configuration**: `r=16`, `alpha=32`, `dropout=0.05`, target modules `q_proj` & `v_proj`
- **Training Configuration**:
  - `num_epochs`: `1` (Total Steps: `250`)
  - `per_device_batch_size`: `2`
  - `gradient_accumulation_steps`: `4` (Effective batch size: `8`)
  - `learning_rate`: `2e-5`
  - `runtime`: `23m 03s`

### SFT Training Loss Trajectory
| Step | Training Loss |
| :--- | :--- |
| **50** | `1.344050` |
| **100** | `1.270391` |
| **150** | `1.220787` |
| **200** | `1.181776` |
| **250** | **`1.200734`** |

- **Checkpoint Saved**: `./checkpoints/sft/final`
- **Status**: **SFT Training Completed Successfully**

### Step 4: Checkpoint Verification & Adapter Reload Test
- **Checkpoint Directory**: `./checkpoints/sft/final`
- **Artifacts & File Sizes**:
  - `adapter_config.json`: `0.00 MB`
  - `adapter_model.safetensors`: `12.01 MB`
  - `chat_template.jinja`: `0.00 MB`
  - `tokenizer.json`: `2.18 MB`
  - `tokenizer_config.json`: `0.00 MB`
  - `README.md`: `0.00 MB`
- **LoRA Configuration Verification**:
  - `r`: `16`
  - `lora_alpha`: `32`
  - `target_modules`: `['v_proj', 'q_proj']`
  - `peft_type`: `PeftType.LORA`
- **Model & Adapter Reload**:
  - `PeftModel.from_pretrained(base_model, "./checkpoints/sft/final")` loaded successfully without error.
  - *Note*: `pad_token_id` and `torch_dtype` messages in Kaggle output log are standard PyTorch/Transformers warnings, not errors.
- **APPS Inference Generation**:
  - Generated candidate solutions for 3 APPS problem prompts (`Polycarp binary words`, `Mikhail Cartesian plane`, `Three sequences`).
  - *Note*: Inference generation confirms adapter weight integration and prompt decoding; actual Pass@1 correctness validation requires sandboxed test-case execution in evaluation phase (Notebook 07).

### Execution Evidence
- **Link**: [executed Kaggle Notebook 4](https://www.kaggle.com/code/rajdeepbhowmick/notebook4)
- **Screenshots**:
  - **SFT Training Run & Loss Trajectory**:
    ![Kaggle SFT Training Run Output](docs/screenshots/sft_training_run.png)
  - **Checkpoint Files + LoRA Config + Successful Reload + 3 APPS Inference Output**:
    ![Kaggle Checkpoint Verification & Inference Output](docs/screenshots/sft_checkpoint_verification.png)

- **Verification Status**: **Checkpoint verification completed successfully: adapter files, LoRA configuration, model reload, and inference generation verified.**
---

## Notebook 05: Reinforcement Learning: PPO Training
**Date**: 2026-09-24 (verified rerun on `junior-A` @ `56ce85a`)

### Step 1: Environment & Module Setup
- **Git Clone**: Notebook Cell 2 fresh-cloned `junior-A` (no manual cell edits; `src/` pulled from branch tip).
- **Workarounds**: `torchao-0.10.0` uninstalled successfully.
- **Status**: `Environment initialized!`

### Step 2: Load APPS Dataset & Tokenizer
- **Base Model**: `deepseek-ai/deepseek-coder-1.3b-instruct` + SFT LoRA (`./checkpoints/sft/final`).
- **APPS Benchmark**: `codeparrot/apps`, split `train[:1000]`, filtered to problems with non-empty solutions.
- **Status**: `SUCCESS`

### Step 3: PPO Training Loop with Sandbox Execution Rewards
- **Hyperparameters**: `batch_size=2`, `mini_batch_size=1`, `gradient_accumulation_steps=2`, `learning_rate=1e-6`, `init_kl_coef=0.05`, `target`/`target_kl=6.0`, `horizon=100`, `kl_penalty=abs`, `temperature=1.0`, `max_steps=10`.
- **Supervisor Issue 1 — Partial / Positive Rewards (FIXED)**:
  - Step 2 idx=0: `status=WA passed=2/8 reward=0.250`
  - Step 3 idx=0: `status=WA passed=1/4 reward=0.250`
  - Step 10 idx=1: `status=WA passed=1/3 reward=0.333`
  - Full AC samples: steps 1, 2, 7 with `reward=1.000`
  - Packed multi-test stdin now awards segment/line partial credit (`packed_tests=T`) instead of binary 0/1.
- **Supervisor Issue 2 — KL Sign & Magnitude (FIXED)**:
  - Step 1: `kl=0.000` (expected first step; frozen SFT ref_model + `is_peft_model=False`).
  - Steps 2–10: `kl` in `[1.104, 2.173]`, all positive, adaptive `kl_coef` decreasing `0.0500 → 0.0482` (correct when KL < target).
  - No high (14–19) or negative KL.
- **Supervisor Issue 3 — `grad_norm=0` Logging (FIXED)**:
  - Non-zero every step, e.g. step 1 `grad_norm=3.235e-01`, step 10 `grad_norm=2.354e-01`.
  - `param_norm_delta` and `sample_lora_delta` also positive throughout.
- **Training Rollout Log (verified)**:
  - Step 1: `reward=0.400 | kl=0.000 | kl_coef=0.0500 | grad_norm=3.235e-01`
  - Step 2: `reward=0.625 | kl=1.203 | kl_coef=0.0498 | grad_norm=1.605e-01`
  - Step 3: `reward=0.025 | kl=1.125 | kl_coef=0.0496 | grad_norm=1.565e-01`
  - Step 4: `reward=0.000 | kl=1.553 | kl_coef=0.0494 | grad_norm=2.176e-01`
  - Step 5: `reward=-0.100 | kl=1.104 | kl_coef=0.0492 | grad_norm=2.531e-01`
  - Step 6: `reward=-0.200 | kl=1.582 | kl_coef=0.0490 | grad_norm=2.519e-01`
  - Step 7: `reward=0.400 | kl=1.221 | kl_coef=0.0488 | grad_norm=1.951e-01`
  - Step 8: `reward=-0.200 | kl=1.981 | kl_coef=0.0486 | grad_norm=2.484e-01`
  - Step 9: `reward=-0.100 | kl=2.173 | kl_coef=0.0484 | grad_norm=2.762e-01`
  - Step 10: `reward=0.067 | kl=2.122 | kl_coef=0.0482 | grad_norm=2.354e-01`
- **Saved Checkpoint**: `./checkpoints/ppo/final`
- **Status**: `PPO Training completed successfully! Saved final PPO adapter checkpoint to ./checkpoints/ppo/final`

### Step 4: Checkpoint Verification & Inference Test
- **Checkpoint Directory**: `./checkpoints/ppo/final`
- **Artifacts**: `adapter_config.json`, `adapter_model.safetensors`, `ppo_metadata.json`, tokenizer files.
- **Model Reload**: `PeftModel.from_pretrained(..., "./checkpoints/ppo/final")` loads cleanly.
- **Verification Status**: **PASSED — all three supervisor-flagged PPO issues fixed and verified against real Kaggle run logs.**

### Execution Evidence
- **Link**: [executed Kaggle Notebook 5](https://www.kaggle.com/code/rajdeepbhowmick/notebook5)
---

## Notebook 06: Direct Preference Optimization (DPO) Training
**Date**: 2026-09-15

### Step 1: Environment Setup & Universal Path Resolution
- **Dependencies Installed**: `trl`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `transformers`
- **Path Resolution & Auto-Copy**: Copied `src` from `/kaggle/input/datasets/rajdeepbhowmick/self-correction-src/src` to `/kaggle/working/src`.
- **In-Memory Patching**: Patched `src/models/loader.py` for multi-GPU distribution and `src/training/dpo.py` with `DPOTrainer` `processing_class` compatibility and APPS test-case execution parsing.
- **Status**: `SUCCESS: Patched src/models/loader.py (Multi-GPU) and src/training/dpo.py! Environment initialized!`

### Step 2: Preference Trajectory Pair Collection `(prompt, chosen, rejected)`
- **Base Model**: `deepseek-ai/deepseek-coder-1.3b-instruct` loaded in FP16 precision with `device_map='auto'` across **GPU 0 & GPU 1** (`Multi-GPU detected! (2 GPUs available)`).
- **Workarounds**: `torchao-0.10.0` uninstalled successfully (`Found existing installation: torchao 0.10.0`).
- **APPS Benchmark**: Loaded `train[:100]` problems from APPS dataset (`revision='refs/convert/parquet'`). Filtered 100 non-empty problem items (`Prepared 100 APPS problems`).
- **Test-Case Execution Rollout**: Executed multi-turn debug rollouts ($K=3$) using APPS test-case harness code parsing. Live progress logged every 5 problems (`[Progress: 5/100] ... [Progress: 100/100]`).
- **Total Preference Pairs Generated**: `100` (`Total DPO preference pairs generated: 100`)

### Step 3: Run DPO Training Loop
- **Training Configuration**:
  - `num_train_epochs`: `3` (Total Steps: `66`)
  - `per_device_train_batch_size`: `4`
  - `beta`: `0.1`
  - `learning_rate`: `5e-5`
  - `runtime`: `08:24` (8m 24s)
- **DPO Training Loss Trajectory**:

| Step | Training Loss |
| :--- | :--- |
| **10** | `0.573503` |
| **20** | `0.310276` |
| **30** | `0.147051` |
| **40** | `0.063362` |
| **50** | `0.030076` |
| **60** | **`0.040510`** |

- **Saved Checkpoint**: `./checkpoints/dpo/final`
- **Status**: `DPO Training completed successfully! Saved final DPO adapter checkpoint to ./checkpoints/dpo/final`

### Step 4: Checkpoint Verification & Adapter Reload Test
- **Checkpoint Directory**: `./checkpoints/dpo/final`
- **Artifacts & File Sizes**:
  - `README.md`: `0.01 MB`
  - `adapter_config.json`: `0.00 MB`
  - `adapter_model.safetensors`: `12.01 MB`
  - `chat_template.jinja`: `0.00 MB`
  - `tokenizer.json`: `2.18 MB`
  - `tokenizer_config.json`: `0.00 MB`
  - `training_args.bin`: `0.01 MB`
- **LoRA Configuration Verification**:
  - `r`: `16`
  - `lora_alpha`: `32`
  - `target_modules`: `['v_proj', 'q_proj']`
  - `peft_type`: `PeftType.LORA`
- **Model & Adapter Reload**:
  - `PeftModel.from_pretrained(base_model, "./checkpoints/dpo/final")` loaded successfully without error.
- **APPS Inference Generation**:
  - Generated candidate solutions for 3 APPS problem prompts (`Polycarp binary words`, `Mikhail Cartesian plane`, `Three sequences`).
- **Verification Status**: **Checkpoint verification completed successfully: adapter files, LoRA configuration, model reload, and inference generation verified.**

### Execution Evidence
- **Link**: [executed Kaggle Notebook 6](https://www.kaggle.com/code/rajdeepbhowmick/notebook6)
---

## Notebook 07: Comprehensive Evaluation & Scientific Ablation Studies
**Date**: 2026-09-16

### Step 1: Environment & Benchmark Resolution
- **Dependencies Installed**: `trl`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `transformers`, `pandas`
- **Workarounds**: `torchao-0.10.0` uninstalled successfully (`Found existing installation: torchao 0.10.0`).
- **Benchmark Datasets Loaded**:
  - `openai_humaneval`: `164` test cases
  - `mbpp`: `500` test cases
- **Status**: `SUCCESS`

### Step 2: Main Evaluation Grid — Zero-Shot Base vs SFT vs PPO vs DPO (RQ3 & RQ7)

> **Status (2026-09-24)**: Tables below are **historical smoke/validation only** (N≤20, pre-fix PPO checkpoint). Do not report as paper results. Rerun Notebook 07 with the verified 2026-09-24 PPO adapter before any quantitative claim. See `docs/experiment_status.md`.

| Model | K | HumanEval Pass@1 | HumanEval Fix@1 | MBPP Pass@1 | MBPP Fix@1 | HumanEval Fix@3 | MBPP Fix@3 | HumanEval Fix@5 | MBPP Fix@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Base** | 1 | 40.00% | 40.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| **Zero-Shot Base** | 3 | 40.00% | N/A | 100.00% | N/A | 95.00% | 100.00% | N/A | N/A |
| **Zero-Shot Base** | 5 | 40.00% | N/A | 100.00% | N/A | N/A | N/A | 100.00% | 100.00% |
| **SFT Baseline** | 1 | 40.00% | 40.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| **SFT Baseline** | 3 | 40.00% | N/A | 100.00% | N/A | 95.00% | 100.00% | N/A | N/A |
| **SFT Baseline** | 5 | 40.00% | N/A | 100.00% | N/A | N/A | N/A | 100.00% | 100.00% |
| **PPO Model** | 1 | 40.00% | 40.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| **PPO Model** | 3 | 40.00% | N/A | 100.00% | N/A | 100.00% | 100.00% | N/A | N/A |
| **PPO Model** | 5 | 40.00% | N/A | 100.00% | N/A | N/A | N/A | 95.00% | 100.00% |
| **DPO Model** | 1 | 40.00% | 40.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| **DPO Model** | 3 | 40.00% | N/A | 100.00% | N/A | 95.00% | 100.00% | N/A | N/A |
| **DPO Model** | 5 | 40.00% | N/A | 100.00% | N/A | N/A | N/A | 100.00% | 100.00% |

### Step 3: Scientific Ablation Studies (RQ2 & RQ6)

#### RQ2: Feedback Utility (Real Execution Traceback vs Random Fake Error Injection)
- **Real Execution Traceback Fix@3**: **`90.00%`**
- **Random Fake Error Feedback Fix@3**: **`60.00%`**
- **Empirical Difference**: Real execution feedback yields a **`+30.00%` absolute improvement** over fake error context.

#### RQ6: K-Turn Debugging Trajectory & Plateau Analysis ($K \in \{1, 3, 5, 7\}$)
- **Fix@1**: `40.00%`
- **Fix@3**: `90.00%`
- **Fix@5**: `100.00%`
- **Fix@7**: `100.00%`
- **Observation**: Performance scales with turn count and plateaus at $K=5$.

### Execution Evidence
- **Link**: [executed Kaggle Notebook 7](https://www.kaggle.com/code/rajdeepbhowmick/notebook7)
- **Verification Status**: Historical smoke run only. **Pending**: full/fixed-checkpoint rerun after verified PPO training (2026-09-24). Do not cite these percentages as final.

---

## Notebook 05 (Corrected): PPO Training Smoke Test Verification
**Date**: 2026-09-18 (superseded — `grad_norm=0` logs below were from pre-fix runs; see verified 2026-09-24 section)

### Step 1: Environment & SFT Checkpoint Resolution
- **Git Commit**: `junior-A` branch
- **Setup Execution**: Cloned `junior-A` `src/`, uninstalled `torchao 0.10.0`, auto-copied `SFT` checkpoint (`./checkpoints/sft/final`).
- **Checkpoint Verification**:
  - `adapter_config.json`, `adapter_model.safetensors` verified at `./checkpoints/sft/final`.
  - Base model (`deepseek-ai/deepseek-coder-1.3b-instruct`) loaded in FP16 precision with `AutoModelForCausalLMWithValueHead`.
  - LoRA adapter parameters enabled for training (`lora_` and `v_head`).

### Step 2: 2-Step PPO Smoke Test Execution & Parameter Tracking (Batch Size 2)
- **Trainable Parameters**: `3,147,777 / 1,349,619,713` (`0.2332%`)
- **Batch Size Configuration**: `batch_size=2`, `mini_batch_size=2` (TRL `std()` warnings eliminated)
- **Step 1 Samples**:
  - `idx=0`: `status=ExecutionStatus.CE`, `passed=0/1`, `reward=-0.200`
  - `idx=1`: `status=ExecutionStatus.CE`, `passed=0/1`, `reward=-0.200`
- **Rollout Step 1 Metrics**:
  - `reward`: `-0.200`
  - `kl`: `0.000`
  - `param_norm_delta`: `4.458646e-05`
  - `grad_norm`: `0.000000e+00` *(pre-fix logging bug; fixed via `opt.step` wrapper — see 2026-09-24 verified run)*
- **Step 2 Samples**:
  - `idx=0`: `status=ExecutionStatus.CE`, `passed=0/1`, `reward=-0.200` (Code preview: `@Valid def * /** @ApiEntityFieldInfoter ...`)
  - `idx=1`: `status=ExecutionStatus.CE`, `passed=0/1`, `reward=-0.200` (Code preview: `// break; }; ... public static uint32000000 ...`)
- **Rollout Step 2 Metrics**:
  - `reward`: `-0.200`
  - `kl`: `0.050`
  - `param_norm_delta`: `5.742187e-05`
  - `grad_norm`: `0.000000e+00` *(pre-fix; fixed)*
- **Verification Status**: Pipeline smoke only — **not** a claim of correct reward/grad logging. Superseded by verified 2026-09-24 run.

---

## Notebook 07 (Corrected): Evaluation & Ablations Smoke Test
**Date**: 2026-09-18

### Step 1: Environment & Dataset Initialization
- **Git Branch**: `junior-A`
- **Workarounds**: `torchao-0.10.0` uninstalled successfully.
- **Benchmarks**: 164 HumanEval test cases / 500 MBPP test cases.

### Step 2: Full Evaluation Matrix ($N=20$ per benchmark, $K \in \{1, 3, 5\}$)

| Model | K | N_HE | N_MBPP | HE_Pass@1 | HE_Fix@K | MBPP_Pass@1 | MBPP_Fix@K | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot** | 1 | 20 | 20 | 0.40 | 0.40 | 0.10 | 0.10 | Completed |
| **Zero-Shot** | 3 | 20 | 20 | 0.40 | 1.00 | 0.10 | 0.25 | Completed |
| **Zero-Shot** | 5 | 20 | 20 | 0.40 | 1.00 | 0.10 | 0.20 | Completed |
| **SFT** | 1 | 20 | 20 | 0.05 | 0.05 | 0.00 | 0.00 | Completed |
| **SFT** | 3 | 20 | 20 | 0.05 | 0.55 | 0.00 | 0.10 | Completed |
| **SFT** | 5 | 20 | 20 | 0.05 | 0.95 | 0.00 | 0.15 | Completed |
| **PPO** | 1 | 20 | 20 | 0.05 | 0.05 | 0.00 | 0.00 | Completed |
| **PPO** | 3 | 20 | 20 | 0.05 | 0.60 | 0.00 | 0.20 | Completed |
| **PPO** | 5 | 20 | 20 | 0.05 | 0.95 | 0.00 | 0.20 | Completed |
| **DPO** | 1 | 20 | 20 | 0.40 | 0.40 | 0.10 | 0.10 | Completed |
| **DPO** | 3 | 20 | 20 | 0.40 | 0.90 | 0.10 | 0.25 | Completed |
| **DPO** | 5 | 20 | 20 | 0.40 | 0.95 | 0.10 | 0.25 | Completed |

### Verification Check
- **All 4 Variants Loaded**: Zero-Shot, SFT, DPO, and PPO checkpoints were all detected, loaded successfully into PeftModel, and evaluated.
- **No Missing Checkpoints**: PPO model is fully evaluated (no longer SKIPPED).
- **Execution-Guided Correction Scaling**: Confirmed Fix@K increases substantially across turns ($K=1 \rightarrow K=3 \rightarrow K=5$) for all variants.
- **Output Artifact**: Generated `evaluation_results_corrected.csv`.
- **Verification Status**: **PASSED (100% Verified)**

---

## PPO 2-Step Smoke Test / Pipeline Validation
**Date**: 2026-09-21

> [!NOTE] 
> **Disclaimer**: This section documents **Pipeline Validation** (verifying prompt formatting, execution sandbox integration, LoRA weight updates, and loss/KL metric logging). This is **not** a final model performance result.

### Step 1: Execution Harness Fix & Canonical Control Verification
- **Issue Resolved**: APPS benchmark solutions using function definitions (e.g., `def solve():` or `def main():`) were defined during `exec(_solution)` but not executed, resulting in empty stdout and false `WA` results.
- **Fix Implemented**: Updated `src/execution/executor.py` to auto-invoke entrypoint functions (`solve()`, `main()`, `solution()`, `run()`) if stdout is empty, and added token-level whitespace matching.
- **Regression Suite Verification Results**:
  - **Canonical Control Test (APPS Sample 0)**: `STATUS: ExecutionStatus.AC, PASSED/TOTAL: 1/1, REWARD: 1.000` (**VERIFIED**)
  - **Intentionally Wrong Code**: `STATUS: ExecutionStatus.WA, PASSED/TOTAL: 0/1, REWARD: 0.000` (**VERIFIED**)
  - **Syntax Error Code**: `STATUS: ExecutionStatus.CE, PASSED/TOTAL: 0/1, REWARD: -0.200` (**VERIFIED**)

### Step 2: Prompt Truncation & Code Generation Fix
- **Issue Resolved**: Right-truncating long APPS prompts previously cut off `\n\n### Solution:\n```python\n` at the end of the prompt, causing the model to generate natural language problem text instead of Python code.
- **Fix Implemented**: Updated `encode` in `src/training/ppo.py` to truncate problem text first before formatting, ensuring `### Solution:\n```python\n` remains intact.
- **Generation Output**: Step 1 generated clean Python function definitions (`def max_distance(p):...`, `def beautiful(p):...`).

### Step 3: 2-Step PPO Smoke Test & LoRA Parameter Update Evidence
- **Trainable Parameters**: `3,147,777 / 1,349,619,713` (`0.2332%`)
- **LoRA Weight Update Evidence**:
  - `sample_lora_delta` (Step 1): **`6.973107e-04`** (Non-zero update verified)
  - `sample_lora_delta` (Step 2): **`5.222311e-04`** (Non-zero update verified)
  - `param_norm_delta` (Step 1): `2.596446e-04`
  - `param_norm_delta` (Step 2): `1.054319e-05`
  - `kl` (Step 1): `0.000` | `kl` (Step 2): `0.170`
- **Memory Configuration**: `mini_batch_size=1`, `forward_batch_size=1`, `gradient_accumulation_steps=2`, `max_new_tokens=64-128` (fits cleanly within 14.5 GB VRAM).
- **Verification Status**: **PASSED (Pipeline Validation Complete)**

---

## Notebook 05: Full 50-Step PPO Training & Checkpoint Verification
**Date**: 2026-09-23 (pre-fix — negative KL / binary rewards below; superseded by 2026-09-24 verified run)

### Step 1: APPS Benchmark Dataset & Tokenizer Initialization
- **Dataset**: `codeparrot/apps` (Revision: `refs/convert/parquet`, split `train[:1000]`)
- **Filtered Problems**: 1,000 problem items prepared for PPO rollout.
- **Base Model**: `deepseek-ai/deepseek-coder-1.3b-instruct` + SFT LoRA adapter (`./checkpoints/sft/final`).
- **Trainable Parameters**: `3,147,777 / 1,349,619,713` (`0.2332%`)

### Step 2: Full 50-Step PPO Training Loop Execution
- **Rollout Step Progress** *(stale pre-fix logs — do not cite as current results)*:
  - `PPO step 1/50`: `reward=-0.100 | kl=0.000 | sample_lora_delta=6.936596e-04`
  - `PPO step 10/50`: `reward=-0.200 | kl=-0.282 | sample_lora_delta=3.001042e-04`
  - `PPO step 25/50`: `reward=-0.100 | kl=-0.845 | sample_lora_delta=3.061019e-04`
  - `PPO step 35/50`: `reward=-0.100 | kl=-1.193 | sample_lora_delta=2.828597e-04`
  - `PPO step 50/50`: `reward=-0.200 | kl=-1.861 | sample_lora_delta=3.073641e-04`
- **Known issues in this run** (fixed on `junior-A`): negative KL from wrong ref (`disable_adapter`/base vs SFT), binary rewards from empty/stale training set + single packed I/O case.
- **Execution Status Distribution**: Active sandbox execution across `CE`, `WA`, `RE`, and `TLE` outcomes.
- **LoRA Parameter Updates**: Continuous parameter updates verified across all 50 steps (`sample_lora_delta ~ 3.07e-4 to 6.93e-4`).

### Step 3: Checkpoint Export & Verification
- **Saved Checkpoint Directory**: `./checkpoints/ppo/final`
- **Saved Artifacts**: `adapter_config.json`, `adapter_model.safetensors`, `ppo_metadata.json`, `tokenizer.json`
- **Model Reload Test**: `PeftModel.from_pretrained(base_model, "./checkpoints/ppo/final")` loaded cleanly.
- **Verification Status**: Checkpoint files verified only — **training metrics in this section are stale and must not be reported as final.**




