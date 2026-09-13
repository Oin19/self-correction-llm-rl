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
  - `runtime`: `24m 22s`

### SFT Training Loss Trajectory
| Step | Training Loss |
| :--- | :--- |
| **50** | `1.343351` |
| **100** | `1.268777` |
| **150** | `1.220009` |
| **200** | `1.181134` |
| **250** | **`1.200138`** |

- **Checkpoint Saved**: `./checkpoints/sft/final`
- **Status**: **SFT Training Completed Successfully**
