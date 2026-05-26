# Qwen3.5-0.8B Instruction Fine-Tuning Pipeline

### NLP with Deep Learning — Assignment 04 | Track 1, Option A (SFT → DPO)

### Institute of Business Administration

---

## Overview

This repository implements a complete **Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO)** pipeline on **Qwen/Qwen3.5-0.8B-Base**, a 753M-parameter base language model. The goal is to transform a raw base model into an instruction-following model that produces human-preferred responses, using parameter-efficient LoRA adapters throughout.

The pipeline runs entirely on **Modal.com** (serverless GPU cloud) using an **NVIDIA B200** GPU. Test set generation uses the **DeepSeek API**.

---

## Results Summary

| Stage              |    BLEU    | BERTScore  | Δ BLEU  | Δ BERTScore |
| :----------------- | :--------: | :--------: | :-----: | :---------: |
| Base Model         |   0.1226   |   0.8025   |    —    |      —      |
| Best SFT (Trial 4) |   0.1256   |   0.8144   | +0.0030 |   +0.0119   |
| Best DPO (Trial 3) | **0.1337** | **0.8230** | +0.0111 |   +0.0205   |

**Best SFT:** Trial 4 — LoRA r=32, target=`q_proj,v_proj`, LR=5e-5, BS=2, 3 epochs  
**Best DPO:** Trial 3 — β=0.1, LR=1e-5, BS=4, 2 epochs

---

## Repository Structure

```
├── generate_test_set.py   # Generates 10 prompts + gold answers via DeepSeek API
├── modal_train.py         # Full Modal pipeline: 5 SFT trials → best SFT → 5 DPO trials
├── test_set.json          # 10 evaluation prompts with gold answers (pre-generated)
├── sft_results.json       # SFT trial scores, configs, and model predictions
├── dpo_results.json       # DPO trial scores, configs, and model predictions
└── README.md
```

> **Note:** Adapter weights are stored in a Modal persistent volume (`qwen-adapters`) and are not included in this repo due to size (~500MB each).

---

## Experimental Setup

### Model

- **Base model:** `Qwen/Qwen3.5-0.8B-Base`
- **Precision:** bfloat16
- **Attention:** SDPA (Scaled Dot-Product Attention)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) via PEFT

### Platform

- **GPU:** NVIDIA B200 (Modal.com serverless)
- **Container:** Debian Slim, Python 3.11
- **Key libraries:** `transformers>=4.51.0`, `trl>=0.15.0`, `peft>=0.15.0`

### Datasets

| Stage | Dataset                           | Subset Used | Train | Val |
| :---- | :-------------------------------- | :---------: | :---: | :-: |
| SFT   | `yahma/alpaca-cleaned`            |    5,000    | 4,500 | 500 |
| DPO   | `trl-lib/ultrafeedback_binarized` |    2,000    | 1,800 | 200 |

### Evaluation

- **BLEU** (sentence-level via `sacrebleu`) — n-gram overlap with gold answer
- **BERTScore F1** (`distilbert-base-uncased`) — semantic similarity via embeddings
- **Validation loss** — tie-breaker when BLEU + BERTScore difference < 0.005
- Evaluated on 10 hand-crafted prompts covering: coding, math, science, history, creative writing, general knowledge

---

## SFT Trial Configurations

|  Trial  | LoRA r | Target Modules |    LR    | Batch Size | Epochs |    BLEU    | BERTScore  | Val Loss |
| :-----: | :----: | :------------- | :------: | :--------: | :----: | :--------: | :--------: | :------: |
|    1    |   8    | q, v           |   2e-4   |     4      |   1    |   0.1180   |   0.8146   |  1.3594  |
|    2    |   16   | q, v           |   2e-4   |     4      |   1    |   0.1230   |   0.8127   |  1.3558  |
|    3    |   8    | q, k, v, o     |   1e-4   |     8      |   1    |   0.1182   |   0.8100   |  1.3582  |
| **4** ✓ | **32** | **q, v**       | **5e-5** |   **2**    | **3**  | **0.1256** | **0.8144** |  1.3790  |
|    5    |   16   | q, k, v, o     |   1e-4   |     4      |   2    |   0.1138   |   0.8169   |  1.3486  |

**✓ Trial 4 selected** — highest combined score (0.9400). Top-2 margin was 0.0043 < 0.005, so val loss tiebreaker was noted; Trial 4 still leads on the primary metric.

---

## DPO Trial Configurations

All DPO trials start from the **best SFT adapter (Trial 4) merged** into the base model. A new LoRA adapter (r=16, α=32, target=`q_proj,v_proj`) is applied on top for each trial.

|  Trial  |    β    |    LR    | Batch Size | Epochs |    BLEU    | BERTScore  |  Val Loss  |
| :-----: | :-----: | :------: | :--------: | :----: | :--------: | :--------: | :--------: |
|    1    |   0.1   |   5e-6   |     4      |   1    |   0.1211   |   0.8142   |   0.6890   |
|    2    |   0.5   |   5e-6   |     4      |   1    |   0.1255   |   0.8175   |   0.6751   |
| **3** ✓ | **0.1** | **1e-5** |   **4**    | **2**  | **0.1337** | **0.8230** | **0.6601** |
|    4    |   0.5   |   1e-5   |     2      |   1    |   0.1163   |   0.8133   |   0.6732   |
|    5    |   0.2   |   5e-6   |     8      |   1    |   0.1201   |   0.8112   |   0.6860   |

**✓ Trial 3 selected** — clearly best on both metrics (combined score 0.9567, margin 0.0137 > 0.005, no tiebreaker needed).

---

## How to Reproduce

### Prerequisites

```bash
pip install modal openai
modal setup    # authenticate your Modal account
```

Set your DeepSeek API key as an environment variable:

```bash
export DEEPSEEK_API_KEY='your_key_here'
```

> ⚠️ **Never hardcode API keys in source files before pushing to GitHub.**

---

### Step 1 — Generate the Test Set

```bash
python generate_test_set.py
```

Calls DeepSeek-v4-flash to generate 10 diverse instruction prompts and gold reference answers. Saves to `test_set.json`.

**Time:** ~1–2 minutes | **Cost:** negligible

---

### Step 2 — Run the Full Pipeline on Modal

```bash
modal run modal_train.py
```

This single command:

1. Uploads `test_set.json` to Modal
2. Spins up a B200 GPU container
3. Runs 5 SFT trials on `yahma/alpaca-cleaned` (5k samples)
4. Evaluates each trial on the 10 test prompts
5. Selects the best SFT adapter automatically
6. Merges the best SFT weights into the base model
7. Runs 5 DPO trials on `trl-lib/ultrafeedback_binarized` (2k pairs)
8. Evaluates each DPO trial
9. Saves all adapters to the Modal volume `qwen-adapters`
10. Prints complete results tables locally and saves `sft_results.json` + `dpo_results.json`

**Total GPU time:** ~130 minutes (~95 min SFT + ~34 min DPO)  
**Your active time:** ~30 seconds (just run the command, it runs in the cloud)

> 💡 If `sft_results.json` already exists locally, the pipeline skips SFT and goes straight to DPO. Delete it to force a full re-run.

---

### Step 3 — Download Adapters (Optional)

```bash
mkdir -p local_adapters
modal volume get qwen-adapters / local_adapters/
```

---

## Key Design Decisions

**Dynamic padding:** `DataCollatorForSeq2Seq` pads each batch to the longest sequence in that batch rather than always padding to `MAX_SEQ_LENGTH=512`, significantly reducing wasted compute on short Alpaca examples.

**Base model caching:** The base model is downloaded once and cached to the Modal volume on first run. All subsequent trials load from the cache, preventing version drift and saving download time.

**DPO reference model:** `ref_model=None` uses TRL's implicit reference (frozen LoRA-disabled model). Since SFT weights are merged into the base before DPO begins, the SFT model correctly acts as the reference distribution.

**Gradient accumulation:** Physical batch size is capped at 4 to prevent CUDA OOM. Gradient accumulation steps are computed automatically to preserve the intended effective batch size.

---

## Known Limitations

- **Test set size:** Only 10 prompts. Scores carry high variance; BERTScore is the more reliable metric given the diverse task types.
- **Model verbosity:** Some model outputs enter repetition loops (e.g., listing the same capital city 8 times) and exhaust the 256-token generation budget before producing a complete response. This is a model behaviour issue, not a token budget issue — the gold answers are all short and well within 256 tokens.
- **Arithmetic errors:** Multiple SFT trials compute the dataset sum as 48 instead of the correct 46 (Prompt 9), indicating pattern-matching rather than actual computation.
- **Historical hallucinations:** Creative writing outputs (Prompt 5) replace the historically accurate Gutenberg context with Thomas Edison references across all trials.

---

## Troubleshooting

**`test_set.json` not found**

```bash
python generate_test_set.py
```

**`TypeError: DPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`**  
Remove `max_prompt_length=MAX_SEQ_LENGTH` from the `DPOConfig` block in `modal_train.py`. This parameter was removed in TRL 0.15+.

**SFT adapter not found during DPO**  
If you cleared the Modal volume, delete `sft_results.json` locally and re-run the full pipeline so SFT adapters are recreated.

**Modal timeout**  
Each stage has a 3-hour timeout. If SFT times out, reduce `SFT_MAX_SAMPLES` in `modal_train.py`. The SFT and DPO stages run as separate Modal functions and can be re-run independently.

---

## References

- Qwen Team. _Qwen Technical Report._ Alibaba Group, 2024.
- Rafailov et al. _Direct Preference Optimization._ NeurIPS 2023.
- Hu et al. _LoRA: Low-Rank Adaptation of Large Language Models._ ICLR 2022.
- von Werra et al. _TRL: Transformer Reinforcement Learning._ HuggingFace, 2020.
- `yahma/alpaca-cleaned` — https://huggingface.co/datasets/yahma/alpaca-cleaned
- `trl-lib/ultrafeedback_binarized` — https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized
