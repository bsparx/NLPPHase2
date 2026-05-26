# Step-by-Step Execution Guide

Follow these exact commands, in order, to run the entire pipeline from a fresh virtual environment.

> **Prerequisites:** Python 3.10+, `modal` CLI installed (`pip install modal` and `modal setup` completed), a Modal.com account with GPU quota, and a stable internet connection.

---

## Step 1: Navigate to the Project Folder

```bash
cd /home/muddasir/Project/NLPProject2
```

---

## Step 2: Create a Python Virtual Environment

```bash
python3 -m venv venv
```

This creates a folder named `venv` inside the project directory. All local Python work will happen inside this environment.

---

## Step 3: Activate the Virtual Environment

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

Your terminal prompt should now show `(venv)` at the beginning, confirming the environment is active.

---

## Step 4: Install Local Dependencies

```bash
pip install --upgrade pip
pip install openai
```

> **Why only `openai`?** The heavy ML libraries (PyTorch, Transformers, TRL, etc.) are installed automatically inside the Modal container. You only need `openai` locally to call the DeepSeek API.

---

## Step 5: Generate the Test Set (Local)

```bash
python generate_test_set.py
```

**What it does:**
- Calls DeepSeek-v4-flash to generate 10 diverse instruction prompts
- Calls DeepSeek-v4-flash again to generate gold (reference) answers for each prompt
- Saves everything to `test_set.json` in the project folder

**Expected time:** ~1–2 minutes  
**Expected output:** A new file `test_set.json` appears in the folder.

---

## Step 6: Verify the Test Set Was Created

```bash
ls -la test_set.json
```

You should see `test_set.json` in the listing. You can peek at its contents:

```bash
head -n 20 test_set.json
```

---

## Step 7: Run the Full Training Pipeline on Modal

```bash
modal run modal_train.py
```

**What it does:**
1. Uploads `test_set.json` to the Modal cloud worker
2. Spins up an **A10G GPU**
3. Downloads `Qwen/Qwen3.5-0.8B-Base` from HuggingFace (into the ephemeral container)
4. Runs **5 SFT trials** on `yahma/alpaca-cleaned` (5,000 samples)
5. Evaluates each SFT trial on your 10 test prompts (**BLEU + BERTScore**)
6. Automatically selects the best SFT trial
7. Runs **5 DPO trials** on `trl-lib/ultrafeedback_binarized` (2,000 pairs) starting from the best SFT adapter
8. Evaluates each DPO trial
9. Saves all LoRA adapters + result JSONs to the Modal volume `qwen-adapters`
10. Prints a complete results table in your local terminal

**Expected time:** ~4 hours of GPU compute  
**Your active time:** ~10 seconds (just the command above)

> 💡 **Pro tip:** This runs entirely on Modal's servers. You can close your laptop after starting it. Check progress anytime with:
> ```bash
> modal logs qwen35-finetune
> ```

---

## Step 8: Verify Local Result Files

After the Modal run completes, two JSON files should appear in your project folder (they are written back by Modal automatically):

```bash
ls -la sft_results.json dpo_results.json
```

If they are missing, you can manually fetch them from the Modal volume:

```bash
modal volume get qwen-adapters /adapters/sft_results.json ./sft_results.json
modal volume get qwen-adapters /adapters/dpo_results.json ./dpo_results.json
```

---

## Step 9: Download Adapters (Optional but Recommended)

If you want the actual LoRA weights for local inspection or to include in your submission:

```bash
mkdir -p local_adapters
modal volume get qwen-adapters / local_adapters/
```

This downloads all 10 adapter folders + result JSONs to `local_adapters/`.

---

## Step 10: Generate the Report Skeleton

```bash
python evaluate.py
```

**What it does:**
- Reads `sft_results.json` and `dpo_results.json`
- Generates `REPORT_SKELETON.md` with all tables, qualitative examples, and analysis sections

**Expected time:** Instant.

---

## Step 11: Convert Report to Word / PDF

**Option A — Pandoc (cleanest):**

```bash
pip install pypandoc
pandoc REPORT_SKELETON.md -o YourName_PartnerName.docx
```

**Option B — Copy-paste:**
Open `REPORT_SKELETON.md` in any text editor, copy everything, and paste into Microsoft Word.

---

## Step 12: Finalize the Report

1. Open `YourName_PartnerName.docx`
2. Add **Figure 1** (a bar chart): Use Excel / Google Sheets with the BLEU + BERTScore numbers from Table 1 and Table 2. Compare Base vs Best SFT vs Best DPO.
3. Fill in the **Base Model** scores in the comparison table (you noted them from the Modal terminal output in Step 7).
4. Add any personal observations to the qualitative examples section.
5. Save as PDF if required by your instructor.

---

## Step 13: Deactivate the Virtual Environment (When Done)

```bash
deactivate
```

This returns you to your system's default Python environment.

---

## Complete Command Cheat Sheet

If you already have the venv set up, here is the entire workflow in one block:

```bash
cd /home/muddasir/Project/NLPProject2
source venv/bin/activate
python generate_test_set.py
modal run modal_train.py
python evaluate.py
pandoc REPORT_SKELETON.md -o YourName_PartnerName.docx
deactivate
```

---

## What Gets Produced

After running the full pipeline, your project folder will contain:

```
NLPProject2/
├── venv/                       # Virtual environment (do NOT submit this)
├── generate_test_set.py
├── modal_train.py
├── evaluate.py
├── test_set.json               # 10 prompts + gold answers
├── sft_results.json            # SFT trial scores + predictions
├── dpo_results.json            # DPO trial scores + predictions
├── REPORT_SKELETON.md          # Full report template
├── YourName_PartnerName.docx   # Final report (you create this)
└── local_adapters/             # (Optional) Downloaded LoRA weights
    ├── sft_trial_1_adapter/
    ├── ...
    └── dpo_trial_5_adapter/
```

> **Do NOT submit the `venv/` folder.** It is huge and unnecessary. Only submit the `.py` files, `.json` files, and the report.

---

## Troubleshooting Quick Fixes

| Symptom | Fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'openai'` | You forgot Step 3 (activate venv) or Step 4 (`pip install openai`). |
| `test_set.json not found` | Run Step 5 (`python generate_test_set.py`) before Step 7. |
| `modal: command not found` | Install Modal CLI globally: `pip install modal` then `modal setup`. |
| Modal job fails with timeout | Each stage (SFT, DPO) has its own 3-hour limit. If one fails, the other still has its own budget. Check logs with `modal logs qwen35-finetune`. |
| `sft_results.json` missing after run | Modal sometimes writes results back asynchronously. Wait 30 seconds, or manually fetch with `modal volume get` (see Step 8). |

---

**Good luck!**
