#!/usr/bin/env python3
"""
modal_train.py
--------------
Complete Modal app that runs the full SFT -> DPO pipeline for
Qwen/Qwen3.5-0.8B on Modal.com (B200 GPU).

Usage:
    # 1. Make sure test_set.json exists locally
    # 2. Run:
    modal run modal_train.py
"""

import json
import os
import sys

import modal

# ---------------------------------------------------------------------------
# MODAL APP SETUP
# ---------------------------------------------------------------------------
APP_NAME = "qwen35-finetune"
VOLUME_NAME = "qwen-adapters"
ADAPTERS_DIR = "/adapters"

# Modal volume (persists across runs)
qwen_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.51.0",
        "trl>=0.15.0",
        "peft>=0.15.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "bert-score>=0.3.13",
        "sacrebleu>=2.4.0",
        "huggingface-hub>=0.25.0",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
MAX_SEQ_LENGTH = 512
SFT_DATASET = "yahma/alpaca-cleaned"
SFT_MAX_SAMPLES = 5_000
DPO_DATASET = "trl-lib/ultrafeedback_binarized"
DPO_MAX_SAMPLES = 2_000

# SFT trial configs: (trial_id, lora_r, lora_target_modules, lr, batch_size, epochs, seed)
SFT_TRIALS = [
    (1, 8,  ["q_proj", "v_proj"],                          2e-4, 4, 1, 42),
    (2, 16, ["q_proj", "v_proj"],                          2e-4, 4, 1, 43),
    (3, 8,  ["q_proj", "k_proj", "v_proj", "o_proj"],     1e-4, 8, 1, 44),
    (4, 32, ["q_proj", "v_proj"],                          5e-5, 2, 3, 45),  # 3 epochs
    (5, 16, ["q_proj", "k_proj", "v_proj", "o_proj"],     1e-4, 4, 2, 46),  # 2 epochs
]

# DPO trial configs: (trial_id, beta, lr, batch_size, epochs, seed)
DPO_TRIALS = [
    (1, 0.1, 5e-6, 4, 1, 42),
    (2, 0.5, 5e-6, 4, 1, 43),
    (3, 0.1, 1e-5, 4, 2, 44),  # 2 epochs
    (4, 0.5, 1e-5, 2, 1, 45),
    (5, 0.2, 5e-6, 8, 1, 46),
]


# ---------------------------------------------------------------------------
# LOCAL HELPER: pretty-print tables
# ---------------------------------------------------------------------------
def print_table(title, rows, headers):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    fmt = "  {:>6}  {:>8}  {:>10}  {:>12}  {:>12}"
    print(fmt.format(*headers))
    print("  " + "-" * 60)
    for r in rows:
        print(fmt.format(*r))
    print("=" * 80)


# ---------------------------------------------------------------------------
# REMOTE FUNCTION: Run all SFT trials
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="B200",
    timeout=10_800,          # 3 hours
    volumes={ADAPTERS_DIR: qwen_volume},
)
def run_sft_trials(test_set: list) -> list:
    """
    Runs 5 SFT trials, evaluates each on the test set, saves adapters.
    Returns a list of result dicts.
    """
    import torch
    import gc
    import os
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,   # CHANGED: import for dynamic padding
    )
    from trl import SFTTrainer, SFTConfig
    from sacrebleu import sentence_bleu
    from bert_score import BERTScorer
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SFT] Using device: {device}")

    # Synchronize Modal volume to capture any updates
    print("[SFT] Reloading volume...")
    qwen_volume.reload()

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    # 1. Load tokenizer once
    print(f"[SFT] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize BERTScorer once to cache DistilBERT on GPU and avoid memory leaks
    print("[SFT] Initializing BERTScorer (distilbert-base-uncased) ...")
    scorer = BERTScorer(
        lang="en",
        model_type="distilbert-base-uncased",
        device=device,
    )

    # 2. Cache base model weights to disk
    cached_model_dir = os.path.join(ADAPTERS_DIR, "base_model_cache")
    if not os.path.isdir(cached_model_dir):
        print(f"[SFT] Downloading and caching base model to {cached_model_dir} ...")
        _cache_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # CHANGED: use Flash Attention 2
            trust_remote_code=True,
        )
        _cache_model.save_pretrained(cached_model_dir)
        tokenizer.save_pretrained(cached_model_dir)
        del _cache_model
        cleanup()
        qwen_volume.commit()
        print(f"[SFT] Base model cached.")
    else:
        print(f"[SFT] Base model already cached at {cached_model_dir}.")

    def load_fresh_base():
        m = AutoModelForCausalLM.from_pretrained(
            cached_model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # CHANGED: use Flash Attention 2
            trust_remote_code=True,
        )
        m.to(device)
        return m

    # 3. Load & format SFT dataset
    print(f"[SFT] Loading dataset: {SFT_DATASET} (max {SFT_MAX_SAMPLES} samples)")
    raw_ds = load_dataset(SFT_DATASET, split="train")
    if len(raw_ds) > SFT_MAX_SAMPLES:
        raw_ds = raw_ds.shuffle(seed=42).select(range(SFT_MAX_SAMPLES))

    def format_alpaca(example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        if input_text and str(input_text).strip():
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
        text = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        return {"text": text}

    formatted_ds = raw_ds.map(format_alpaca, remove_columns=raw_ds.column_names)

    # CHANGED: removed padding="max_length" — the DataCollatorForSeq2Seq below
    # will pad each batch dynamically to its longest sequence instead, cutting
    # wasted compute on short Alpaca examples significantly.
    def tokenize(example):
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            # padding="max_length" removed
        )
        out["labels"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
            for labels_seq in out["input_ids"]
        ]
        return out

    tokenized_ds = formatted_ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    split = tokenized_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"[SFT] Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

    # 4. Evaluation Helper
    def evaluate_model(model, label: str):
        model.eval()
        predictions = []
        references = []

        # Find the ID for the ChatML end token to prevent run-on generations
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            eos_token_ids = [tokenizer.eos_token_id, im_end_id]
        else:
            eos_token_ids = tokenizer.eos_token_id

        for item in test_set:
            prompt = item["prompt"]
            gold = item["gold_answer"]
            prompt_text = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )

            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            predictions.append(generated)
            references.append(gold)

        bleu_scores = [
            sentence_bleu(hyp, [ref]).score / 100.0
            for hyp, ref in zip(predictions, references)
        ]
        avg_bleu = float(np.mean(bleu_scores))

        # Reusing the cached scorer rather than reloading DistilBERT weights onto the GPU
        _, _, f1 = scorer.score(predictions, references)
        avg_bert = float(f1.mean())

        model.train()
        return {"bleu": avg_bleu, "bertscore": avg_bert, "predictions": predictions}

    # 5. Evaluate BASE model
    print("[SFT] Evaluating BASE model on test set...")
    base_model_for_eval = load_fresh_base()
    base_model_for_eval.eval()
    base_eval = evaluate_model(base_model_for_eval, "base")
    print(f"      BASE → BLEU: {base_eval['bleu']:.4f} | BERTScore: {base_eval['bertscore']:.4f}")
    del base_model_for_eval
    cleanup()

    # 6. Run SFT trials
    results = []

    for trial_id, lora_r, target_modules, lr, batch_size, epochs, seed in SFT_TRIALS:
        print(f"\n[SFT] ===== Trial {trial_id} =====")
        print(f"      LoRA r={lora_r}, target={target_modules}, lr={lr}, bs={batch_size}, epochs={epochs}, seed={seed}")

        trial_model = load_fresh_base()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        trial_model = get_peft_model(trial_model, lora_config)
        trial_model.print_trainable_parameters()

        output_dir = os.path.join(ADAPTERS_DIR, f"sft_trial_{trial_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Cap physical batch size on GPU at 4 to prevent CUDA OOM
        # Mathematically preserve original effective batch size target using gradient accumulation
        max_physical_batch_size = 4
        micro_batch_size = min(batch_size, max_physical_batch_size)
        accum_steps = (batch_size * max(1, 8 // batch_size)) // micro_batch_size

        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=accum_steps,
            learning_rate=lr,
            num_train_epochs=epochs,
            max_steps=-1,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            seed=seed,
            report_to="none",
            bf16=True,
            remove_unused_columns=False,
            max_length=MAX_SEQ_LENGTH,
        )

        # CHANGED: DataCollatorForSeq2Seq pads each batch to the longest sequence
        # in that batch only, rather than always padding to MAX_SEQ_LENGTH=512.
        # pad_to_multiple_of=8 keeps tensor shapes aligned for efficient GPU kernels.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=trial_model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        trainer = SFTTrainer(
            model=trial_model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            data_collator=data_collator,   # CHANGED: pass dynamic padding collator
        )

        trainer.train()

        eval_losses = [
            e["eval_loss"]
            for e in trainer.state.log_history
            if "eval_loss" in e
        ]
        final_val_loss = eval_losses[-1] if eval_losses else 999.0

        adapter_path = os.path.join(ADAPTERS_DIR, f"sft_trial_{trial_id}_adapter")
        os.makedirs(adapter_path, exist_ok=True)
        trial_model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"      Adapter saved to: {adapter_path}")

        eval_res = evaluate_model(trial_model, f"sft_trial_{trial_id}")
        print(
            f"      Trial {trial_id} → "
            f"BLEU: {eval_res['bleu']:.4f} | "
            f"BERTScore: {eval_res['bertscore']:.4f} | "
            f"ValLoss: {final_val_loss:.4f}"
        )

        results.append({
            "trial_id": trial_id,
            "stage": "SFT",
            "lora_r": lora_r,
            "target_modules": target_modules,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "bleu": eval_res["bleu"],
            "bertscore": eval_res["bertscore"],
            "val_loss": final_val_loss,
            "adapter_path": adapter_path,
            "predictions": eval_res["predictions"],
        })

        del trial_model, trainer
        cleanup()

    # Free scorer and clean up memory final time
    del scorer
    cleanup()

    # Save results to Volume
    results_path = os.path.join(ADAPTERS_DIR, "sft_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    qwen_volume.commit()

    print("\n[SFT] All SFT trials complete. Results saved.")
    return results


# ---------------------------------------------------------------------------
# REMOTE FUNCTION: Run all DPO trials
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="B200",
    timeout=10_800,          # 3 hours
    volumes={ADAPTERS_DIR: qwen_volume},
)
def run_dpo_trials(best_sft_adapter_name: str, test_set: list) -> list:
    """
    Loads the best SFT adapter, merges it, runs 5 DPO trials, and evaluates.
    """
    import torch
    import gc
    import os
    from datasets import load_dataset
    from peft import PeftModel, LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from sacrebleu import sentence_bleu
    from bert_score import BERTScorer
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DPO] Using device: {device}")

    # Synchronize Modal volume to capture any updates
    print("[DPO] Reloading volume...")
    qwen_volume.reload()

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[DPO] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize BERTScorer once to cache DistilBERT on GPU and avoid memory leaks
    print("[DPO] Initializing BERTScorer (distilbert-base-uncased) ...")
    scorer = BERTScorer(
        lang="en",
        model_type="distilbert-base-uncased",
        device=device,
    )

    cached_model_dir = os.path.join(ADAPTERS_DIR, "base_model_cache")
    best_adapter_path = os.path.join(ADAPTERS_DIR, best_sft_adapter_name)

    # Automatically check and download base model cache if missing
    if not os.path.isdir(cached_model_dir):
        print(f"[DPO] Base model cache not found at {cached_model_dir}. Downloading and caching now...")
        _cache_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # CHANGED: use Flash Attention 2
            trust_remote_code=True,
        )
        _cache_model.save_pretrained(cached_model_dir)
        tokenizer.save_pretrained(cached_model_dir)
        del _cache_model
        cleanup()
        qwen_volume.commit()
        print("[DPO] Base model cached successfully.")
    else:
        print(f"[DPO] Base model already cached at {cached_model_dir}.")

    # Safety Check: Verify SFT adapter path exists in the persistent volume
    if not os.path.isdir(best_adapter_path):
        raise FileNotFoundError(
            f"The SFT adapter directory was not found at '{best_adapter_path}' inside the Modal volume. "
            "If you recently cleared the volume or are running on a fresh setup, please delete your local "
            "'sft_results.json' file and re-run the pipeline to recreate the SFT adapters."
        )

    def load_sft_model():
        """
        Loads base model and merges the best SFT adapter.
        This provides a mathematically clean foundation to start DPO and allows
        the DPOTrainer's implicit reference model functionality to work properly.
        """
        base = AutoModelForCausalLM.from_pretrained(
            cached_model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # CHANGED: use Flash Attention 2
            trust_remote_code=True,
        )
        base.to(device)
        # Load and merge SFT weights permanently into model layers
        model = PeftModel.from_pretrained(base, best_adapter_path)
        merged_model = model.merge_and_unload()
        return merged_model

    # 3. Load & format DPO dataset
    print(f"[DPO] Loading dataset: {DPO_DATASET} (max {DPO_MAX_SAMPLES} samples)")
    raw_ds = load_dataset(DPO_DATASET, split="train")
    if len(raw_ds) > DPO_MAX_SAMPLES:
        raw_ds = raw_ds.shuffle(seed=42).select(range(DPO_MAX_SAMPLES))

    def format_dpo(example):
        # BUGFIX: Handle conversational preference structure (no top-level prompt column)
        chosen = example["chosen"]
        rejected = example["rejected"]

        # Extract user prompt from the first dialogue turn
        if isinstance(chosen, list) and len(chosen) > 0 and isinstance(chosen[0], dict):
            prompt = chosen[0]["content"]
        else:
            prompt = ""

        # Extract assistant responses (last element)
        if isinstance(chosen, list) and len(chosen) > 1 and isinstance(chosen[-1], dict):
            chosen_text = chosen[-1]["content"]
        else:
            chosen_text = str(chosen)

        if isinstance(rejected, list) and len(rejected) > 1 and isinstance(rejected[-1], dict):
            rejected_text = rejected[-1]["content"]
        else:
            rejected_text = str(rejected)

        prompt_text = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # BUGFIX: Append <|im_end|> so model learns when to stop generating
        return {
            "prompt": prompt_text,
            "chosen": chosen_text + "<|im_end|>",
            "rejected": rejected_text + "<|im_end|>",
        }

    formatted_ds = raw_ds.map(format_dpo, remove_columns=raw_ds.column_names)

    split = formatted_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"[DPO] Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

    # 4. Evaluation Helper
    def evaluate_model(model, label: str):
        model.eval()
        predictions = []
        references = []

        # Find the ID for the ChatML end token to prevent run-on generations
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
            eos_token_ids = [tokenizer.eos_token_id, im_end_id]
        else:
            eos_token_ids = tokenizer.eos_token_id

        for item in test_set:
            prompt = item["prompt"]
            gold = item["gold_answer"]
            prompt_text = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )

            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            predictions.append(generated)
            references.append(gold)

        bleu_scores = [
            sentence_bleu(hyp, [ref]).score / 100.0
            for hyp, ref in zip(predictions, references)
        ]
        avg_bleu = float(np.mean(bleu_scores))

        # Reusing the cached scorer
        _, _, f1 = scorer.score(predictions, references)
        avg_bert = float(f1.mean())

        model.train()
        return {"bleu": avg_bleu, "bertscore": avg_bert, "predictions": predictions}

    # 5. Run DPO trials
    results = []

    for trial_id, beta, lr, batch_size, epochs, seed in DPO_TRIALS:
        print(f"\n[DPO] ===== Trial {trial_id} =====")
        print(f"      beta={beta}, lr={lr}, bs={batch_size}, epochs={epochs}, seed={seed}")

        # Loads base with merged SFT weights
        trial_model = load_sft_model()

        # Config for new DPO LoRA adapter on top of SFT-merged base
        dpo_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        trial_model = get_peft_model(trial_model, dpo_lora_config)
        trial_model.print_trainable_parameters()

        output_dir = os.path.join(ADAPTERS_DIR, f"dpo_trial_{trial_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Cap physical batch size on GPU at 4 to prevent CUDA OOM
        max_physical_batch_size = 4
        micro_batch_size = min(batch_size, max_physical_batch_size)
        accum_steps = (batch_size * max(1, 8 // batch_size)) // micro_batch_size

        training_args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=accum_steps,
            learning_rate=lr,
            num_train_epochs=epochs,
            max_steps=-1,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            seed=seed,
            report_to="none",
            bf16=True,
            remove_unused_columns=False,
            beta=beta,
            max_length=MAX_SEQ_LENGTH,
        )

        # ref_model=None will safely use the frozen model (adapter inactive)
        # as reference. Since the model has SFT merged, the SFT weights act as the reference.
        trainer = DPOTrainer(
            model=trial_model,
            ref_model=None,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

        trainer.train()

        eval_losses = [
            e["eval_loss"]
            for e in trainer.state.log_history
            if "eval_loss" in e
        ]
        final_val_loss = eval_losses[-1] if eval_losses else 999.0

        adapter_path = os.path.join(ADAPTERS_DIR, f"dpo_trial_{trial_id}_adapter")
        os.makedirs(adapter_path, exist_ok=True)
        trial_model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"      Adapter saved to: {adapter_path}")

        eval_res = evaluate_model(trial_model, f"dpo_trial_{trial_id}")
        print(
            f"      Trial {trial_id} → "
            f"BLEU: {eval_res['bleu']:.4f} | "
            f"BERTScore: {eval_res['bertscore']:.4f} | "
            f"ValLoss: {final_val_loss:.4f}"
        )

        results.append({
            "trial_id": trial_id,
            "stage": "DPO",
            "beta": beta,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "bleu": eval_res["bleu"],
            "bertscore": eval_res["bertscore"],
            "val_loss": final_val_loss,
            "adapter_path": adapter_path,
            "predictions": eval_res["predictions"],
        })

        del trial_model, trainer
        cleanup()

    # Free scorer and clean up memory final time
    del scorer
    cleanup()

    # Save results to Volume
    results_path = os.path.join(ADAPTERS_DIR, "dpo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    qwen_volume.commit()

    print("\n[DPO] All DPO trials complete. Results saved.")
    return results


# ---------------------------------------------------------------------------
# LOCAL ENTRYPOINT
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  Qwen3.5-0.8B SFT -> DPO Pipeline on Modal.com")
    print("=" * 80)

    # 1. Load test set locally
    if not os.path.exists("test_set.json"):
        print("\nERROR: test_set.json not found!")
        print("Run first:  python generate_test_set.py")
        sys.exit(1)

    with open("test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"\nLoaded test set with {len(test_set)} prompts from test_set.json")

    # 2. Run SFT trials on Modal (Skip if sft_results.json exists)
    print("\n" + "-" * 80)
    print("  STAGE 1: SUPERVISED FINE-TUNING (5 trials)")
    print("-" * 80)

    sft_results_path = "sft_results.json"
    if os.path.exists(sft_results_path):
        print(f"\n  [SFT] Local file '{sft_results_path}' already exists.")
        print("        Loading cached SFT metrics and skipping SFT execution on Modal.")
        with open(sft_results_path, "r", encoding="utf-8") as f:
            sft_results = json.load(f)
    else:
        print("  [SFT] No cached results found. Running SFT trials on Modal...")
        sft_results = run_sft_trials.remote(test_set)
        # Save SFT results locally
        with open(sft_results_path, "w", encoding="utf-8") as f:
            json.dump(sft_results, f, indent=2)
        print(f"\n  Saved {sft_results_path} locally.")

    # Print SFT table
    print_table(
        "SFT TRIAL RESULTS",
        [
            (
                r["trial_id"],
                r["lora_r"],
                f"{r['lr']:.0e}",
                f"{r['bleu']:.4f}",
                f"{r['bertscore']:.4f}",
            )
            for r in sft_results
        ],
        ["Trial", "LoRA_r", "LR", "BLEU", "BERTScore"],
    )

    # Pick best SFT: maximize BLEU + BERTScore, tie-break with lower val_loss
    def sft_score(r):
        return (r["bleu"] + r["bertscore"], -r["val_loss"])

    best_sft = max(sft_results, key=sft_score)
    print(f"\n  >>> BEST SFT TRIAL: {best_sft['trial_id']} <<<")
    print(f"      LoRA r={best_sft['lora_r']}, LR={best_sft['lr']}, BS={best_sft['batch_size']}, Epochs={best_sft['epochs']}")
    print(f"      BLEU={best_sft['bleu']:.4f} | BERTScore={best_sft['bertscore']:.4f} | ValLoss={best_sft['val_loss']:.4f}")

    # Explain selection decision
    sorted_sft = sorted(sft_results, key=sft_score, reverse=True)
    if len(sorted_sft) >= 2:
        delta = (sorted_sft[0]["bleu"] + sorted_sft[0]["bertscore"]) - \
                (sorted_sft[1]["bleu"] + sorted_sft[1]["bertscore"])
        if delta < 0.005:
            print(
                f"      Note: Top-2 trials are within 0.005 combined score; "
                f"val_loss tiebreaker applied (Trial {sorted_sft[0]['trial_id']} "
                f"loss={sorted_sft[0]['val_loss']:.4f} vs "
                f"Trial {sorted_sft[1]['trial_id']} loss={sorted_sft[1]['val_loss']:.4f})"
            )

    adapter_name = os.path.basename(best_sft["adapter_path"])

    # 3. Run DPO trials on Modal
    print("\n" + "-" * 80)
    print("  STAGE 2: DIRECT PREFERENCE OPTIMIZATION (5 trials)")
    print("-" * 80)
    dpo_results = run_dpo_trials.remote(adapter_name, test_set)

    # Print DPO table
    print_table(
        "DPO TRIAL RESULTS",
        [
            (
                r["trial_id"],
                r["beta"],
                f"{r['lr']:.0e}",
                f"{r['bleu']:.4f}",
                f"{r['bertscore']:.4f}",
            )
            for r in dpo_results
        ],
        ["Trial", "Beta", "LR", "BLEU", "BERTScore"],
    )

    def dpo_score(r):
        return (r["bleu"] + r["bertscore"], -r["val_loss"])

    best_dpo = max(dpo_results, key=dpo_score)
    print(f"\n  >>> BEST DPO TRIAL: {best_dpo['trial_id']} <<<")
    print(f"      Beta={best_dpo['beta']}, LR={best_dpo['lr']}, BS={best_dpo['batch_size']}, Epochs={best_dpo['epochs']}")
    print(f"      BLEU={best_dpo['bleu']:.4f} | BERTScore={best_dpo['bertscore']:.4f} | ValLoss={best_dpo['val_loss']:.4f}")

    # Explain DPO selection decision
    sorted_dpo = sorted(dpo_results, key=dpo_score, reverse=True)
    if len(sorted_dpo) >= 2:
        delta = (sorted_dpo[0]["bleu"] + sorted_dpo[0]["bertscore"]) - \
                (sorted_dpo[1]["bleu"] + sorted_dpo[1]["bertscore"])
        if delta < 0.005:
            print(
                f"      Note: Top-2 trials are within 0.005 combined score; "
                f"val_loss tiebreaker applied (Trial {sorted_dpo[0]['trial_id']} "
                f"loss={sorted_dpo[0]['val_loss']:.4f} vs "
                f"Trial {sorted_dpo[1]['trial_id']} loss={sorted_dpo[1]['val_loss']:.4f})"
            )

    # Save DPO results locally
    with open("dpo_results.json", "w") as f:
        json.dump(dpo_results, f, indent=2)
    print("\n  Saved dpo_results.json locally")

    # 4. Final summary
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print("  Adapters saved to Modal volume: qwen-adapters")
    print("  To download adapters for local evaluation, run:")
    print("    modal volume get qwen-adapters / local_adapters/")
    print("\n  Next step: python evaluate.py")
    print("=" * 80)