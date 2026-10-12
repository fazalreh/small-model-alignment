"""Task 5 ("Idiot Sandwich"): TRL DPOTrainer + PEFT LoRA persona tuning.

    python -m sma.train_lora_dpo --config configs/lora_dpo.yaml

Model: HuggingFaceTB/SmolLM2-360M-Instruct. Data: the provided
data/persona_lora/ramsay_style_preferences.json, converted to TRL's
conversational format (prompt / chosen / rejected as role-message lists) so
DPOTrainer applies the model's own chat template. With ``peft_config`` and no
``ref_model``, TRL uses the base model with the adapter disabled as the frozen
reference, so no second model copy is needed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

from sma.utils import env_report, load_config, read_json, set_seed, write_json

DEFAULTS = {
    "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "data_path": "data/persona_lora/ramsay_style_preferences.json",
    "output_dir": "outputs/lora_dpo",
    "val_fraction": 0.1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "learning_rate": 5.0e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.0,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "beta": 0.1,
    "max_length": 512,
    "max_prompt_length": 256,
    "max_grad_norm": 1.0,
    "logging_steps": 2,
    "eval_steps": 5,
    "seed": 42,
    "precision": "auto",
}


def to_conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"].strip()}],
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
    }


def main(argv=None):
    cfg = load_config(argv, DEFAULTS, "TRL DPOTrainer + LoRA persona tuning")
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    rows = read_json(cfg.data_path)
    random.Random(cfg.seed).shuffle(rows)
    n_val = max(1, int(round(len(rows) * cfg.val_fraction)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    write_json({"train": train_rows, "val": val_rows}, out_dir / "split.json")
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conversational(r) for r in val_rows])

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and cfg.precision in ("auto", "bf16")
    use_fp16 = torch.cuda.is_available() and not use_bf16 and cfg.precision in ("auto", "fp16")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[m.strip() for m in cfg.target_modules.split(",") if m.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        beta=cfg.beta,
        loss_type="sigmoid",
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainable, total = 0, 0
    for p in trainer.model.parameters():
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0

    initial_eval = trainer.evaluate()
    trainer.train()
    final_eval = trainer.evaluate()

    adapter_dir = out_dir / "final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(trainer.state.log_history, out_dir / "log_history.json")
    import trl, peft

    summary = {
        "config": vars(cfg),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": trainable / max(1, total),
        "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "optimizer_steps": trainer.state.global_step,
        "env": {**env_report(), "trl": trl.__version__, "peft": peft.__version__},
    }
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
