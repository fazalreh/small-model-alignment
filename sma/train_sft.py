"""From-scratch supervised fine-tuning loop (no HF Trainer).

    python -m sma.train_sft --config configs/sft.yaml

Features: chat-template prompt formatting, response-only loss masking,
right padding with attention masks, gradient accumulation with token-mean
normalization over the *whole* accumulation window, mixed precision
(bf16 autocast on H100, or fp16 autocast + GradScaler on T4) with FP32
master weights, gradient clipping, periodic validation loss, best + final
checkpoints, and a ``metrics.jsonl`` written after every log/eval step.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sma import core
from sma.prompts import format_prompt
from sma.utils import (
    Timer,
    append_jsonl,
    autocast_ctx,
    env_report,
    load_config,
    param_groups_for_weight_decay,
    read_jsonl,
    resolve_precision,
    set_seed,
    warmup_cosine_lambda,
    write_json,
)

DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "train_path": "outputs/data/sft_train.jsonl",
    "val_path": "outputs/data/sft_val.jsonl",
    "output_dir": "outputs/sft",
    "max_length": 512,
    "max_train_examples": 30000,
    "max_val_examples": 1000,
    "per_device_batch_size": 16,
    "eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2.0e-5,
    "warmup_ratio": 0.03,
    "lr_schedule": "cosine",
    "min_lr_ratio": 0.0,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1.0e-8,
    "max_grad_norm": 1.0,
    "num_epochs": 1,
    "precision": "auto",  # auto -> bf16 on H100/A100, fp16 (+GradScaler) on T4
    "gradient_checkpointing": False,
    "eval_every": 100,
    "log_every": 10,
    "seed": 42,
    "attn_implementation": "sdpa",
}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return format_prompt(row["prompt"]), row["response"]


class SFTCollator:
    """Picklable collate_fn (DataLoader workers may be spawned, e.g. on Windows)."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items):
        prompts = [p for p, _ in items]
        responses = [r for _, r in items]
        return core.build_lm_batch(self.tokenizer, prompts, responses, self.max_length)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    total_nll, total_tokens, total_seq_nll, n_rows = 0.0, 0, 0.0, 0
    for batch in loader:
        with autocast_ctx(device, amp_dtype):
            logps, counts = core.forward_logprobs(model, batch)
        total_nll += -logps.sum().item()
        total_tokens += counts.sum().item()
        total_seq_nll += -logps.sum().item()
        n_rows += logps.shape[0]
    model.train()
    token_loss = total_nll / max(1, total_tokens)
    return {
        "val_loss": token_loss,
        "val_ppl": math.exp(min(token_loss, 50)),
        "val_seq_nll": total_seq_nll / max(1, n_rows),
        "val_tokens": total_tokens,
    }


def save_checkpoint(model, tokenizer, path: Path, info: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(info, path / "checkpoint_info.json")


def main(argv=None):
    cfg = load_config(argv, DEFAULTS, "From-scratch SFT")
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler, precision = resolve_precision(cfg.precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # FP32 master weights; autocast handles the low-precision compute.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32, attn_implementation=cfg.attn_implementation
    ).to(device)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    train_rows = read_jsonl(cfg.train_path)[: cfg.max_train_examples]
    val_rows = read_jsonl(cfg.val_path)[: cfg.max_val_examples]
    collate = SFTCollator(tokenizer, cfg.max_length)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        PairDataset(train_rows), batch_size=cfg.per_device_batch_size, shuffle=True,
        collate_fn=collate, generator=generator, num_workers=2, drop_last=False,
    )
    val_loader = DataLoader(
        PairDataset(val_rows), batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=2
    )

    micro_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_per_epoch / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(round(cfg.warmup_ratio * total_steps))

    optimizer = torch.optim.AdamW(
        param_groups_for_weight_decay(model, cfg.weight_decay),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        fused=torch.cuda.is_available(),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(total_steps, warmup_steps, cfg.lr_schedule, cfg.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    run_info = {
        "config": vars(cfg),
        "precision": precision,
        "effective_batch_size": cfg.per_device_batch_size * cfg.gradient_accumulation_steps,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "loss_reduction": "token-mean over all response tokens in the accumulation window",
        "env": env_report(),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    write_json(run_info, out_dir / "run_config.json")
    print(json.dumps({k: v for k, v in run_info.items() if k != "config"}, indent=2))

    timer = Timer()
    best_val = float("inf")
    step = 0
    window_loss, window_tokens = 0.0, 0

    initial = evaluate(model, val_loader, device, amp_dtype)
    append_jsonl({"type": "eval", "step": 0, "epoch": 0.0, **initial}, metrics_path)
    print(f"[sft] step 0 val_loss={initial['val_loss']:.4f}")

    for epoch in range(cfg.num_epochs):
        micro_batches = iter(train_loader)
        for _ in range(steps_per_epoch):
            window = []
            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    window.append(next(micro_batches))
                except StopIteration:
                    break
            if not window:
                break
            # Normalize by the token count of the whole window -> exact token-mean gradient.
            window_token_count = sum(int(b["labels"][:, 1:].ne(core.IGNORE_INDEX).sum()) for b in window)
            for batch in window:
                with autocast_ctx(device, amp_dtype):
                    loss, n_tok = core.sft_loss_from_batch(model, batch, token_normalizer=max(1, window_token_count))
                scaler.scale(loss).backward()
                window_loss += loss.item() * max(1, window_token_count)
                window_tokens += int(n_tok)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                train_loss = window_loss / max(1, window_tokens)
                rec = {
                    "type": "train",
                    "step": step,
                    "epoch": step / max(1, steps_per_epoch),
                    "train_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "tokens": window_tokens,
                    "elapsed_s": timer.elapsed(),
                }
                if use_scaler:
                    rec["grad_scale"] = scaler.get_scale()
                append_jsonl(rec, metrics_path)
                print(f"[sft] step {step}/{total_steps} loss={train_loss:.4f} lr={rec['lr']:.2e} gn={rec['grad_norm']:.2f}")
                window_loss, window_tokens = 0.0, 0

            if step % cfg.eval_every == 0 or step == total_steps:
                metrics = evaluate(model, val_loader, device, amp_dtype)
                append_jsonl({"type": "eval", "step": step, "elapsed_s": timer.elapsed(), **metrics}, metrics_path)
                print(f"[sft] step {step} val_loss={metrics['val_loss']:.4f}")
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    save_checkpoint(model, tokenizer, out_dir / "best", {"step": step, **metrics})
                    print(f"[sft] new best checkpoint at step {step}")

    save_checkpoint(model, tokenizer, out_dir / "final", {"step": step})
    summary = {"best_val_loss": best_val, "steps": step, "elapsed_s": timer.elapsed()}
    write_json(summary, out_dir / "summary.json")
    print(f"[sft] done: {summary}")


if __name__ == "__main__":
    main(sys.argv[1:])
