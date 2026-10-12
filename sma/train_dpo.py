"""From-scratch DPO training loop (no TRL / HF Trainer). Same code for HH and UF.

    python -m sma.train_dpo --config configs/dpo_hh.yaml
    python -m sma.train_dpo --config configs/dpo_uf.yaml

* policy  = SFT best checkpoint, trainable, FP32 master weights + autocast
* reference = second copy of the SFT best checkpoint, frozen (eval mode,
  requires_grad=False, forward under torch.no_grad()). It is loaded in the
  same dtype and run under the same autocast as the policy, so at step 0 the
  policy and reference log-probs are bit-identical and every implicit reward
  is exactly 0 (reward accuracy 0, loss log 2).
* chosen and rejected rows are scored in one concatenated forward per model.
* validation: DPO loss, reward accuracy (strict r_w > r_l), reward margin,
  chosen / rejected rewards and log-probs; best checkpoint = highest
  validation reward accuracy (ties broken by margin), plus a final checkpoint.
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
    read_jsonl,
    resolve_precision,
    set_seed,
    warmup_cosine_lambda,
    write_json,
)

DEFAULTS = {
    "track": "uf",
    "sft_checkpoint": "outputs/sft/best",
    "reference_checkpoint": None,  # defaults to sft_checkpoint
    "train_path": "outputs/data/uf_train.jsonl",
    "val_path": "outputs/data/uf_val.jsonl",
    "output_dir": "outputs/dpo_uf",
    "max_length": 512,
    "max_train_pairs": 15000,
    "max_val_pairs": 1000,
    "per_device_batch_size": 4,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 8,
    "beta": 0.1,
    "learning_rate": 1.0e-6,
    "lr_schedule": "constant",
    "warmup_steps": 0,
    "optimizer": "adamw",  # adamw | rmsprop
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "num_epochs": 1,
    "precision": "auto",
    "gradient_checkpointing": False,
    "reference_device": "auto",  # auto -> cuda:1 if available else same device as policy
    "eval_every": 50,
    "log_every": 5,
    "seed": 42,
    "attn_implementation": "sdpa",
}


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return format_prompt(row["prompt"]), row["chosen"], row["rejected"]


class PreferenceCollator:
    """Picklable collate_fn returning {"chosen": LM batch, "rejected": LM batch}."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items):
        prompts, chosen, rejected = zip(*items)
        return core.build_preference_batch(self.tokenizer, list(prompts), list(chosen), list(rejected), self.max_length)


def _mean(xs):
    return float(sum(xs) / max(1, len(xs)))


@torch.no_grad()
def evaluate(policy, reference, loader, beta, pad_id, device, amp_dtype):
    policy.eval()
    agg = {k: [] for k in ("losses", "chosen_rewards", "rejected_rewards", "margins", "accuracies",
                           "policy_chosen_logps", "policy_rejected_logps")}
    for pair_batch in loader:
        with autocast_ctx(device, amp_dtype):
            _, stats = core.dpo_loss_from_batch(policy, reference, pair_batch, beta, pad_id)
        for key in agg:
            agg[key].extend(stats[key].float().cpu().tolist())
    policy.train()
    return {
        "val_loss": _mean(agg["losses"]),
        "val_reward_accuracy": _mean(agg["accuracies"]),
        "val_reward_margin": _mean(agg["margins"]),
        "val_chosen_reward": _mean(agg["chosen_rewards"]),
        "val_rejected_reward": _mean(agg["rejected_rewards"]),
        "val_policy_chosen_logp": _mean(agg["policy_chosen_logps"]),
        "val_policy_rejected_logp": _mean(agg["policy_rejected_logps"]),
        "val_pairs": len(agg["losses"]),
    }


def load_model(path, device, attn_implementation, trainable):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, attn_implementation=attn_implementation
    ).to(device)
    model.config.use_cache = False
    if trainable:
        model.train()
    else:
        model.eval()
        model.requires_grad_(False)
    return model


def save_checkpoint(model, tokenizer, path: Path, info: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(info, path / "checkpoint_info.json")


def is_better(metrics, best):
    if best is None:
        return True
    key = (metrics["val_reward_accuracy"], metrics["val_reward_margin"])
    return key > (best["val_reward_accuracy"], best["val_reward_margin"])


def main(argv=None):
    cfg = load_config(argv, DEFAULTS, "From-scratch DPO")
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    from transformers import AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.reference_device == "auto":
        ref_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    else:
        ref_device = torch.device(cfg.reference_device)
    amp_dtype, use_scaler, precision = resolve_precision(cfg.precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    reference_path = cfg.reference_checkpoint or cfg.sft_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_checkpoint)
    pad_id = core.get_pad_id(tokenizer)
    policy = load_model(cfg.sft_checkpoint, device, cfg.attn_implementation, trainable=True)
    if cfg.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    reference = load_model(reference_path, ref_device, cfg.attn_implementation, trainable=False)

    train_rows = read_jsonl(cfg.train_path)[: cfg.max_train_pairs]
    val_rows = read_jsonl(cfg.val_path)[: cfg.max_val_pairs]
    collate = PreferenceCollator(tokenizer, cfg.max_length)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        PreferenceDataset(train_rows), batch_size=cfg.per_device_batch_size, shuffle=True,
        collate_fn=collate, generator=generator, num_workers=2,
    )
    val_loader = DataLoader(
        PreferenceDataset(val_rows), batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=2
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    if cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=torch.cuda.is_available()
        )
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(total_steps, cfg.warmup_steps, cfg.lr_schedule)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    run_info = {
        "config": vars(cfg),
        "reference_checkpoint": str(reference_path),
        "precision": precision,
        "policy_device": str(device),
        "reference_device": str(ref_device),
        "effective_batch_size": cfg.per_device_batch_size * cfg.gradient_accumulation_steps,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "total_optimizer_steps": total_steps,
        "env": env_report(),
    }
    write_json(run_info, out_dir / "run_config.json")
    print(json.dumps({k: v for k, v in run_info.items() if k != "config"}, indent=2))

    timer = Timer()
    initial = evaluate(policy, reference, val_loader, cfg.beta, pad_id, device, amp_dtype)
    append_jsonl({"type": "eval", "step": 0, **initial}, metrics_path)
    print(f"[dpo-{cfg.track}] step 0 {initial}")

    best = None
    step = 0
    keys = ("losses", "accuracies", "margins", "chosen_rewards", "rejected_rewards",
            "policy_chosen_logps", "policy_rejected_logps")
    log_buf = {k: [] for k in keys}
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
            n_pairs = sum(b["chosen"]["input_ids"].shape[0] for b in window)
            for pair_batch in window:
                with autocast_ctx(device, amp_dtype):
                    loss, stats = core.dpo_loss_from_batch(policy, reference, pair_batch, cfg.beta, pad_id)
                # loss is a per-micro-batch mean; reweight so the window gradient is the mean over all pairs.
                micro_pairs = pair_batch["chosen"]["input_ids"].shape[0]
                scaler.scale(loss * (micro_pairs / n_pairs)).backward()
                for key in keys:
                    log_buf[key].extend(stats[key].float().cpu().tolist())

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                rec = {
                    "type": "train",
                    "step": step,
                    "epoch": step / max(1, steps_per_epoch),
                    "train_loss": _mean(log_buf["losses"]),
                    "train_reward_accuracy": _mean(log_buf["accuracies"]),
                    "train_reward_margin": _mean(log_buf["margins"]),
                    "train_chosen_reward": _mean(log_buf["chosen_rewards"]),
                    "train_rejected_reward": _mean(log_buf["rejected_rewards"]),
                    "train_policy_chosen_logp": _mean(log_buf["policy_chosen_logps"]),
                    "train_policy_rejected_logp": _mean(log_buf["policy_rejected_logps"]),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "elapsed_s": timer.elapsed(),
                }
                append_jsonl(rec, metrics_path)
                print(
                    f"[dpo-{cfg.track}] step {step}/{total_steps} loss={rec['train_loss']:.4f} "
                    f"acc={rec['train_reward_accuracy']:.3f} margin={rec['train_reward_margin']:.4f}"
                )
                log_buf = {k: [] for k in keys}

            if step % cfg.eval_every == 0 or step == total_steps:
                metrics = evaluate(policy, reference, val_loader, cfg.beta, pad_id, device, amp_dtype)
                append_jsonl({"type": "eval", "step": step, "elapsed_s": timer.elapsed(), **metrics}, metrics_path)
                print(
                    f"[dpo-{cfg.track}] eval step {step}: loss={metrics['val_loss']:.4f} "
                    f"acc={metrics['val_reward_accuracy']:.4f} margin={metrics['val_reward_margin']:.4f}"
                )
                if is_better(metrics, best):
                    best = {"step": step, **metrics}
                    save_checkpoint(policy, tokenizer, out_dir / "best", {**best, "reference": str(reference_path)})
                    print(f"[dpo-{cfg.track}] new best checkpoint at step {step}")

    save_checkpoint(policy, tokenizer, out_dir / "final", {"step": step, "reference": str(reference_path)})
    summary = {"best": best, "steps": step, "elapsed_s": timer.elapsed(), "initial": initial}
    write_json(summary, out_dir / "summary.json")
    print(f"[dpo-{cfg.track}] done: {json.dumps(summary)}")


if __name__ == "__main__":
    main(sys.argv[1:])
