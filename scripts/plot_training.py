"""Training plots and a compact metrics table for the report.

    python scripts/plot_training.py --outputs outputs --fig_dir report/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ema(values, alpha):
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def plot_sft(run_dir: Path, fig_dir: Path, smooth: float):
    rows = read_jsonl(run_dir / "metrics.jsonl")
    tr = [r for r in rows if r["type"] == "train"]
    ev = [r for r in rows if r["type"] == "eval"]
    best = json.loads((run_dir / "best" / "checkpoint_info.json").read_text()) if (run_dir / "best" / "checkpoint_info.json").exists() else None
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot([r["step"] for r in tr], [r["train_loss"] for r in tr], color="#9ecae1", lw=0.8, label="train (raw, per log window)")
    ax.plot([r["step"] for r in tr], ema([r["train_loss"] for r in tr], smooth), color="#08519c", lw=1.6, label=f"train (EMA {smooth})")
    ax.plot([r["step"] for r in ev], [r["val_loss"] for r in ev], "o-", color="#d94801", lw=1.6, ms=3, label="validation (1,000 ex.)")
    if best:
        ax.axvline(best["step"], color="grey", ls="--", lw=1)
        ax.annotate(f"best ckpt (step {best['step']})", (best["step"], best["val_loss"]), textcoords="offset points",
                    xytext=(-110, 25), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL (token mean)")
    ax.set_title("SFT: Qwen2.5-0.5B on 30k UltraChat examples")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "sft_loss.png", dpi=200)
    plt.close(fig)
    return {"sft": {"initial_val_loss": ev[0]["val_loss"], "final_val_loss": ev[-1]["val_loss"],
                    "best": best, "steps": tr[-1]["step"] if tr else 0}}


def plot_dpo(runs: dict, fig_dir: Path, smooth: float):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {"dpo_hh": "#b2182b", "dpo_uf": "#2166ac"}
    labels = {"dpo_hh": "HH-RLHF DPO", "dpo_uf": "UltraFeedback DPO"}
    table = {}
    for name, run_dir in runs.items():
        rows = read_jsonl(run_dir / "metrics.jsonl")
        tr = [r for r in rows if r["type"] == "train"]
        ev = [r for r in rows if r["type"] == "eval"]
        c = colors.get(name, None)
        lab = labels.get(name, name)
        axes[0].plot([r["step"] for r in tr], ema([r["train_loss"] for r in tr], smooth), color=c, lw=1.2, alpha=0.6, label=f"{lab} train (EMA)")
        axes[0].plot([r["step"] for r in ev], [r["val_loss"] for r in ev], "o-", color=c, ms=3, lw=1.6, label=f"{lab} val")
        axes[1].plot([r["step"] for r in ev], [r["val_reward_accuracy"] for r in ev], "o-", color=c, ms=3, label=lab)
        axes[2].plot([r["step"] for r in ev], [r["val_reward_margin"] for r in ev], "o-", color=c, ms=3, label=f"{lab} margin")
        axes[2].plot([r["step"] for r in ev], [r["val_chosen_reward"] for r in ev], ":", color=c, lw=1, label=f"{lab} r_w")
        axes[2].plot([r["step"] for r in ev], [r["val_rejected_reward"] for r in ev], "--", color=c, lw=1, label=f"{lab} r_l")
        best_path = run_dir / "best" / "checkpoint_info.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        if best:
            for ax in axes:
                ax.axvline(best["step"], color=c, ls="--", lw=0.8, alpha=0.6)
        table[name] = {"initial": ev[0], "final": ev[-1], "best": best}
    axes[0].axhline(0.6931, color="grey", lw=0.8, ls=":")
    axes[0].set_title("DPO loss (dotted: log 2)")
    axes[1].set_title("validation reward accuracy (r_w > r_l)")
    axes[2].set_title("validation implicit rewards")
    for ax in axes:
        ax.set_xlabel("optimizer step (32 pairs / step)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "dpo_curves.png", dpi=200)
    plt.close(fig)
    return table


def plot_lora(run_dir: Path, fig_dir: Path):
    hist = json.loads((run_dir / "log_history.json").read_text())
    tr = [h for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [h for h in hist if "eval_loss" in h]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot([h["step"] for h in tr], [h["loss"] for h in tr], label="train")
    axes[0].plot([h["step"] for h in ev], [h["eval_loss"] for h in ev], "o-", label="val")
    axes[0].set_title("LoRA-DPO loss")
    key = "eval_rewards/accuracies"
    if ev and key in ev[0]:
        axes[1].plot([h["step"] for h in ev], [h[key] for h in ev], "o-", label="val reward acc")
        axes[1].plot([h["step"] for h in ev], [h["eval_rewards/margins"] for h in ev], "s-", label="val margin")
    axes[1].set_title("LoRA-DPO preference diagnostics")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "lora_dpo_curves.png", dpi=200)
    plt.close(fig)
    return {"lora_dpo": {"first_eval": ev[0] if ev else None, "last_eval": ev[-1] if ev else None}}


def plot_dpo_sweep(runs: dict, fig_dir: Path, name: str, title: str):
    """Validation reward accuracy and margin for several from-scratch DPO runs (an LR sweep)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    colors = ["#fcae91", "#de2d26", "#67000d", "#3182bd"]
    for (label, run_dir), c in zip(runs.items(), colors):
        ev = [r for r in read_jsonl(run_dir / "metrics.jsonl") if r["type"] == "eval"]
        steps = [r["step"] for r in ev]
        axes[0].plot(steps, [r["val_reward_accuracy"] for r in ev], "o-", ms=3, color=c, label=label)
        axes[1].plot(steps, [r["val_reward_margin"] for r in ev], "o-", ms=3, color=c, label=label)
    axes[0].set_title(f"{title}: validation reward accuracy")
    axes[1].set_title(f"{title}: validation reward margin")
    axes[0].set_ylim(0.5, None)
    for ax in axes:
        ax.set_xlabel("optimizer step (32 pairs / step)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=200)
    plt.close(fig)


def plot_lora_sweep(runs: dict, fig_dir: Path):
    """Overlay several LoRA-DPO runs: validation margin and chosen / rejected log-probs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    table = {}
    for (label, run_dir), c in zip(runs.items(), colors):
        hist = json.loads((run_dir / "log_history.json").read_text())
        ev = [h for h in hist if "eval_rewards/margins" in h]
        steps = [h["step"] for h in ev]
        axes[0].plot(steps, [h["eval_rewards/margins"] for h in ev], "o-", ms=3, color=c, label=label)
        axes[1].plot(steps, [h["eval_logps/chosen"] for h in ev], "o-", ms=3, color=c, label=f"{label}: chosen")
        axes[1].plot(steps, [h["eval_logps/rejected"] for h in ev], "s--", ms=3, color=c, alpha=0.7, label=f"{label}: rejected")
        table[label] = ev[-1]
    axes[0].set_title("LoRA-DPO validation reward margin")
    axes[1].set_title("validation log p (summed over response)")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "lora_sweep.png", dpi=200)
    plt.close(fig)
    return {"lora_sweep": table}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--fig_dir", default="report/figures")
    p.add_argument("--smooth", type=float, default=0.9)
    args = p.parse_args()
    out = Path(args.outputs)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if (out / "sft" / "metrics.jsonl").exists():
        summary.update(plot_sft(out / "sft", fig_dir, args.smooth))
    runs = {n: out / n for n in ("dpo_hh", "dpo_uf") if (out / n / "metrics.jsonl").exists()}
    if runs:
        summary["dpo"] = plot_dpo(runs, fig_dir, args.smooth)
    if (out / "lora_dpo" / "log_history.json").exists():
        summary.update(plot_lora(out / "lora_dpo", fig_dir))
    sweep = {
        "lr 5e-5, 3 ep": out / "lora_dpo",
        "lr 1e-4, 3 ep": out / "lora_dpo_lr1e-4_ep3",
        "lr 2e-4, 5 ep": out / "lora_dpo_lr2e-4_ep5",
    }
    sweep = {k: v for k, v in sweep.items() if (v / "log_history.json").exists()}
    if len(sweep) > 1:
        summary.update(plot_lora_sweep(sweep, fig_dir))
    hh_sweep = {
        "lr 1e-6 (guide)": out / "dpo_hh",
        "lr 3e-6 (selected)": out / "dpo_hh_lr3e-6",
        "lr 1e-5": out / "dpo_hh_lr1e-5",
    }
    hh_sweep = {k: v for k, v in hh_sweep.items() if (v / "metrics.jsonl").exists()}
    if len(hh_sweep) > 1:
        plot_dpo_sweep(hh_sweep, fig_dir, "dpo_hh_sweep.png", "HH-DPO")
    (fig_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
