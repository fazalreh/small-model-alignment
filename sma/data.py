"""Build and save the exact data subsets used by every run.

    python -m sma.data --task all --out_dir outputs/data

Writes (all single-turn, raw instruction text; the chat template is applied by
the training / generation code via ``sma.prompts.format_prompt``):

    outputs/data/sft_train.jsonl, sft_val.jsonl             {id, prompt, response}
    outputs/data/hh_train.jsonl,  hh_val.jsonl              {id, prompt, chosen, rejected, source}
    outputs/data/uf_train.jsonl,  uf_val.jsonl              {id, prompt, chosen, rejected, score_*}
    eval_sets/alpaca_eval_300_seed42.json                   {id, instruction, dataset, ...}
    outputs/data/data_stats.json                            filtering statistics
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sma.prompts import format_prompt
from sma.utils import read_json, write_json, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
HH_EVAL_PATH = REPO_ROOT / "eval_sets" / "hh_rlhf_eval_300_seed42.json"
ALPACA_EVAL_PATH = REPO_ROOT / "eval_sets" / "alpaca_eval_300_seed42.json"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def eval_instruction_set() -> set:
    """Normalized eval prompts, used to drop exact-match contamination from training data."""
    seen = set()
    for path in (HH_EVAL_PATH, ALPACA_EVAL_PATH):
        if path.exists():
            seen.update(_norm(row["instruction"]) for row in read_json(path))
    return seen


class LengthFilter:
    """Token-length checks with the same template / EOS convention as training."""

    def __init__(self, tokenizer, max_length: int, max_prompt_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def lengths(self, texts: List[str]) -> List[int]:
        out = []
        for start in range(0, len(texts), 1000):
            enc = self.tokenizer(texts[start : start + 1000], add_special_tokens=False)["input_ids"]
            out.extend(len(ids) for ids in enc)
        return out

    def prompt_ok(self, prompt_lengths: List[int]) -> List[bool]:
        return [n <= self.max_prompt_length for n in prompt_lengths]


def _take(rows: Iterable[Dict], n: int) -> List[Dict]:
    out = []
    for row in rows:
        out.append(row)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# SFT: UltraChat 200k (first user turn + first assistant turn)
# ---------------------------------------------------------------------------


def build_sft(tokenizer, args, stats: Dict, contaminated: set) -> None:
    from datasets import load_dataset

    lf = LengthFilter(tokenizer, args.sft_max_length, args.sft_max_prompt_length)

    def candidates(split: str, seed: int):
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=seed)
        chunk = 4000
        for start in range(0, len(ds), chunk):
            part = ds.select(range(start, min(start + chunk, len(ds))))
            rows = []
            for ex in part:
                msgs = ex["messages"]
                if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
                    continue
                prompt, response = msgs[0]["content"].strip(), msgs[1]["content"].strip()
                if not prompt or not response or _norm(prompt) in contaminated:
                    continue
                rows.append({"id": ex["prompt_id"], "prompt": prompt, "response": response})
            p_len = lf.lengths([format_prompt(r["prompt"]) for r in rows])
            r_len = lf.lengths([r["response"] for r in rows])
            for row, pl, rl in zip(rows, p_len, r_len):
                stats_key = f"sft_{split}"
                stats.setdefault(stats_key, {"seen": 0, "kept": 0})
                stats[stats_key]["seen"] += 1
                if pl > args.sft_max_prompt_length:
                    continue
                # +1 for the EOS token; require the full example to fit so the model learns to stop.
                if args.sft_require_fit and pl + rl + 1 > args.sft_max_length:
                    continue
                stats[stats_key]["kept"] += 1
                yield row

    train = _take(candidates("train_sft", args.seed), args.sft_train_size)
    val = _take(candidates("test_sft", args.seed), args.sft_val_size)
    for i, row in enumerate(train):
        row["id"] = f"sft-train-{i:05d}-{row['id'][:8]}"
    for i, row in enumerate(val):
        row["id"] = f"sft-val-{i:05d}-{row['id'][:8]}"
    write_jsonl(train, Path(args.out_dir) / "sft_train.jsonl")
    write_jsonl(val, Path(args.out_dir) / "sft_val.jsonl")
    stats["sft_sizes"] = {"train": len(train), "val": len(val)}
    print(f"[data] SFT train={len(train)} val={len(val)}")


# ---------------------------------------------------------------------------
# HH-RLHF (single-turn only)
# ---------------------------------------------------------------------------

_HH_TURN = re.compile(r"\n\n(Human|Assistant): ")


def parse_hh_single_turn(chosen: str, rejected: str) -> Optional[Dict[str, str]]:
    """Return {prompt, chosen, rejected} for single-turn HH pairs, else None."""

    def split(text: str):
        if not text.startswith("\n\nHuman: "):
            text = "\n\n" + text.lstrip("\n")
        parts = _HH_TURN.split(text)
        # parts = ['', 'Human', msg, 'Assistant', msg]
        if len(parts) != 5 or parts[1] != "Human" or parts[3] != "Assistant":
            return None
        return parts[2].strip(), parts[4].strip()

    c, r = split(chosen), split(rejected)
    if c is None or r is None or c[0] != r[0]:
        return None
    prompt, chosen_resp, rejected_resp = c[0], c[1], r[1]
    if not prompt or not chosen_resp or not rejected_resp or chosen_resp == rejected_resp:
        return None
    return {"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp}


def build_hh(tokenizer, args, stats: Dict, contaminated: set) -> None:
    from datasets import load_dataset

    lf = LengthFilter(tokenizer, args.dpo_max_length, args.dpo_max_prompt_length)

    def load_subset(subset: str, split: str) -> List[Dict]:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
        rows = []
        for i, ex in enumerate(ds):
            parsed = parse_hh_single_turn(ex["chosen"], ex["rejected"])
            if parsed is None or _norm(parsed["prompt"]) in contaminated:
                continue
            parsed["source"] = subset
            parsed["id"] = f"{subset}-{split}-{i}"
            rows.append(parsed)
        p_len = lf.lengths([format_prompt(r["prompt"]) for r in rows])
        kept = [r for r, n in zip(rows, p_len) if n <= args.dpo_max_prompt_length]
        stats[f"hh_{subset}_{split}"] = {"total": len(ds), "single_turn": len(rows), "kept": len(kept)}
        random.Random(args.seed).shuffle(kept)
        return kept

    def mix(split: str, n: int) -> List[Dict]:
        harmless = load_subset("harmless-base", split)
        helpful = load_subset("helpful-base", split)
        n_harmless = min(len(harmless), int(round(n * args.hh_harmless_fraction)))
        n_helpful = min(len(helpful), n - n_harmless)
        n_harmless = min(len(harmless), n - n_helpful)  # top up if helpful ran short
        rows = harmless[:n_harmless] + helpful[:n_helpful]
        random.Random(args.seed).shuffle(rows)
        return rows

    train = mix("train", args.dpo_train_size)
    val = mix("test", args.dpo_val_size)
    write_jsonl(train, Path(args.out_dir) / "hh_train.jsonl")
    write_jsonl(val, Path(args.out_dir) / "hh_val.jsonl")
    stats["hh_sizes"] = {
        "train": len(train),
        "val": len(val),
        "train_harmless": sum(r["source"] == "harmless-base" for r in train),
        "val_harmless": sum(r["source"] == "harmless-base" for r in val),
    }
    print(f"[data] HH train={len(train)} val={len(val)} {stats['hh_sizes']}")


# ---------------------------------------------------------------------------
# UltraFeedback (binarized, single-turn)
# ---------------------------------------------------------------------------


def build_uf(tokenizer, args, stats: Dict, contaminated: set) -> None:
    from datasets import load_dataset

    lf = LengthFilter(tokenizer, args.dpo_max_length, args.dpo_max_prompt_length)

    def load_split(split: str, n: int) -> List[Dict]:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).shuffle(seed=args.seed)
        rows, counts = [], {"total": len(ds), "multi_turn": 0, "low_margin": 0, "dup_or_empty": 0, "contaminated": 0}
        for ex in ds:
            if len(ex["chosen"]) != 2 or len(ex["rejected"]) != 2:
                counts["multi_turn"] += 1
                continue
            prompt = ex["prompt"].strip()
            chosen = ex["chosen"][-1]["content"].strip()
            rejected = ex["rejected"][-1]["content"].strip()
            if not prompt or not chosen or not rejected or chosen == rejected:
                counts["dup_or_empty"] += 1
                continue
            if _norm(prompt) in contaminated:
                counts["contaminated"] += 1
                continue
            if ex["score_chosen"] - ex["score_rejected"] < args.uf_min_score_margin:
                counts["low_margin"] += 1
                continue
            rows.append(
                {
                    "id": f"uf-{split}-{ex['prompt_id'][:12]}",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "score_chosen": ex["score_chosen"],
                    "score_rejected": ex["score_rejected"],
                }
            )
        p_len = lf.lengths([format_prompt(r["prompt"]) for r in rows])
        kept = [r for r, pl in zip(rows, p_len) if pl <= args.dpo_max_prompt_length]
        counts["prompt_too_long"] = len(rows) - len(kept)
        counts["kept_before_cap"] = len(kept)
        stats[f"uf_{split}"] = counts
        return kept[:n]

    train = load_split("train_prefs", args.dpo_train_size)
    val = load_split("test_prefs", args.dpo_val_size)
    write_jsonl(train, Path(args.out_dir) / "uf_train.jsonl")
    write_jsonl(val, Path(args.out_dir) / "uf_val.jsonl")
    stats["uf_sizes"] = {"train": len(train), "val": len(val)}
    print(f"[data] UF train={len(train)} val={len(val)}")


# ---------------------------------------------------------------------------
# AlpacaEval fixed subset (300 prompts, seed 42)
# ---------------------------------------------------------------------------


def build_alpaca(args, stats: Dict) -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    full = read_json(path)
    indices = random.Random(args.seed).sample(range(len(full)), args.alpaca_size)
    rows = []
    for k, idx in enumerate(indices):
        ex = full[idx]
        rows.append(
            {
                "id": f"alpaca-eval-{args.seed}-{k:04d}",
                "instruction": ex["instruction"],
                "dataset": "alpaca_eval",
                "category": ex.get("dataset", "alpaca_eval"),
                "source_index": idx,
            }
        )
    write_json(rows, ALPACA_EVAL_PATH)
    stats["alpaca_eval"] = {"full_size": len(full), "subset": len(rows), "seed": args.seed}
    print(f"[data] AlpacaEval subset={len(rows)} -> {ALPACA_EVAL_PATH}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="all", choices=["all", "sft", "hh", "uf", "alpaca"])
    parser.add_argument("--out_dir", default="outputs/data")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_train_size", type=int, default=30000)
    parser.add_argument("--sft_val_size", type=int, default=1000)
    parser.add_argument("--sft_max_length", type=int, default=512)
    parser.add_argument("--sft_max_prompt_length", type=int, default=256)
    parser.add_argument("--sft_require_fit", type=int, default=1)
    parser.add_argument("--dpo_train_size", type=int, default=15000)
    parser.add_argument("--dpo_val_size", type=int, default=1000)
    parser.add_argument("--dpo_max_length", type=int, default=512)
    parser.add_argument("--dpo_max_prompt_length", type=int, default=256)
    parser.add_argument("--hh_harmless_fraction", type=float, default=0.5)
    parser.add_argument("--uf_min_score_margin", type=float, default=1.0)
    parser.add_argument("--alpaca_size", type=int, default=300)
    args = parser.parse_args(argv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.out_dir) / "data_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    stats["args"] = vars(args)

    if args.task in ("all", "alpaca"):
        build_alpaca(args, stats)
    contaminated = eval_instruction_set()
    stats["n_eval_prompts_for_decontamination"] = len(contaminated)

    if args.task != "alpaca":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if args.task in ("all", "sft"):
            build_sft(tokenizer, args, stats, contaminated)
        if args.task in ("all", "hh"):
            build_hh(tokenizer, args, stats, contaminated)
        if args.task in ("all", "uf"):
            build_uf(tokenizer, args, stats, contaminated)
    write_json(stats, stats_path)


if __name__ == "__main__":
    main()
