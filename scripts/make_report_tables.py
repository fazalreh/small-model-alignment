"""Markdown tables + example picks for the report from judged outputs.

    python scripts/make_report_tables.py --judged_dir outputs/judged --outputs outputs --out report/results.md

Reads ``*_judged_summary.json`` / ``*_judged_final.jsonl`` written by
scripts/judge_pairwise_kaggle.py, the DPO ``summary.json`` files, and the
generation ``.meta.json`` files (length / truncation statistics).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judged_dir", default="outputs/judged")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--out", default="report/results.md")
    p.add_argument("--examples_per_outcome", type=int, default=2)
    args = p.parse_args()

    lines = ["# Auto-generated results tables", ""]
    summaries = sorted(glob.glob(os.path.join(args.judged_dir, "*_judged_summary.json")))

    lines += ["## Pairwise results (candidate vs baseline, both orders judged; inconsistent -> tie)", "",
              "| comparison | criterion | n | W | T | L | unknown | fail | win rate | win rate excl. ties | ties = 1/2 (95% CI) | position consistency | mean chars cand / base |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    per_cat_blocks = []
    for path in summaries:
        s = load(path)
        for crit, c in s["criteria"].items():
            ci = c["win_rate_ties_half_ci95"]
            lines.append(
                f"| {s['comparison']} | {crit} | {c['n']} | {c['wins']} | {c['ties']} | {c['losses']} | {c['unknown']} | "
                f"{c['failures']} | {pct(c['win_rate'])} | {pct(c['win_rate_excl_ties'])} | "
                f"{pct(c['win_rate_ties_half'])} ({pct(ci[0])}-{pct(ci[1])}) | {pct(c['position_consistency'])} | "
                f"{s['mean_chars_candidate']:.0f} / {s['mean_chars_baseline']:.0f} |"
            )
            block = [f"### {s['comparison']} [{crit}] by category", "",
                     "| category | n | W | T | L | ties = 1/2 |", "|---|---|---|---|---|---|"]
            for cat, b in c["per_category"].items():
                block.append(f"| {cat} | {b['n']} | {b['wins']} | {b['ties']} | {b['losses']} | {pct(b['win_rate_ties_half'])} |")
            per_cat_blocks += block + [""]
    lines.append("")
    lines += [f"Judge model: {load(summaries[0])['judge_model'] if summaries else 'n/a'}", ""]
    lines += per_cat_blocks

    # DPO diagnostics
    lines += ["## DPO validation diagnostics", "",
              "| run | best step | val loss | val reward acc | val margin | r_w | r_l | policy logp chosen (init -> best) | policy logp rejected (init -> best) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for run in sorted(glob.glob(os.path.join(args.outputs, "dpo_*", "summary.json"))):
        s = load(run)
        b, i = s["best"], s["initial"]
        lines.append(
            f"| {Path(run).parent.name} | {b['step']} | {b['val_loss']:.4f} | {b['val_reward_accuracy']:.3f} | "
            f"{b['val_reward_margin']:.4f} | {b['val_chosen_reward']:.4f} | {b['val_rejected_reward']:.4f} | "
            f"{i['val_policy_chosen_logp']:.1f} -> {b['val_policy_chosen_logp']:.1f} | "
            f"{i['val_policy_rejected_logp']:.1f} -> {b['val_policy_rejected_logp']:.1f} |"
        )
    lines.append("")

    # generation statistics
    lines += ["## Generation statistics", "", "| eval set | model | mean new tokens | hit max_new_tokens |", "|---|---|---|---|"]
    for meta in sorted(glob.glob(os.path.join(args.outputs, "generations", "*", "*.json.meta.json"))):
        if "sanity" in meta:
            continue
        m = load(meta)
        lines.append(f"| {Path(meta).parent.name} | {m['generator']} | {m['mean_new_tokens']:.0f} | {pct(m['frac_hit_max_new_tokens'])} |")
    lines.append("")

    # representative examples
    lines += ["## Representative examples (for error analysis)", ""]
    for path in sorted(glob.glob(os.path.join(args.judged_dir, "*_judged_final.jsonl"))):
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        key = next(k for k in rows[0] if k.startswith("outcome_"))
        lines += [f"### {Path(path).name.replace('_judged_final.jsonl', '')}", ""]
        for outcome in ("win", "loss", "tie"):
            picks = sorted([r for r in rows if r[key] == outcome], key=lambda r: len(r["candidate_response"]) + len(r["baseline_response"]))
            for r in picks[: args.examples_per_outcome]:
                lines += [f"**{outcome.upper()}** `{r['id']}` ({r.get('category')}) - *{r['instruction'][:300]}*", "",
                          f"- **{r['candidate']}**: {r['candidate_response'][:600]!r}",
                          f"- **{r['baseline']}**: {r['baseline_response'][:600]!r}",
                          f"- judge: {' / '.join(x for x in r['judge_reasons'] if x)[:400]}", ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
