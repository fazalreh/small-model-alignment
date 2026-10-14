"""Judge-free style diagnostics for the persona (LoRA-DPO) track.

    python scripts/persona_style_metrics.py --gen_dir outputs/generations/persona --out outputs/judged/persona_style_metrics.json

For each model's generations on eval_sets/persona_style_eval_40.json it reports:
  * persona-marker rate: fraction of responses containing at least one chef-persona marker
    (exasperated exclamations or kitchen insults, from the persona training data's register)
  * exclamation marks per 100 words
  * repeated-4-gram rate: fraction of 4-grams that repeat an earlier 4-gram in the same response
    (measures degenerate looping)
  * mean words and the fraction of responses that hit max_new_tokens
overall and per category (cooking / coding / general_help / safety_sensitive).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

MARKERS = [
    r"\bdonkey\b", r"\bdisaster\b", r"\bidiot\b", r"\bsandwich\b", r"\bbloody\b", r"\bridiculous\b",
    r"\bdisgusting\b", r"\bpathetic\b", r"\bnonsense\b", r"\bmonster\b", r"\bjoke\b", r"\bunacceptable\b",
    r"\bare you (out of your mind|serious|kidding|insane)\b", r"\bwhat are you doing\b", r"\bcome on\b",
    r"\boh no\b", r"\bfor god'?s sake\b", r"\bshut it down\b", r"\byou can'?t even\b", r"\bwake up\b",
    r"\bseriously\?", r"\bwhat (on earth|the hell)\b", r"\bmy (gran|nan|mum)\b", r"\babsolute (mess|shambles)\b",
]
MARKER_RE = re.compile("|".join(MARKERS), re.I)


def repeated_4gram_rate(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


def stats(rows):
    n = len(rows)
    words = [len(re.findall(r"\w+", r["output"])) for r in rows]
    return {
        "n": n,
        "persona_marker_rate": sum(bool(MARKER_RE.search(r["output"])) for r in rows) / n,
        "exclamations_per_100_words": 100 * sum(r["output"].count("!") for r in rows) / max(1, sum(words)),
        "repeated_4gram_rate": sum(repeated_4gram_rate(r["output"]) for r in rows) / n,
        "mean_words": sum(words) / n,
        "frac_hit_max_new_tokens": sum(bool(r.get("hit_max_new_tokens")) for r in rows) / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", default="outputs/generations/persona")
    p.add_argument("--out", default="outputs/judged/persona_style_metrics.json")
    args = p.parse_args()
    result = {}
    for path in sorted(Path(args.gen_dir).glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r.get("category", "all")].append(r)
        result[path.stem] = {"overall": stats(rows), "per_category": {c: stats(v) for c, v in sorted(by_cat.items())}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'model':<28} {'marker%':>8} {'!/100w':>7} {'rep4g':>6} {'words':>6} {'capped':>7}")
    for name, r in result.items():
        o = r["overall"]
        print(f"{name:<28} {100*o['persona_marker_rate']:>7.0f}% {o['exclamations_per_100_words']:>7.2f} "
              f"{o['repeated_4gram_rate']:>6.3f} {o['mean_words']:>6.0f} {100*o['frac_hit_max_new_tokens']:>6.0f}%")


if __name__ == "__main__":
    main()
