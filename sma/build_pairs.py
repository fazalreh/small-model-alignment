"""Build blinded pairwise comparison files (candidate vs baseline).

    python -m sma.build_pairs --candidate outputs/generations/alpaca/uf_dpo.json \
        --baseline outputs/generations/alpaca/sft.json --rubric general \
        --name uf_dpo_vs_sft_alpaca --output_path outputs/eval/uf_dpo_vs_sft_alpaca_pairs.jsonl

Each line holds the prompt, ``response_a`` / ``response_b`` in a seeded random
order, and a ``metadata`` block with the position -> model mapping. The judge
script never puts ``metadata``, ``pair_id`` or generator names in the judge
prompt; it maps the A/B decision back to the model only after judging.
"""

from __future__ import annotations

import argparse
import random

from sma.utils import read_json, write_jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--rubric", required=True, choices=["general", "safety", "persona"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cand = read_json(args.candidate)
    base = {r["id"]: r for r in read_json(args.baseline)}
    rng = random.Random(f"{args.seed}-{args.name}")
    rows, n_a = [], 0
    for c in cand:
        b = base.get(c["id"])
        if b is None:
            raise KeyError(f"prompt {c['id']} missing from baseline file")
        assert b["instruction"] == c["instruction"], f"instruction mismatch for {c['id']}"
        candidate_first = rng.random() < 0.5
        n_a += candidate_first
        first, second = (c, b) if candidate_first else (b, c)
        rows.append(
            {
                "pair_id": f"{args.name}:{c['id']}",
                "comparison": args.name,
                "id": c["id"],
                "instruction": c["instruction"],
                "category": c.get("category", c.get("dataset", "")),
                "expected_behavior": c.get("expected_behavior"),
                "rubric": args.rubric,
                "response_a": first["output"],
                "response_b": second["output"],
                "metadata": {
                    "model_a": first["generator"],
                    "model_b": second["generator"],
                    "candidate": c["generator"],
                    "baseline": b["generator"],
                    "candidate_position": "A" if candidate_first else "B",
                },
            }
        )
    write_jsonl(rows, args.output_path)
    print(f"[pairs] {args.name}: {len(rows)} pairs, candidate in position A for {n_a} -> {args.output_path}")


if __name__ == "__main__":
    main()
