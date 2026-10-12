"""Batched deterministic generation on a fixed evaluation set.

    python -m sma.generate --model_path outputs/sft/best --generator sft \
        --eval_path eval_sets/alpaca_eval_300_seed42.json --output_path outputs/generations/alpaca/sft.json

Output: a JSON list of {id, instruction, output, generator, dataset, category?}
plus ``<output>.meta.json`` with the exact decoding configuration.
Use ``--limit 5`` for the manual sanity check before a full run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from sma.prompts import format_prompt, format_with_chat_template, stop_token_ids
from sma.utils import read_json, set_seed, write_json

DECODING_DEFAULTS = dict(max_new_tokens=512, do_sample=False, temperature=None, top_p=None,
                         repetition_penalty=1.1, no_repeat_ngram_size=0)


def load_for_generation(model_path: str, base_model: str | None, dtype: torch.dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if base_model:  # PEFT adapter directory on top of a base model
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(model_path if Path(model_path, "tokenizer_config.json").exists() else base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device).eval()
    model.config.use_cache = True
    return model, tokenizer


def build_prompt(tokenizer, instruction: str, prompt_format: str) -> str:
    if prompt_format == "chatml":
        return format_prompt(instruction)
    if prompt_format == "chat_template":
        return format_with_chat_template(tokenizer, instruction)
    raise ValueError(prompt_format)


@torch.no_grad()
def generate_outputs(model, tokenizer, prompts, batch_size, decoding, device):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stops = stop_token_ids(tokenizer)
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -len(encoded[i]))  # length-sorted batches
    outputs = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        width = max(len(encoded[i]) for i in idx)
        input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):  # left padding for decoder-only generation
            ids = encoded[i]
            input_ids[row, width - len(ids):] = torch.tensor(ids)
            attention[row, width - len(ids):] = 1
        gen_kwargs = dict(
            max_new_tokens=decoding["max_new_tokens"],
            do_sample=decoding["do_sample"],
            repetition_penalty=decoding["repetition_penalty"],
            eos_token_id=stops,
            pad_token_id=pad_id,
        )
        if decoding["no_repeat_ngram_size"]:
            gen_kwargs["no_repeat_ngram_size"] = decoding["no_repeat_ngram_size"]
        if decoding["do_sample"]:
            gen_kwargs.update(temperature=decoding["temperature"], top_p=decoding["top_p"])
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        out = model.generate(input_ids=input_ids.to(device), attention_mask=attention.to(device), **gen_kwargs)
        new_tokens = out[:, width:]
        for row, i in enumerate(idx):
            toks = new_tokens[row].tolist()
            finished = False
            for k, t in enumerate(toks):
                if t in stops:
                    toks, finished = toks[:k], True
                    break
            text = tokenizer.decode(toks, skip_special_tokens=True).strip()
            outputs[i] = {"output": text, "n_tokens": len(toks), "hit_max_new_tokens": not finished}
        print(f"[gen] {min(start + batch_size, len(order))}/{len(order)}")
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--base_model", default=None, help="set when model_path is a PEFT/LoRA adapter")
    p.add_argument("--generator", required=True, help="name stored in the output (hidden from the judge)")
    p.add_argument("--eval_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--prompt_format", default="chatml", choices=["chatml", "chat_template"])
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    for key, value in DECODING_DEFAULTS.items():
        if key == "do_sample":
            p.add_argument(f"--{key}", type=int, default=int(value))
        else:
            p.add_argument(f"--{key}", type=float if isinstance(value, float) or value is None else int, default=value)
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model, tokenizer = load_for_generation(args.model_path, args.base_model, dtype, device)

    rows = read_json(args.eval_path)
    if args.limit:
        rows = rows[: args.limit]
    prompts = [build_prompt(tokenizer, r["instruction"], args.prompt_format) for r in rows]
    decoding = {k: getattr(args, k) for k in DECODING_DEFAULTS}
    decoding["do_sample"] = bool(decoding["do_sample"])
    decoding["max_new_tokens"] = int(decoding["max_new_tokens"])
    decoding["no_repeat_ngram_size"] = int(decoding["no_repeat_ngram_size"])

    t0 = time.time()
    gens = generate_outputs(model, tokenizer, prompts, args.batch_size, decoding, device)
    results = []
    for row, gen in zip(rows, gens):
        item = {
            "id": row["id"],
            "instruction": row["instruction"],
            "output": gen["output"],
            "generator": args.generator,
            "dataset": row.get("dataset", Path(args.eval_path).stem),
        }
        for extra in ("category", "expected_behavior"):
            if extra in row:
                item[extra] = row[extra]
        item["n_tokens"] = gen["n_tokens"]
        item["hit_max_new_tokens"] = gen["hit_max_new_tokens"]
        results.append(item)
    write_json(results, args.output_path)
    meta = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "generator": args.generator,
        "eval_path": args.eval_path,
        "n": len(results),
        "prompt_format": args.prompt_format,
        "prompt_example": prompts[0] if prompts else None,
        "stop_token_ids": stop_token_ids(tokenizer),
        "dtype": args.dtype,
        "decoding": decoding,
        "mean_new_tokens": sum(r["n_tokens"] for r in results) / max(1, len(results)),
        "frac_hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in results) / max(1, len(results)),
        "elapsed_s": time.time() - t0,
    }
    write_json(meta, str(args.output_path) + ".meta.json")
    print(f"[gen] wrote {len(results)} -> {args.output_path} ({meta['elapsed_s']:.0f}s, "
          f"mean tokens {meta['mean_new_tokens']:.0f}, truncated {meta['frac_hit_max_new_tokens']:.1%})")
    for r in results[: min(3, len(results))]:
        print("=" * 80, f"\n[{r['id']}] {r['instruction'][:200]}\n---\n{r['output'][:800]}")


if __name__ == "__main__":
    main()
