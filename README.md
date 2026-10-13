# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

From-scratch SFT and DPO (no HF `Trainer`, no TRL) for `Qwen/Qwen2.5-0.5B`, plus a TRL `DPOTrainer` + LoRA
persona run on `HuggingFaceTB/SmolLM2-360M-Instruct`, blinded pairwise evaluation through Kaggle's model proxy,
and report tooling.

| What | Where |
|---|---|
| Core math: batching, masking, response log-probs, SFT and DPO losses | [`sma/core.py`](sma/core.py) |
| Grader interface (thin wrappers over `sma/core.py`) | [`submission_adapter.py`](submission_adapter.py) |
| Prompt template (Qwen ChatML, used for SFT, DPO, and eval) | [`sma/prompts.py`](sma/prompts.py) |
| Data subsets (UltraChat SFT, HH-RLHF, UltraFeedback, AlpacaEval-300/seed 42) | [`sma/data.py`](sma/data.py) |
| SFT loop | [`sma/train_sft.py`](sma/train_sft.py), [`configs/sft.yaml`](configs/sft.yaml) |
| DPO loop (same code for both tracks) | [`sma/train_dpo.py`](sma/train_dpo.py), [`configs/dpo_hh.yaml`](configs/dpo_hh.yaml), [`configs/dpo_uf.yaml`](configs/dpo_uf.yaml) |
| TRL DPOTrainer + LoRA persona | [`sma/train_lora_dpo.py`](sma/train_lora_dpo.py), [`configs/lora_dpo.yaml`](configs/lora_dpo.yaml) |
| Generation, blinded pair files | [`sma/generate.py`](sma/generate.py), [`sma/build_pairs.py`](sma/build_pairs.py) |
| Resumable Kaggle judge | [`scripts/judge_pairwise_kaggle.py`](scripts/judge_pairwise_kaggle.py), [`notebooks/kaggle_judge.ipynb`](notebooks/kaggle_judge.ipynb) |
| Plots and report tables | [`scripts/plot_training.py`](scripts/plot_training.py), [`scripts/make_report_tables.py`](scripts/make_report_tables.py) |
| H100 pipeline | [`modal_app.py`](modal_app.py) |
| Kaggle T4 x2 reference pipeline | [`notebooks/kaggle_t4_pipeline.ipynb`](notebooks/kaggle_t4_pipeline.ipynb) |
| Fixed eval sets | `eval_sets/hh_rlhf_eval_300_seed42.json`, `eval_sets/alpaca_eval_300_seed42.json`, `eval_sets/persona_style_eval_40.json` |

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests
```

`tests/test_public_adapter_smoke.py` holds the 7 public tests. `tests/test_core_edge_cases.py` adds 15 more:
truncation that drops EOS, prompts longer than `max_length`, `pad_token_id == eos_token_id` (as in Qwen), missing
pad or EOS tokens, row order, padding invariance, log-probs checked against a brute-force computation on a real
attention model, the token-mean SFT loss, the DPO formula, numerical stability, a frozen reference, the exact
`log 2` loss and zero rewards at initialization, gradient-accumulation equivalence, and a few-step DPO
optimization.

## Run on Modal (H100)

```bash
pip install modal
modal setup
modal run modal_app.py
```

Stages run in order: tests, data, SFT (about 15 min), then HH-DPO, UF-DPO and LoRA-DPO in parallel, then
generations (each model on AlpacaEval-300 and HH-eval-300, plus the persona set), blinded pair files, and plots.
To run a single stage, pass `--stage tests|data|sft|dpo|lora|generate|plots`. The optional
`--stage ablation` runs a UltraFeedback learning-rate sweep (5e-7, 3e-6) around the guide value and produces
pair files for it.

Download the artifacts:

```bash
modal volume get sma-outputs eval outputs/
modal volume get sma-outputs generations outputs/
modal volume get sma-outputs figures outputs/
modal volume get sma-outputs sft/metrics.jsonl outputs/sft/
modal volume get sma-outputs dpo_uf/metrics.jsonl outputs/dpo_uf/
modal volume get sma-outputs dpo_hh/metrics.jsonl outputs/dpo_hh/
modal volume get sma-outputs data/data_stats.json outputs/data/
```

## Judge on Kaggle (Task 7)

1. Create a private Kaggle dataset containing `outputs/eval/*_pairs.jsonl` and `scripts/judge_pairwise_kaggle.py`.
2. Create a benchmark-task notebook at https://www.kaggle.com/benchmarks/tasks/new (this gives `kaggle_benchmarks`
   model-proxy access), attach the dataset, and run the cells of `notebooks/kaggle_judge.ipynb`. These are the exact
   cells used for the submitted results.
3. Download `/kaggle/working/judged.zip` and unzip it into `outputs/judged/`.

The judge sends 5 pairs per request, sleeps at least 5 s between attempts, and retries up to 3 times. It
checkpoints after every batch, logs failures to `*_failures.jsonl`, keeps raw replies in `*_raw.jsonl`, and resumes
with `--reuse_judgments`. `--swap` judges both A/B orders; a verdict counts only when both orders agree, otherwise
the pair is a tie. Pair ids, model names and metadata never reach the judge prompt.

## Build the report

```bash
python scripts/plot_training.py --outputs outputs --fig_dir report/figures
python scripts/make_report_tables.py --judged_dir outputs/judged --outputs outputs --out report/results.md
python scripts/fill_report_results.py --judged_dir outputs/judged --repo https://github.com/<you>/small-model-alignment
modal run modal_app.py --stage report      # compiles report/report.tex in a TeX Live container
modal volume get sma-outputs report/report.pdf report/report.pdf --force
```

With a local LaTeX install, `cd report && latexmk -pdf report.tex` works too.

## Key design choices

- **Template.** Qwen ChatML with the system message `You are a helpful assistant.`. Each response ends with
  `<|endoftext|>` (the tokenizer EOS). Generation stops on `<|im_end|>` or `<|endoftext|>`.
- **Masking.** Prompt and response are tokenized separately and concatenated. EOS is appended before truncation.
  Rows are right-padded, and masks are built from row lengths, so the pad-equals-EOS case is handled correctly.
- **SFT loss reduction.** Token mean over all response tokens in the accumulation window, which equals the
  token-mean gradient of the full effective batch.
- **Precision.** bf16 autocast on H100, or fp16 autocast with GradScaler on T4, always with FP32 master weights.
  The DPO reference uses the same dtype and autocast as the policy, so at step 0 the implicit rewards are exactly
  0 and the loss is exactly `log 2`.
