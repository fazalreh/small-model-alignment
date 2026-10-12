"""Run the whole pipeline on Modal H100s.

    pip install modal && modal setup           # once
    modal run modal_app.py                      # everything: tests -> data -> SFT -> 2x DPO + LoRA -> generations -> pairs -> plots
    modal run modal_app.py --stage sft          # or a single stage: tests|data|sft|dpo|lora|generate|plots|ablation|lora_ablation|report
    modal volume get sma-outputs eval ./outputs/eval            # download pair files for Kaggle judging
    modal volume get sma-outputs generations ./outputs/generations

All artifacts live in the Modal volume ``sma-outputs`` (mounted at
``/root/project/outputs``), which mirrors the local ``outputs/`` layout:
    data/  sft/  dpo_hh/  dpo_uf/  lora_dpo/  generations/  eval/  figures/
Every stage writes durable files as it goes and commits the volume when done,
so a stage can be re-run on its own after a failure.

Model judging is NOT done here: it uses Kaggle's free model proxy (see README).
"""

from __future__ import annotations

import shlex
import subprocess

import modal

APP_DIR = "/root/project"
OUT = f"{APP_DIR}/outputs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "datasets==3.6.0",
        "accelerate==1.6.0",
        "peft==0.15.2",
        "trl==0.19.1",
        "huggingface_hub>=0.30,<1.0",
        "tqdm",
        "pyyaml",
        "sentencepiece",
        "safetensors",
        "matplotlib",
        "pytest>=8",
        "numpy<2.3",
    )
    .env(
        {
            "PYTHONPATH": APP_DIR,
            "HF_HOME": "/hf-cache",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        ignore=["outputs", "outputs/**", ".git", ".git/**", "**/__pycache__", "**/*.pyc", ".venv", ".venv/**", "report/figures/**"],
    )
)

app = modal.App("small-model-alignment", image=image)
outputs_vol = modal.Volume.from_name("sma-outputs", create_if_missing=True)
hf_vol = modal.Volume.from_name("sma-hf-cache", create_if_missing=True)
VOLUMES = {OUT: outputs_vol, "/hf-cache": hf_vol}

BASE_QWEN = "Qwen/Qwen2.5-0.5B"
BASE_SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ALPACA = "eval_sets/alpaca_eval_300_seed42.json"
HH_EVAL = "eval_sets/hh_rlhf_eval_300_seed42.json"
PERSONA_EVAL = "eval_sets/persona_style_eval_40.json"


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(shlex.split(cmd), cwd=APP_DIR, check=True)


def commit() -> None:
    outputs_vol.commit()
    hf_vol.commit()


@app.function(cpu=4, timeout=1800)
def tests():
    sh("python -m pytest -q tests")


@app.function(cpu=8, memory=32768, volumes=VOLUMES, timeout=3 * 3600)
def prepare_data():
    sh(f"python -m sma.data --task all --out_dir {OUT}/data")
    commit()


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600)
def sft():
    sh("nvidia-smi")
    sh("python -c \"from sma.utils import env_report; print(env_report())\"")
    sh(f"python -m sma.train_sft --config configs/sft.yaml --output_dir {OUT}/sft "
       f"--train_path {OUT}/data/sft_train.jsonl --val_path {OUT}/data/sft_val.jsonl")
    commit()


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600)
def dpo(track: str, extra_args: str = "", run_name: str = ""):
    outputs_vol.reload()
    run = run_name or f"dpo_{track}"
    sh(f"python -m sma.train_dpo --config configs/dpo_{track}.yaml --sft_checkpoint {OUT}/sft/best "
       f"--train_path {OUT}/data/{track}_train.jsonl --val_path {OUT}/data/{track}_val.jsonl "
       f"--output_dir {OUT}/{run} {extra_args}")
    commit()


@app.function(gpu="H100", volumes=VOLUMES, timeout=2 * 3600)
def lora(extra_args: str = "", run_name: str = "lora_dpo"):
    sh(f"python -m sma.train_lora_dpo --config configs/lora_dpo.yaml --output_dir {OUT}/{run_name} {extra_args}")
    commit()


@app.function(gpu="H100", volumes=VOLUMES, timeout=2 * 3600)
def persona_variant(extra_args: str, run_name: str):
    """LoRA-DPO ablation: train, generate on the persona set, and pair against the base model."""
    outputs_vol.reload()
    sh(f"python -m sma.train_lora_dpo --config configs/lora_dpo.yaml --output_dir {OUT}/{run_name} {extra_args}")
    _gen(f"{OUT}/{run_name}/final", run_name, PERSONA_EVAL, "persona",
         f"--base_model {BASE_SMOL} --prompt_format chat_template")
    _pairs(run_name, "smollm2-360m-instruct", "persona", "persona")
    commit()


def _gen(model_path, name, eval_path, subdir, extra=""):
    sh(f"python -m sma.generate --model_path {model_path} --generator {name} --eval_path {eval_path} "
       f"--output_path {OUT}/generations/{subdir}/{name}.json {extra}")


def _pairs(candidate, baseline, subdir, rubric, name=None):
    name = name or f"{candidate}_vs_{baseline}_{subdir}"
    sh(f"python -m sma.build_pairs --candidate {OUT}/generations/{subdir}/{candidate}.json "
       f"--baseline {OUT}/generations/{subdir}/{baseline}.json --rubric {rubric} --name {name} "
       f"--output_path {OUT}/eval/{name}_pairs.jsonl")


@app.function(gpu="H100", volumes=VOLUMES, timeout=4 * 3600)
def generate(extra_dpo_runs: str = ""):
    """Sanity check on 5 prompts, then full generations + blinded pair files."""
    outputs_vol.reload()
    qwen_models = {
        "qwen2.5-0.5b-base": BASE_QWEN,
        "sft": f"{OUT}/sft/best",
        "dpo_uf": f"{OUT}/dpo_uf/best",
        "dpo_hh": f"{OUT}/dpo_hh/best",
    }
    for run in filter(None, extra_dpo_runs.split(",")):
        qwen_models[run] = f"{OUT}/{run}/best"

    # Task 6 sanity check: 5 prompts per eval set, printed for manual inspection.
    for subdir, path in (("alpaca", ALPACA), ("hh", HH_EVAL)):
        _gen(f"{OUT}/sft/best", "sft", path, f"sanity_{subdir}", "--limit 5")
    _gen(f"{OUT}/lora_dpo/final", "lora_dpo", PERSONA_EVAL, "sanity_persona",
         f"--limit 5 --base_model {BASE_SMOL} --prompt_format chat_template")

    for name, path in qwen_models.items():
        _gen(path, name, ALPACA, "alpaca")
        _gen(path, name, HH_EVAL, "hh")
        commit()
    _gen(BASE_SMOL, "smollm2-360m-instruct", PERSONA_EVAL, "persona", "--prompt_format chat_template")
    _gen(f"{OUT}/lora_dpo/final", "lora_dpo", PERSONA_EVAL, "persona",
         f"--base_model {BASE_SMOL} --prompt_format chat_template")

    # Required comparisons
    _pairs("dpo_uf", "sft", "alpaca", "general")
    _pairs("dpo_hh", "sft", "hh", "safety")
    _pairs("lora_dpo", "smollm2-360m-instruct", "persona", "persona")
    # Extra analysis: cross-set transfer / alignment tax, and SFT vs base
    _pairs("dpo_hh", "sft", "alpaca", "general")
    _pairs("dpo_uf", "sft", "hh", "safety")
    _pairs("sft", "qwen2.5-0.5b-base", "alpaca", "general")
    for run in filter(None, extra_dpo_runs.split(",")):
        _pairs(run, "sft", "alpaca", "general")
        _pairs(run, "sft", "hh", "safety")
    commit()


@app.function(gpu="H100", volumes=VOLUMES, timeout=2 * 3600)
def generate_runs(runs: str):
    """Generate + pair only the given extra DPO runs (comma-separated output dirs) on both eval sets."""
    outputs_vol.reload()
    for run in filter(None, runs.split(",")):
        for subdir, path, rubric in (("hh", HH_EVAL, "safety"), ("alpaca", ALPACA, "general")):
            _gen(f"{OUT}/{run}/best", run, path, subdir)
            _pairs(run, "sft", subdir, rubric)
        commit()


@app.function(cpu=2, volumes=VOLUMES, timeout=1800)
def plots():
    outputs_vol.reload()
    sh(f"python scripts/plot_training.py --outputs {OUT} --fig_dir {OUT}/figures")
    commit()


latex_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra",
                 "texlive-fonts-recommended", "lmodern", "latexmk", "poppler-utils")
    .add_local_dir("report", remote_path="/root/report", ignore=["*.pdf", "*.aux", "*.log"])
)


@app.function(image=latex_image, volumes={OUT: outputs_vol}, timeout=1800)
def compile_report():
    """latexmk report/report_final.tex (or report.tex) -> outputs/report/report.pdf (+ page previews)"""
    import shutil
    from pathlib import Path

    outputs_vol.reload()
    work = Path("/tmp/report")
    shutil.copytree("/root/report", work, dirs_exist_ok=True)
    if not (work / "figures").exists() and Path(f"{OUT}/figures").exists():  # prefer local report/figures
        shutil.copytree(f"{OUT}/figures", work / "figures", dirs_exist_ok=True)
    stem = "report_final" if (work / "report_final.tex").exists() else "report"
    print(f"compiling {stem}.tex")
    proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
                          cwd=work, capture_output=True, text=True)
    print(proc.stdout[-4000:])
    dest = Path(f"{OUT}/report")
    dest.mkdir(parents=True, exist_ok=True)
    if (work / f"{stem}.log").exists():
        shutil.copy(work / f"{stem}.log", dest / "report.log")
    if proc.returncode != 0:
        raise RuntimeError("latexmk failed; see outputs/report/report.log")
    shutil.copy(work / f"{stem}.pdf", dest / "report.pdf")
    shutil.rmtree(dest / "pages", ignore_errors=True)
    subprocess.run(["pdftoppm", "-png", "-r", "60", f"{stem}.pdf", "page"], cwd=work, check=True)
    (dest / "pages").mkdir(exist_ok=True)
    for png in work.glob("page-*.png"):
        shutil.copy(png, dest / "pages" / png.name)
    pages = subprocess.run(["bash", "-c", f"grep -a 'Output written' {stem}.log"], cwd=work, capture_output=True, text=True)
    print(pages.stdout)
    outputs_vol.commit()


@app.local_entrypoint()
def main(stage: str = "all"):
    if stage in ("all", "tests"):
        tests.remote()
    if stage in ("all", "data"):
        prepare_data.remote()
    if stage in ("all", "sft"):
        sft.remote()
    if stage in ("all", "dpo", "lora"):
        calls = []
        if stage in ("all", "dpo"):
            calls += [dpo.spawn("hh"), dpo.spawn("uf")]
        if stage in ("all", "lora"):
            calls.append(lora.spawn())
        for call in calls:
            call.get()
    if stage == "ablation":
        # learning-rate sweep around the guide value for the UltraFeedback track
        calls = [
            dpo.spawn("uf", "--learning_rate 5.0e-7", "dpo_uf_lr5e-7"),
            dpo.spawn("uf", "--learning_rate 3.0e-6", "dpo_uf_lr3e-6"),
        ]
        for call in calls:
            call.get()
        generate.remote("dpo_uf_lr5e-7,dpo_uf_lr3e-6")
    if stage == "lora_ablation":
        persona_variant.remote("--learning_rate 2.0e-4 --num_train_epochs 5", "lora_dpo_lr2e-4_ep5")
    if stage in ("all", "generate"):
        generate.remote()
    if stage in ("all", "plots", "generate", "ablation"):
        plots.remote()
    if stage in ("all", "report"):
        compile_report.remote()
    print(
        "\nDone. Download artifacts with e.g.\n"
        "  modal volume get sma-outputs eval outputs/\n"
        "  modal volume get sma-outputs generations outputs/\n"
        "  modal volume get sma-outputs figures outputs/\n"
        "  modal volume get sma-outputs sft/metrics.jsonl outputs/sft/\n"
        "Then upload outputs/eval/*_pairs.jsonl to a Kaggle dataset and run the judge notebook."
    )
