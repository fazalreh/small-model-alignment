"""Small shared utilities: seeding, JSON(L) IO, configs, precision, schedules."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_jsonl(path) -> List[Dict[str, Any]]:
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(row: Dict[str, Any], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_config(argv: Optional[List[str]], defaults: Dict[str, Any], description: str) -> argparse.Namespace:
    """``--config file.yaml`` plus ``--key value`` overrides for every default."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    values = dict(defaults)
    if known.config:
        with open(known.config, "r", encoding="utf-8") as f:
            file_values = yaml.safe_load(f) or {}
        unknown = set(file_values) - set(values)
        if unknown:
            raise ValueError(f"unknown config keys in {known.config}: {sorted(unknown)}")
        values.update(file_values)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None)
    for key, value in values.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=_str2bool, default=value)
        elif value is None:
            parser.add_argument(f"--{key}", default=None)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    return parser.parse_args(argv)


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def resolve_precision(name: str):
    """Map a precision name to (autocast dtype or None, use GradScaler)."""
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            name = "bf16"
        elif torch.cuda.is_available():
            name = "fp16"
        else:
            name = "fp32"
    if name == "bf16":
        return torch.bfloat16, False, "bf16"
    if name == "fp16":
        return torch.float16, True, "fp16"
    if name == "fp32":
        return None, False, "fp32"
    raise ValueError(f"unknown precision {name!r}")


def autocast_ctx(device: torch.device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def warmup_cosine_lambda(total_steps: int, warmup_steps: int, schedule: str, min_ratio: float = 0.0):
    """LR multiplier: linear warmup, then cosine decay / constant."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
        raise ValueError(f"unknown schedule {schedule!r}")

    return fn


def param_groups_for_weight_decay(model, weight_decay: float):
    """No weight decay on biases, norms, and embeddings (standard practice)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name.lower() or "embed" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start


def env_report() -> Dict[str, Any]:
    import transformers

    info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        pass
    return info
