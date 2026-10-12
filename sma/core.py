"""Core SFT / DPO math shared by the training scripts and submission_adapter.py.

Conventions (used everywhere in this repo):

* A training row is ``prompt_ids + response_ids + [eos]``. The prompt and the
  response are tokenized *separately* (no special tokens added by the
  tokenizer) and concatenated, so the prompt/response boundary is exact.
* The EOS token is appended to the response *before* truncation; truncation
  keeps the first ``max_length`` tokens, so a long response may lose its EOS.
* Padding is on the right, up to the longest row in the batch (never longer
  than ``max_length``). Attention-mask and label masks are derived from the
  true row lengths, never from ``input_ids == pad_token_id``; this matters for
  tokenizers such as Qwen2.5 where ``pad_token_id == eos_token_id``.
* ``labels`` hold the target token id on response positions and ``-100`` on
  prompt and padding positions. The causal shift happens in
  :func:`sequence_logprobs` (logits at ``t`` predict the label at ``t+1``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _encode(tokenizer, text: str) -> List[int]:
    """Tokenize ``text`` without BOS/EOS or other automatically added tokens."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # tokenizers whose encode() has no add_special_tokens kwarg
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(i) for i in ids]


def get_eos_id(tokenizer) -> Optional[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return None if eos is None else int(eos)


def get_pad_id(tokenizer) -> int:
    """Pad id with fallbacks for tokenizers that define no pad token."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = get_eos_id(tokenizer)
    return 0 if pad is None else int(pad)


def _as_list(texts) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def tokenize_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    add_eos: bool = True,
) -> Tuple[List[int], List[int]]:
    """Return ``(input_ids, labels)`` for one prompt/response pair (unpadded)."""
    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    eos_id = get_eos_id(tokenizer)
    if add_eos and eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
    if max_length is not None and max_length > 0:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return input_ids, labels


def collate_tokenized(
    rows: Sequence[Tuple[List[int], List[int]]],
    pad_id: int,
    pad_to: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Right-pad pre-tokenized ``(input_ids, labels)`` rows into tensors."""
    width = max([len(ids) for ids, _ in rows] + [1])
    if pad_to is not None:
        width = max(width, int(pad_to))
    batch = len(rows)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, width), dtype=torch.long)
    labels = torch.full((batch, width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, labs) in enumerate(rows):
        n = len(ids)
        if n == 0:
            continue
        input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :n] = 1
        labels[row, :n] = torch.tensor(labs, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_lm_batch(
    tokenizer,
    prompts,
    responses,
    max_length: int,
    pad_to: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize, truncate, right-pad, and mask a batch of prompt/response pairs."""
    prompts = _as_list(prompts)
    responses = _as_list(responses)
    if len(prompts) != len(responses):
        raise ValueError(f"got {len(prompts)} prompts but {len(responses)} responses")
    rows = [tokenize_example(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    return collate_tokenized(rows, get_pad_id(tokenizer), pad_to=pad_to)


def build_preference_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Separate response-masked LM batches for chosen and rejected rows."""
    prompts = _as_list(prompts)
    chosen_responses = _as_list(chosen_responses)
    rejected_responses = _as_list(rejected_responses)
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must have equal length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def pad_batch_to(batch: Dict[str, torch.Tensor], width: int, pad_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad an LM batch to ``width`` columns (no-op if already that wide)."""
    extra = width - batch["input_ids"].shape[1]
    if extra <= 0:
        return batch
    return {
        "input_ids": F.pad(batch["input_ids"], (0, extra), value=pad_id),
        "attention_mask": F.pad(batch["attention_mask"], (0, extra), value=0),
        "labels": F.pad(batch["labels"], (0, extra), value=IGNORE_INDEX),
    }


def concat_preference_batch(pair_batch, pad_id: int) -> Dict[str, torch.Tensor]:
    """Stack chosen rows then rejected rows into one batch for a single forward."""
    chosen, rejected = pair_batch["chosen"], pair_batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    chosen = pad_batch_to(chosen, width, pad_id)
    rejected = pad_batch_to(rejected, width, pad_id)
    return {key: torch.cat([chosen[key], rejected[key]], dim=0) for key in chosen}


# ---------------------------------------------------------------------------
# Log-probabilities and losses
# ---------------------------------------------------------------------------


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Summed log p(label) over unmasked positions, plus the token counts.

    ``logits[:, t]`` is the next-token distribution after reading position
    ``t``, so it scores ``labels[:, t + 1]``. Computation is done in fp32.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(shift_logits.device)
    mask = shift_labels.ne(IGNORE_INDEX)
    # cross_entropy(reduction="none") == -log_softmax gathered at the label;
    # ignore_index positions come back as exactly 0.
    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    token_logps = -token_nll * mask
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def forward_logprobs(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` on an LM batch and return (summed response logps, token counts)."""
    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return sequence_logprobs(logits, batch["labels"].to(device))


def sft_loss_from_batch(model, batch, token_normalizer: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Response-only NLL using the *token-mean* reduction.

    Returns ``(loss, n_tokens)``. ``loss = -sum(logp) / token_normalizer`` where
    the normalizer defaults to the number of response tokens in this batch.
    Training passes the token count of the whole accumulation window, so the
    accumulated gradient equals the gradient of the token-mean over the full
    effective batch.
    """
    logps, counts = forward_logprobs(model, batch)
    n_tokens = counts.sum()
    denom = n_tokens.clamp_min(1).to(logps.dtype) if token_normalizer is None else float(token_normalizer)
    return -logps.sum() / denom, n_tokens


def dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DPO objective  -log sigmoid(beta * [(pi_w - pi_l) - (ref_w - ref_l)]).

    Returns ``(loss, stats)`` where ``stats`` holds the per-example implicit
    rewards ``r_w = beta (pi_w - ref_w)``, ``r_l = beta (pi_l - ref_l)`` and
    the per-example losses. Reference terms are detached.
    """
    pc = torch.as_tensor(policy_chosen_logps)
    device = pc.device
    dtype = pc.dtype if pc.is_floating_point() else torch.float32
    pc = pc.to(dtype)
    pr = torch.as_tensor(policy_rejected_logps).to(device=device, dtype=dtype)
    rc = torch.as_tensor(reference_chosen_logps).detach().to(device=device, dtype=dtype)
    rr = torch.as_tensor(reference_rejected_logps).detach().to(device=device, dtype=dtype)

    policy_gap = pc - pr
    reference_gap = rc - rr
    logits = beta * (policy_gap - reference_gap)
    losses = -F.logsigmoid(logits)  # numerically stable form of -log(sigmoid(x))

    chosen_rewards = beta * (pc - rc).detach()
    rejected_rewards = beta * (pr - rr).detach()
    stats = {
        "losses": losses.detach(),
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "margins": chosen_rewards - rejected_rewards,
        "accuracies": (chosen_rewards > rejected_rewards).float(),
    }
    if reduction == "mean":
        loss = losses.mean()
    elif reduction == "sum":
        loss = losses.sum()
    elif reduction == "none":
        loss = losses
    else:
        raise ValueError(f"unknown reduction {reduction!r}")
    return loss, stats


def preference_logprobs(model, pair_batch, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chosen and rejected summed response logps from one concatenated forward."""
    n = pair_batch["chosen"]["input_ids"].shape[0]
    stacked = concat_preference_batch(pair_batch, pad_id)
    logps, _ = forward_logprobs(model, stacked)
    return logps[:n], logps[n:]


def dpo_loss_from_batch(
    policy_model,
    reference_model,
    pair_batch,
    beta: float,
    pad_id: int,
    reference_logps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full DPO step: policy forward with grad, reference forward under no_grad."""
    policy_chosen, policy_rejected = preference_logprobs(policy_model, pair_batch, pad_id)
    if reference_logps is None:
        with torch.no_grad():
            ref_chosen, ref_rejected = preference_logprobs(reference_model, pair_batch, pad_id)
    else:
        ref_chosen, ref_rejected = reference_logps
    loss, stats = dpo_loss_from_logps(
        policy_chosen, policy_rejected, ref_chosen.detach(), ref_rejected.detach(), beta
    )
    stats.update(
        policy_chosen_logps=policy_chosen.detach(),
        policy_rejected_logps=policy_rejected.detach(),
        reference_chosen_logps=ref_chosen.detach().to(policy_chosen.device),
        reference_rejected_logps=ref_rejected.detach().to(policy_chosen.device),
    )
    return loss, stats
