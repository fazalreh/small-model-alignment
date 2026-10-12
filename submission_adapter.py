"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function is a thin wrapper
around the implementation in ``sma/core.py``, which the training scripts
(``sma/train_sft.py`` and ``sma/train_dpo.py``) use as well.

Documented choices:

* Prompt and response are tokenized separately without special tokens; EOS is
  appended to the response, then the concatenation is truncated to
  ``max_length`` (so EOS can be cut off for long responses).
* Rows are right-padded to the longest row in the batch (at most
  ``max_length``). Prompt and padding labels are ``-100``.
* ``compute_sft_loss`` uses the **token-mean** reduction: the sum of response
  token negative log-probabilities divided by the number of unmasked response
  tokens in the batch.
* ``compute_dpo_loss_from_logps`` / ``compute_dpo_loss`` return the batch mean
  of ``-log sigmoid(beta * [(pi_w - pi_l) - (ref_w - ref_l)])``; the reference
  model is run under ``torch.no_grad()``.
"""

from __future__ import annotations

import torch

from sma import core


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return core.build_lm_batch(tokenizer, prompts, responses, max_length)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = core.build_lm_batch(tokenizer, prompts, responses, max_length)
    logps, _ = core.forward_logprobs(model, batch)
    return logps


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    batch = core.build_lm_batch(tokenizer, prompts, responses, max_length)
    loss, _ = core.sft_loss_from_batch(model, batch)
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return core.build_preference_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    loss, _ = core.dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    return loss


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    pair_batch = core.build_preference_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    pad_id = core.get_pad_id(tokenizer)
    policy_chosen, policy_rejected = core.preference_logprobs(policy_model, pair_batch, pad_id)
    with torch.no_grad():
        reference_chosen, reference_rejected = core.preference_logprobs(reference_model, pair_batch, pad_id)
    loss, _ = core.dpo_loss_from_logps(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    )
    return loss
