"""Extra checks beyond the public smoke tests: truncation boundaries, unusual
tokenizer configurations, mixed lengths, row order, numerical correctness
against brute-force computations, and small optimization steps."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from sma import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PadIsEosTokenizer(ToyTokenizer):
    """Qwen-like: pad token id equals eos token id."""

    pad_token_id = 1


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None  # token string kept only so the toy encoder still works


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class TinyAttnLM(torch.nn.Module):
    """Small real causal model (position-dependent) for brute-force checks."""

    def __init__(self, vocab_size, dim=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Embedding(vocab_size, dim)
        self.pos = torch.nn.Embedding(64, dim)
        self.attn = torch.nn.MultiheadAttention(dim, 2, batch_first=True)
        self.out = torch.nn.Linear(dim, vocab_size)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)

    def forward(self, input_ids, attention_mask=None):
        T = input_ids.shape[1]
        h = self.emb(input_ids) + self.pos(torch.arange(T))[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h, _ = self.attn(h, h, h, attn_mask=causal)
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.out(h))


def brute_force_logp(model, tok, prompt, response, max_length):
    ids = tok.encode(prompt) + tok.encode(response)
    n_prompt = len(tok.encode(prompt))
    if tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    ids = ids[:max_length]
    x = torch.tensor([ids])
    logits = model(x).logits[0]
    logp = F.log_softmax(logits.float(), -1)
    total = 0.0
    for t in range(max(1, n_prompt), len(ids)):
        total += logp[t - 1, ids[t]].item()
    return total


def test_truncation_drops_eos_and_respects_max_length():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: ", "a: "], ["cdefgh", "x"], max_length=6)
    assert batch["input_ids"].shape[1] <= 6
    # row 0: 4 prompt + 6 response + eos = 11 -> truncated at 6, eos removed
    assert batch["labels"][0].tolist() == [-100] * 4 + tok.encode("cd")
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    # row 1 fits exactly: "a: " (3) + "x" + eos = 5
    assert batch["labels"][1, :5].tolist() == [-100] * 3 + tok.encode("x") + [tok.eos_token_id]
    assert batch["attention_mask"][1].tolist() == [1] * 5 + [0]


def test_prompt_longer_than_max_length_has_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    batch = adapter.build_sft_batch(tok, ["abcdefghij"], ["xyz"], max_length=5)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefghij"], ["xyz"], max_length=5)
    assert logps.item() == 0.0


def test_pad_equal_to_eos_keeps_eos_label_and_masks_padding():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Question: "], ["yes", "y"], max_length=32)
    row0 = batch["labels"][0]
    n = len(tok.encode("Q: ")) + 3 + 1
    assert row0[n - 1].item() == tok.eos_token_id
    assert row0[n:].eq(-100).all()
    assert batch["attention_mask"][0, :n].eq(1).all() and batch["attention_mask"][0, n:].eq(0).all()
    assert batch["input_ids"][0, n:].eq(1).all()


def test_missing_pad_token_falls_back_to_eos():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["b", "d"], max_length=16)
    assert batch["input_ids"][0, -1].item() == tok.eos_token_id
    assert batch["labels"][0, -1].item() == -100


def test_missing_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ok"], max_length=16)
    assert batch["labels"][0].ne(-100).sum().item() == 2


def test_logprobs_match_brute_force_and_preserve_row_order():
    tok = ToyTokenizer()
    model = TinyAttnLM(tok.vocab_size)
    prompts = ["What: ", "a: ", "Longer prompt here: ", "x"]
    responses = ["fine", "a much longer answer", "ok", "yy"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=24)
    want = torch.tensor([brute_force_logp(model, tok, p, r, 24) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    # reversing the input reverses the output
    rev = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], max_length=24)
    torch.testing.assert_close(rev, got.flip(0), rtol=1e-4, atol=1e-4)


def test_padding_does_not_change_logprobs():
    tok = ToyTokenizer()
    model = TinyAttnLM(tok.vocab_size)
    alone = adapter.compute_response_logprobs(model, tok, ["Q: "], ["hi"], max_length=32)
    mixed = adapter.compute_response_logprobs(model, tok, ["Q: ", "A much longer prompt: "], ["hi", "long reply"], max_length=32)
    torch.testing.assert_close(alone[0], mixed[0], rtol=1e-5, atol=1e-5)


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = TinyAttnLM(tok.vocab_size)
    prompts, responses = ["Do: ", "Say: "], ["a", "longer"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=16)
    n_tokens = (1 + 1) + (6 + 1)
    want = -sum(brute_force_logp(model, tok, p, r, 16) for p, r in zip(prompts, responses)) / n_tokens
    assert math.isclose(loss.item(), want, rel_tol=1e-4)


def test_dpo_loss_from_logps_matches_formula():
    pc = torch.tensor([-3.0, -10.0, -1.0])
    pr = torch.tensor([-4.0, -2.0, -1.5])
    rc = torch.tensor([-3.5, -9.0, -1.2])
    rr = torch.tensor([-3.5, -3.0, -1.4])
    beta = 0.3
    got = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    want = sum(
        -math.log(1 / (1 + math.exp(-beta * ((a - b) - (c - d)))))
        for a, b, c, d in zip(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist())
    ) / 3
    assert got.ndim == 0 and math.isclose(got.item(), want, rel_tol=1e-6)


def test_dpo_loss_is_stable_for_extreme_logits():
    big = torch.tensor([1e4])
    zero = torch.tensor([0.0])
    loss_good = adapter.compute_dpo_loss_from_logps(big, zero, zero, zero, beta=1.0)
    loss_bad = adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, beta=1.0)
    assert torch.isfinite(loss_good) and loss_good.item() < 1e-6
    assert torch.isfinite(loss_bad) and math.isclose(loss_bad.item(), 1e4, rel_tol=1e-6)


def test_dpo_loss_matches_manual_composition_and_reference_is_frozen():
    tok = ToyTokenizer()
    policy = TinyAttnLM(tok.vocab_size, seed=1)
    reference = TinyAttnLM(tok.vocab_size, seed=2)
    prompts, chosen, rejected = ["Pick: ", "Q: "], ["good one", "y"], ["bad", "no way"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.1, max_length=20)
    pc = adapter.compute_response_logprobs(policy, tok, prompts, chosen, 20)
    pr = adapter.compute_response_logprobs(policy, tok, prompts, rejected, 20)
    rc = adapter.compute_response_logprobs(reference, tok, prompts, chosen, 20)
    rr = adapter.compute_response_logprobs(reference, tok, prompts, rejected, 20)
    want = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.1)
    torch.testing.assert_close(loss, want, rtol=1e-5, atol=1e-6)
    loss.backward()
    assert all(p.grad is None for p in reference.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def test_identical_policy_and_reference_give_log2_and_zero_rewards():
    tok = ToyTokenizer()
    policy = TinyAttnLM(tok.vocab_size, seed=3)
    reference = TinyAttnLM(tok.vocab_size, seed=3)
    batch = adapter.build_dpo_batch(tok, ["P: "], ["aa"], ["bbb"], 16)
    loss, stats = core.dpo_loss_from_batch(policy, reference, batch, 0.1, core.get_pad_id(tok))
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert stats["accuracies"].sum().item() == 0  # strict r_w > r_l -> 0 at init
    assert stats["margins"].abs().max().item() < 1e-6


def test_small_dpo_step_increases_preference_gap():
    tok = ToyTokenizer()
    policy = TinyAttnLM(tok.vocab_size, seed=4)
    reference = TinyAttnLM(tok.vocab_size, seed=4)
    reference.requires_grad_(False)
    prompts, chosen, rejected = ["Q: "] * 2, ["yes", "sure"], ["no", "nope"]
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.5, max_length=16)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] and losses[0] == pytest.approx(math.log(2), rel=1e-5)


def test_dpo_batch_rows_align_between_chosen_and_rejected():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["A: ", "Longer: "], ["x", "yy"], ["zzz", "w"], max_length=32)
    for side, resps in (("chosen", ["x", "yy"]), ("rejected", ["zzz", "w"])):
        for row, (p, r) in enumerate(zip(["A: ", "Longer: "], resps)):
            n_p = len(tok.encode(p))
            lab = batch[side]["labels"][row]
            assert lab[:n_p].eq(-100).all()
            assert lab[n_p : n_p + len(r) + 1].tolist() == tok.encode(r) + [tok.eos_token_id]


def test_grad_accumulation_token_normalizer_equals_full_batch_token_mean():
    tok = ToyTokenizer()
    model = TinyAttnLM(tok.vocab_size, seed=5)
    prompts, responses = ["a: ", "bb: ", "c: ", "dddd: "], ["x", "long answer", "mid one", "z"]
    full = core.build_lm_batch(tok, prompts, responses, 32)
    loss_full, _ = core.sft_loss_from_batch(model, full)
    loss_full.backward()
    g_full = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    halves = [core.build_lm_batch(tok, prompts[:2], responses[:2], 32), core.build_lm_batch(tok, prompts[2:], responses[2:], 32)]
    total = sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in halves)
    for b in halves:
        loss, _ = core.sft_loss_from_batch(model, b, token_normalizer=total)
        loss.backward()
    for a, b in zip(g_full, [p.grad for p in model.parameters()]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
