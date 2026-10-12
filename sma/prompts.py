"""The single prompt template used for SFT, both DPO runs, and evaluation.

We use Qwen's native ChatML layout with Qwen's default system message. The
response is followed by the tokenizer EOS (``<|endoftext|>`` for Qwen2.5);
generation stops on either ``<|endoftext|>`` or ``<|im_end|>``.

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|endoftext|>
"""

from __future__ import annotations

DEFAULT_SYSTEM = "You are a helpful assistant."

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def format_prompt(instruction: str, system: str = DEFAULT_SYSTEM) -> str:
    return CHATML_TEMPLATE.format(system=system, instruction=instruction.strip())


def stop_token_ids(tokenizer):
    ids = []
    for token in STOP_STRINGS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            ids.append(int(token_id))
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def format_with_chat_template(tokenizer, instruction: str, system: str | None = None) -> str:
    """Prompt string from the tokenizer's own chat template (used for SmolLM2)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
