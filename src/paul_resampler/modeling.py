from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedScorer:
    tokenizer: object
    model: PeftModel
    device: torch.device


def best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_generator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=best_dtype(),
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def load_scorer(base_model_name: str, adapter_path: str) -> LoadedScorer:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=best_dtype(),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    device = next(model.parameters()).device
    return LoadedScorer(tokenizer=tokenizer, model=model, device=device)


def mean_logprob(model, tokenizer, text: str, *, max_length: int = 1024) -> float:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    if input_ids.shape[1] < 2:
        return float("-inf")

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask[:, 1:].bool()
            token_logp = token_logp[mask]
        else:
            token_logp = token_logp.reshape(-1)
    return token_logp.mean().item()


def style_delta_score(scorer: LoadedScorer, text: str, *, max_length: int = 1024) -> dict[str, float]:
    """Return log p_adapter(text) - log p_base(text), averaged per predicted token."""
    paul_logp = mean_logprob(scorer.model, scorer.tokenizer, text, max_length=max_length)
    # PEFT exposes the untuned base by temporarily disabling adapters.
    with scorer.model.disable_adapter():
        base_logp = mean_logprob(scorer.model, scorer.tokenizer, text, max_length=max_length)
    return {
        "paul_logp": paul_logp,
        "base_logp": base_logp,
        "style_delta": paul_logp - base_logp,
        "style_ratio": math.exp(min(20.0, max(-20.0, paul_logp - base_logp))),
    }


def apply_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    # Qwen3 supports enable_thinking=False. The fallback keeps the demo usable
    # if a tokenizer revision does not expose that keyword.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_text(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.95,
    top_p: float = 0.92,
    max_new_tokens: int = 500,
    seed: int | None = None,
    do_sample: bool = True,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    prompt = apply_chat(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
