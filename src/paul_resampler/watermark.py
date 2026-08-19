from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch
from transformers import SynthIDTextWatermarkingConfig

from .modeling import generate_text, load_generator


# Public 30-key configuration from Google DeepMind's SynthID Text reference
# implementation. It is deliberately public: this is a reproducibility/robustness
# experiment, not a production provenance system.
DEFAULT_SYNTHID_KEYS: tuple[int, ...] = (
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
    590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
    118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
)


@dataclass(frozen=True)
class SynthIDConfig:
    keys: tuple[int, ...] = DEFAULT_SYNTHID_KEYS
    ngram_len: int = 5
    sampling_table_size: int = 2**16
    sampling_table_seed: int = 0
    context_history_size: int = 1024
    skip_first_ngram_calls: bool = False

    def hf_config(self) -> SynthIDTextWatermarkingConfig:
        return SynthIDTextWatermarkingConfig(
            keys=list(self.keys),
            ngram_len=self.ngram_len,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SynthIDConfig":
        return cls(
            keys=tuple(int(x) for x in data["keys"]),
            ngram_len=int(data["ngram_len"]),
            sampling_table_size=int(data.get("sampling_table_size", 2**16)),
            sampling_table_seed=int(data.get("sampling_table_seed", 0)),
            context_history_size=int(data.get("context_history_size", 1024)),
            skip_first_ngram_calls=bool(data.get("skip_first_ngram_calls", False)),
        )


@dataclass(frozen=True)
class SynthIDScore:
    mean: float | None
    weighted_mean: float | None
    tokens: int
    ngrams: int
    usable_ngrams: int


def parse_keys(value: str) -> tuple[int, ...]:
    keys = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not keys:
        raise ValueError("At least one SynthID key is required")
    return keys


def _processor(config: SynthIDConfig, *, vocab_size: int, device: torch.device):
    # Constructing via the public config method keeps generation and detection
    # on exactly the same Transformers implementation.
    return config.hf_config().construct_processor(vocab_size=vocab_size, device=device)


def _score_g_values(g_values: torch.Tensor, mask: torch.Tensor) -> tuple[float | None, float | None]:
    """Mean and Google-reference-style weighted-mean scores.

    The reference weighted detector uses linearly decreasing depth weights from
    10 to 1, normalized so the weights sum to the watermark depth.
    """
    if g_values.ndim != 3 or mask.ndim != 2:
        raise ValueError(f"Unexpected SynthID shapes: g={tuple(g_values.shape)}, mask={tuple(mask.shape)}")
    depth = g_values.shape[-1]
    usable = mask.float().sum()
    if usable.item() <= 0:
        return None, None

    mask3 = mask.float().unsqueeze(-1)
    mean = (g_values.float() * mask3).sum() / (depth * usable)

    weights = torch.linspace(10.0, 1.0, steps=depth, device=g_values.device)
    weights = weights * (depth / weights.sum())
    weighted = (g_values.float() * weights.view(1, 1, -1) * mask3).sum() / (depth * usable)
    return float(mean.item()), float(weighted.item())


def score_synthid_text(tokenizer, text: str, config: SynthIDConfig, *, device: str | torch.device = "cpu") -> SynthIDScore:
    target = torch.device(device)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(target)
    tokens = int(input_ids.shape[1])
    if tokens < config.ngram_len:
        return SynthIDScore(mean=None, weighted_mean=None, tokens=tokens, ngrams=0, usable_ngrams=0)

    processor = _processor(config, vocab_size=len(tokenizer), device=target)
    g_values = processor.compute_g_values(input_ids=input_ids)
    repetition_mask = processor.compute_context_repetition_mask(input_ids=input_ids).bool()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_mask = torch.ones_like(repetition_mask, dtype=torch.bool)
    else:
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]
        eos_mask_full = processor.compute_eos_token_mask(input_ids=input_ids, eos_token_id=int(eos_id)).bool()
        eos_mask = eos_mask_full[:, config.ngram_len - 1 :]

    mask = repetition_mask & eos_mask
    # If generation was configured to skip the initial watermark calls, those
    # positions cannot carry the watermark and should not contribute to score.
    if config.skip_first_ngram_calls and mask.shape[1] > 0:
        mask[:, : min(config.ngram_len - 1, mask.shape[1])] = False

    mean, weighted = _score_g_values(g_values, mask)
    return SynthIDScore(
        mean=mean,
        weighted_mean=weighted,
        tokens=tokens,
        ngrams=int(g_values.shape[1]),
        usable_ngrams=int(mask.sum().item()),
    )


def _load_prompt_records(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No prompts found in {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON prompt files must contain a list")
        rows = data
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    out: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            row = {"prompt": row}
        if not isinstance(row, dict) or not str(row.get("prompt", "")).strip():
            raise ValueError(f"Prompt record {i} must contain non-empty 'prompt'")
        out.append({"id": str(row.get("id", f"prompt_{i:03d}")), "prompt": str(row["prompt"]).strip()})
    return out


def _source_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the user's request directly and naturally. Give enough detail that the response is at least "
                "a few paragraphs when the topic supports it. Return only the answer."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _jsonable_config(config: SynthIDConfig) -> dict:
    data = asdict(config)
    data["keys"] = list(config.keys)
    return data


def run_watermark_generate(
    *,
    prompts_path: str,
    output_path: str,
    generator_model_name: str,
    synthid: SynthIDConfig,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 350,
    seed: int = 4242,
    limit: int | None = None,
    overwrite: bool = False,
):
    prompts = _load_prompt_records(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError("No prompt records selected")

    tokenizer, model = load_generator(generator_model_name)
    detector_device = next(model.parameters()).device
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists. Frozen watermark sources are not overwritten by default; "
            "choose a new path or pass --overwrite deliberately."
        )

    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(prompts):
            item_seed = seed + i * 100
            messages = _source_messages(row["prompt"])

            # Same seed for the plain/watermarked pair. The watermark modifies
            # the sampling distribution, so outputs can diverge while the
            # underlying RNG starting point is controlled.
            plain = generate_text(
                tokenizer,
                model,
                messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=item_seed,
            )
            watermarked = generate_text(
                tokenizer,
                model,
                messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=item_seed,
                watermarking_config=synthid.hf_config(),
            )

            plain_score = score_synthid_text(tokenizer, plain, synthid, device=detector_device)
            wm_score = score_synthid_text(tokenizer, watermarked, synthid, device=detector_device)
            record = {
                "schema_version": 1,
                "id": row["id"],
                "prompt": row["prompt"],
                "generator_model": generator_model_name,
                "seed": item_seed,
                "generation": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                },
                "synthid": _jsonable_config(synthid),
                "plain": {"text": plain, "detector": asdict(plain_score)},
                "watermarked": {"text": watermarked, "detector": asdict(wm_score)},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{row['id']}: plain={plain_score.weighted_mean!s} "
                f"watermarked={wm_score.weighted_mean!s}"
            )

    print(f"Wrote frozen watermark sources to {out}")
    print("Keep this JSONL fixed while iterating on the downstream resampler.")
