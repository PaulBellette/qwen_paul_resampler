from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .corpus import read_corpus, sample_style_excerpts
from .modeling import generate_text, load_generator, load_scorer, style_delta_score


WEAK_SYSTEM = """You rewrite text while preserving its claims, intent, and technical content.
Use natural conversational English. Do not add facts, remove important qualifications, or mention the rewriting process.
Return only the rewritten text."""

NULL_SYSTEM = """You rewrite text in the writing style demonstrated by the supplied examples.
Preserve the source's claims, intent, and technical content. Do not add facts, remove important qualifications, or mention the rewriting process.
Return only the rewritten text."""


@dataclass
class Candidate:
    index: int
    text: str
    paul_logp: float
    base_logp: float
    style_delta: float
    style_ratio: float


def weak_messages(source: str):
    return [
        {"role": "system", "content": WEAK_SYSTEM},
        {"role": "user", "content": f"Rewrite this:\n\n{source}"},
    ]


def null_messages(source: str, excerpts: list[str]):
    examples = "\n\n--- EXAMPLE ---\n\n".join(excerpts)
    return [
        {"role": "system", "content": NULL_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here are examples of my writing:\n\n"
                f"{examples}\n\n"
                "--- SOURCE TO REWRITE ---\n\n"
                f"{source}\n\n"
                "Rewrite the source to sound like me."
            ),
        },
    ]


def run_rewrite(
    *,
    source: str,
    adapter_path: str,
    corpus_path: str,
    output_path: str,
    generator_model_name: str,
    base_model_name: str,
    n: int = 16,
    temperature: float = 0.95,
    top_p: float = 0.92,
    max_new_tokens: int = 500,
    score_max_length: int = 1024,
    seed: int = 1234,
):
    gen_tok, gen_model = load_generator(generator_model_name)
    scorer = load_scorer(base_model_name, adapter_path)

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for i in range(n):
        text = generate_text(
            gen_tok,
            gen_model,
            weak_messages(source),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed + i,
        )
        if text in seen:
            continue
        seen.add(text)
        scores = style_delta_score(scorer, text, max_length=score_max_length)
        candidates.append(Candidate(index=i, text=text, **scores))
        print(f"candidate {i:02d}: style_delta={scores['style_delta']:+.4f}")

    if not candidates:
        raise RuntimeError("No candidates were generated")
    ranked = sorted(candidates, key=lambda c: c.style_delta, reverse=True)

    docs = read_corpus(corpus_path)
    excerpts = sample_style_excerpts(docs, seed=seed)
    null_text = generate_text(
        gen_tok,
        gen_model,
        null_messages(source, excerpts),
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        seed=seed + 10_000,
    )
    null_scores = style_delta_score(scorer, null_text, max_length=score_max_length)
    source_scores = style_delta_score(scorer, source, max_length=score_max_length)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = []
    report.append("# Qwen personal-style resampling demo\n")
    report.append("## Source\n\n" + source.strip() + "\n")
    report.append(
        "## Winner: weak prompt + resampling\n\n"
        f"**Style delta:** `{ranked[0].style_delta:+.4f}` nats/token  "
        f"(**likelihood ratio:** `{ranked[0].style_ratio:.3f}x`)\n\n"
        + ranked[0].text
        + "\n"
    )
    report.append(
        "## Null: explicit ‘sound like me’ prompt\n\n"
        f"**Style delta:** `{null_scores['style_delta']:+.4f}` nats/token  "
        f"(**likelihood ratio:** `{null_scores['style_ratio']:.3f}x`)\n\n"
        + null_text
        + "\n"
    )
    report.append(
        "## Original source score\n\n"
        f"`{source_scores['style_delta']:+.4f}` nats/token\n"
    )
    report.append("## Ranked candidates\n")
    for rank, c in enumerate(ranked, 1):
        report.append(
            f"### {rank}. candidate {c.index}\n\n"
            f"Style delta: `{c.style_delta:+.4f}` nats/token; ratio `{c.style_ratio:.3f}x`\n\n"
            f"{c.text}\n"
        )

    out.write_text("\n".join(report), encoding="utf-8")

    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "source": source,
                "source_score": source_scores,
                "winner": asdict(ranked[0]),
                "null": {"text": null_text, **null_scores},
                "candidates": [asdict(c) for c in ranked],
                "config": {
                    "generator_model": generator_model_name,
                    "base_model": base_model_name,
                    "n_requested": n,
                    "n_unique": len(candidates),
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    print(f"Wrote {json_path}")
