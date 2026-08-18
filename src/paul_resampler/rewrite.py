from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .corpus import read_corpus, sample_style_excerpts
from .modeling import generate_text, load_generator, load_scorer, style_delta_score
from .semantic import DEFAULT_NLI_MODEL, LoadedNLI, SemanticCheck, load_nli, verify_candidate


COMMON_REWRITE_RULES = """Preserve every substantive claim, qualification, intent, causal direction, and level of certainty.
Do not add facts, anecdotes, examples, or opinions that are absent from the source.
Do not mention the rewriting process. Return only the rewritten text.
Do not imitate any named person or supplied writing sample."""

REALIZATION_MODES: tuple[tuple[str, str], ...] = (
    (
        "conversational",
        """Express the same ideas as natural conversational prose, as if explaining the point to an intelligent colleague.
Use contractions, varied rhythm, and occasional fragments where they fit. Do not preserve the source sentence structure.""",
    ),
    (
        "informal-discussion",
        """Rebuild the argument as an informal contribution to an ongoing discussion.
It can be compact, slightly uneven, and less polished than assistant prose, while keeping the same substance.""",
    ),
    (
        "from-memory",
        """Read the source for meaning, then reconstruct it from memory rather than paraphrasing sentence by sentence.
Keep the ideas and qualifications but deliberately choose fresh phrasing, ordering, and sentence boundaries.""",
    ),
    (
        "compressed-rebuild",
        """Compress the source to its conceptual skeleton, then expand that skeleton back into natural prose.
Avoid formulaic transitions and avoid copying the original register or syntax.""",
    ),
)

NULL_SYSTEM = """You rewrite text in the writing style demonstrated by the supplied examples.
Preserve the source's claims, intent, and technical content. Do not add facts, remove important qualifications, or mention the rewriting process.
Return only the rewritten text."""


@dataclass
class Candidate:
    index: int
    mode: str
    text: str
    paul_logp: float
    base_logp: float
    style_delta: float
    style_ratio: float
    semantic_pass: bool | None = None
    semantic_verdict: str = "UNCHECKED"
    semantic_raw: str = ""
    semantic_forward_entailment: float | None = None
    semantic_reverse_entailment: float | None = None
    semantic_source_coverage: float | None = None
    semantic_candidate_support: float | None = None


def realization_messages(source: str, mode_index: int):
    mode_name, mode_prompt = REALIZATION_MODES[mode_index % len(REALIZATION_MODES)]
    system = f"{mode_prompt}\n\n{COMMON_REWRITE_RULES}"
    return mode_name, [
        {"role": "system", "content": system},
        {"role": "user", "content": f"SOURCE:\n\n{source}\n\nWrite a fresh realization of the same content."},
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


def _check_candidate(
    nli: LoadedNLI,
    source: str,
    candidate: Candidate,
    *,
    entailment_threshold: float,
    max_contradiction: float,
    nli_max_length: int,
    nli_source_coverage: float,
    nli_candidate_support: float,
) -> SemanticCheck:
    check = verify_candidate(
        nli,
        source=source,
        candidate=candidate.text,
        entailment_threshold=entailment_threshold,
        max_contradiction=max_contradiction,
        required_source_coverage=nli_source_coverage,
        required_candidate_support=nli_candidate_support,
        max_length=nli_max_length,
    )
    candidate.semantic_pass = check.passed
    candidate.semantic_verdict = check.verdict
    candidate.semantic_raw = check.raw
    candidate.semantic_forward_entailment = check.forward.entailment
    candidate.semantic_reverse_entailment = check.reverse.entailment
    candidate.semantic_source_coverage = check.source_coverage
    candidate.semantic_candidate_support = check.candidate_support
    return check


def run_rewrite(
    *,
    source: str,
    adapter_path: str,
    corpus_path: str,
    output_path: str,
    generator_model_name: str,
    base_model_name: str,
    n: int = 32,
    temperature: float = 1.05,
    top_p: float = 0.95,
    max_new_tokens: int = 500,
    score_max_length: int = 1024,
    seed: int = 1234,
    semantic_gate: bool = True,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    nli_device: str = "auto",
    nli_entailment_threshold: float = 0.50,
    nli_max_contradiction: float = 0.20,
    nli_max_length: int = 512,
    nli_source_coverage: float = 1.0,
    nli_candidate_support: float = 1.0,
):
    gen_tok, gen_model = load_generator(generator_model_name)
    scorer = load_scorer(base_model_name, adapter_path)
    nli = load_nli(nli_model_name, device=nli_device) if semantic_gate else None

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for i in range(n):
        mode, messages = realization_messages(source, i)
        text = generate_text(
            gen_tok,
            gen_model,
            messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed + i,
        )
        if text in seen:
            continue
        seen.add(text)
        scores = style_delta_score(scorer, text, max_length=score_max_length)
        candidates.append(Candidate(index=i, mode=mode, text=text, **scores))
        print(f"candidate {i:02d} [{mode}]: style_delta={scores['style_delta']:+.4f}")

    if not candidates:
        raise RuntimeError("No candidates were generated")
    ranked = sorted(candidates, key=lambda c: c.style_delta, reverse=True)

    # Constrained optimization: style-rank first, then spend semantic checks only
    # until the highest-ranked feasible candidate is found.
    winner = ranked[0]
    winner_mode = "style-only"
    if semantic_gate:
        assert nli is not None
        winner_mode = "semantic-gated"
        passed = None
        for c in ranked:
            check = _check_candidate(
                nli,
                source,
                c,
                entailment_threshold=nli_entailment_threshold,
                max_contradiction=nli_max_contradiction,
                nli_max_length=nli_max_length,
                nli_source_coverage=nli_source_coverage,
                nli_candidate_support=nli_candidate_support,
            )
            print(
                f"semantic candidate {c.index:02d}: {check.verdict} "
                f"(source coverage={check.source_coverage:.2f}, candidate support={check.candidate_support:.2f})"
            )
            if check.passed:
                passed = c
                break
        if passed is not None:
            winner = passed
        else:
            winner_mode = "fallback-no-semantic-pass"
            print("WARNING: no candidate passed the semantic gate; falling back to top style score")

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
    null_check = None
    if semantic_gate:
        assert nli is not None
        null_check = verify_candidate(
            nli,
            source=source,
            candidate=null_text,
            entailment_threshold=nli_entailment_threshold,
            max_contradiction=nli_max_contradiction,
            required_source_coverage=nli_source_coverage,
            required_candidate_support=nli_candidate_support,
            max_length=nli_max_length,
        )
    source_scores = style_delta_score(scorer, source, max_length=score_max_length)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = []
    report.append("# Qwen personal-style resampling demo\n")
    report.append("## Source\n\n" + source.strip() + "\n")
    report.append(
        "## Candidate search\n\n"
        + "Realization modes: "
        + ", ".join(f"`{name}`" for name, _ in REALIZATION_MODES)
        + ". Candidates are distributed round-robin across modes; none are shown Paul examples.\n"
    )
    if semantic_gate:
        report.append(
            "## Semantic gate\n\n"
            f"Sentence/claim coverage NLI using `{nli_model_name}`. Each source sentence is checked "
            f"against the whole candidate, and each candidate sentence against the whole source. "
            f"Entailment threshold `{nli_entailment_threshold:.2f}`, contradiction ceiling "
            f"`{nli_max_contradiction:.2f}`, required source coverage `{nli_source_coverage:.2f}`, "
            f"candidate support `{nli_candidate_support:.2f}`.\n"
        )
    winner_sem = ""
    if semantic_gate:
        winner_sem = (
            f"  \n**Semantic gate:** `{winner.semantic_verdict}`"
            f" (source coverage `{winner.semantic_source_coverage:.2f}`, "
            f"candidate support `{winner.semantic_candidate_support:.2f}`; "
            f"whole cand→src `{winner.semantic_forward_entailment:.3f}`, "
            f"src→cand `{winner.semantic_reverse_entailment:.3f}`)"
        )
    report.append(
        f"## Winner: resampling ({winner_mode})\n\n"
        f"**Mode:** `{winner.mode}`  \n"
        f"**Style delta:** `{winner.style_delta:+.4f}` nats/token  "
        f"(**likelihood ratio:** `{winner.style_ratio:.3f}x`){winner_sem}\n\n"
        + winner.text
        + "\n"
    )
    null_sem = ""
    if null_check is not None:
        null_sem = (
            f"  \n**Semantic gate:** `{null_check.verdict}`"
            f" (source coverage `{null_check.source_coverage:.2f}`, "
            f"candidate support `{null_check.candidate_support:.2f}`; "
            f"whole cand→src `{null_check.forward.entailment:.3f}`, "
            f"src→cand `{null_check.reverse.entailment:.3f}`)"
        )
    report.append(
        "## Null: explicit ‘sound like me’ prompt\n\n"
        f"**Style delta:** `{null_scores['style_delta']:+.4f}` nats/token  "
        f"(**likelihood ratio:** `{null_scores['style_ratio']:.3f}x`){null_sem}\n\n"
        + null_text
        + "\n"
    )
    report.append(
        "## Original source score\n\n"
        f"`{source_scores['style_delta']:+.4f}` nats/token\n"
    )
    report.append("## Ranked candidates\n")
    for rank, c in enumerate(ranked, 1):
        sem = "not checked"
        if c.semantic_pass is True:
            sem = "PASS"
        elif c.semantic_pass is False:
            sem = "FAIL"
        report.append(
            f"### {rank}. candidate {c.index} · {c.mode}\n\n"
            f"Style delta: `{c.style_delta:+.4f}` nats/token; ratio `{c.style_ratio:.3f}x`; semantic: `{sem}`\n\n"
            f"{c.text}\n"
        )
        if c.semantic_raw:
            report.append("<details><summary>Semantic check</summary>\n\n```text\n" + c.semantic_raw + "\n```\n\n</details>\n")

    if null_check is not None:
        report.append("## Null semantic check\n\n```text\n" + null_check.raw + "\n```\n")

    out.write_text("\n".join(report), encoding="utf-8")

    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "source": source,
                "source_score": source_scores,
                "semantic_gate": {
                    "enabled": semantic_gate,
                    "model": nli_model_name if semantic_gate else None,
                    "entailment_threshold": nli_entailment_threshold,
                    "max_contradiction": nli_max_contradiction,
                    "max_length": nli_max_length,
                    "required_source_coverage": nli_source_coverage,
                    "required_candidate_support": nli_candidate_support,
                },
                "realization_modes": [name for name, _ in REALIZATION_MODES],
                "winner_mode": winner_mode,
                "winner": asdict(winner),
                "null": {
                    "text": null_text,
                    **null_scores,
                    "semantic": None if null_check is None else {
                        "passed": null_check.passed,
                        "verdict": null_check.verdict,
                        "forward": asdict(null_check.forward),
                        "reverse": asdict(null_check.reverse),
                        "source_coverage": null_check.source_coverage,
                        "candidate_support": null_check.candidate_support,
                        "source_claims": [asdict(x) for x in null_check.source_claims],
                        "candidate_claims": [asdict(x) for x in null_check.candidate_claims],
                    },
                },
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
