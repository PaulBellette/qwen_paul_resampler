from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean

from .modeling import generate_text, load_generator, load_scorer, style_delta_score
from .rewrite import resample_with_loaded_models
from .semantic import DEFAULT_NLI_MODEL, load_nli, verify_candidate
from .watermark import SynthIDConfig, score_synthid_text


GENERIC_PARAPHRASE_SYSTEM = """Rewrite the source as a fresh, natural paraphrase.
Preserve every substantive claim, qualification, causal direction, and level of certainty.
Use different wording and sentence structure, but do not imitate any person or writing sample.
Do not add examples, anecdotes, or facts. Return only the rewritten text."""


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("schema_version", 0)) != 1:
                raise ValueError(f"Unsupported watermark source schema at line {line_no}")
            if "watermarked" not in row or "plain" not in row or "synthid" not in row:
                raise ValueError(f"Incomplete watermark source record at line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError("No watermark source records found")
    return rows


def _generic_messages(source: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GENERIC_PARAPHRASE_SYSTEM},
        {"role": "user", "content": f"SOURCE:\n\n{source}\n\nWrite the paraphrase."},
    ]


def _semantic_summary(check) -> dict:
    return {
        "passed": check.passed,
        "verdict": check.verdict,
        "source_coverage": check.source_coverage,
        "candidate_support": check.candidate_support,
        "whole_candidate_to_source": check.forward.entailment,
        "whole_source_to_candidate": check.reverse.entailment,
        "raw": check.raw,
    }


def _generic_control(
    *,
    source: str,
    gen_tok,
    gen_model,
    nli,
    attempts: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    entailment_threshold: float,
    max_contradiction: float,
    nli_max_length: int,
    source_coverage: float,
    candidate_support: float,
):
    tried = []
    for i in range(attempts):
        text = generate_text(
            gen_tok,
            gen_model,
            _generic_messages(source),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed + i,
        )
        check = verify_candidate(
            nli,
            source=source,
            candidate=text,
            entailment_threshold=entailment_threshold,
            max_contradiction=max_contradiction,
            required_source_coverage=source_coverage,
            required_candidate_support=candidate_support,
            max_length=nli_max_length,
        )
        tried.append((text, check))
        if check.passed:
            return text, check, i + 1

    # If none passes, use the least-bad candidate by symmetric coverage. The
    # report keeps the FAIL visible rather than silently dropping the control.
    text, check = max(tried, key=lambda pair: min(pair[1].source_coverage, pair[1].candidate_support))
    return text, check, attempts


def _avg(rows: list[dict], getter) -> float | None:
    vals = [getter(r) for r in rows]
    vals = [float(v) for v in vals if v is not None]
    return mean(vals) if vals else None


def _fmt(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def run_watermark_test(
    *,
    inputs_path: str,
    adapter_path: str,
    output_path: str,
    generator_model_name: str,
    base_model_name: str,
    n: int = 16,
    temperature: float = 1.05,
    top_p: float = 0.95,
    max_new_tokens: int = 500,
    score_max_length: int = 1024,
    seed: int = 9000,
    generic_attempts: int = 4,
    generic_temperature: float = 0.9,
    generic_top_p: float = 0.95,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    nli_device: str = "auto",
    nli_entailment_threshold: float = 0.50,
    nli_max_contradiction: float = 0.20,
    nli_max_length: int = 512,
    nli_source_coverage: float = 1.0,
    nli_candidate_support: float = 1.0,
    limit: int | None = None,
):
    input_path = Path(inputs_path)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_rows = _load_jsonl(input_path)
    if limit is not None:
        source_rows = source_rows[:limit]

    gen_tok, gen_model = load_generator(generator_model_name)
    scorer = load_scorer(base_model_name, adapter_path)
    nli = load_nli(nli_model_name, device=nli_device)
    detector_device = next(gen_model.parameters()).device

    results: list[dict] = []
    for idx, row in enumerate(source_rows):
        item_id = row["id"]
        source = row["watermarked"]["text"]
        synthid = SynthIDConfig.from_dict(row["synthid"])
        item_seed = seed + idx * 10_000

        # Detector measurement happens outside the resampler. Only `source`
        # crosses the boundary into resample_with_loaded_models().
        plain_detector = score_synthid_text(gen_tok, row["plain"]["text"], synthid, device=detector_device)
        before_detector = score_synthid_text(gen_tok, source, synthid, device=detector_device)
        before_style = style_delta_score(scorer, source, max_length=score_max_length)

        generic_text, generic_sem, generic_used = _generic_control(
            source=source,
            gen_tok=gen_tok,
            gen_model=gen_model,
            nli=nli,
            attempts=generic_attempts,
            temperature=generic_temperature,
            top_p=generic_top_p,
            max_new_tokens=max_new_tokens,
            seed=item_seed + 5000,
            entailment_threshold=nli_entailment_threshold,
            max_contradiction=nli_max_contradiction,
            nli_max_length=nli_max_length,
            source_coverage=nli_source_coverage,
            candidate_support=nli_candidate_support,
        )
        generic_detector = score_synthid_text(gen_tok, generic_text, synthid, device=detector_device)
        generic_style = style_delta_score(scorer, generic_text, max_length=score_max_length)

        resampled = resample_with_loaded_models(
            source=source,
            gen_tok=gen_tok,
            gen_model=gen_model,
            scorer=scorer,
            nli=nli,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_max_length=score_max_length,
            seed=item_seed,
            semantic_gate=True,
            nli_entailment_threshold=nli_entailment_threshold,
            nli_max_contradiction=nli_max_contradiction,
            nli_max_length=nli_max_length,
            nli_source_coverage=nli_source_coverage,
            nli_candidate_support=nli_candidate_support,
        )
        personal = resampled.winner
        personal_detector = score_synthid_text(gen_tok, personal.text, synthid, device=detector_device)

        results.append(
            {
                "id": item_id,
                "prompt": row["prompt"],
                "synthid": row["synthid"],
                "plain": {
                    "text": row["plain"]["text"],
                    "detector": asdict(plain_detector),
                },
                "watermarked_before": {
                    "text": source,
                    "detector": asdict(before_detector),
                    "style": before_style,
                },
                "generic_paraphrase": {
                    "text": generic_text,
                    "attempts_used": generic_used,
                    "detector": asdict(generic_detector),
                    "style": generic_style,
                    "semantic": _semantic_summary(generic_sem),
                },
                "personal_resample": {
                    "text": personal.text,
                    "winner_mode": resampled.winner_mode,
                    "realization_mode": personal.mode,
                    "detector": asdict(personal_detector),
                    "style": {
                        "paul_logp": personal.paul_logp,
                        "base_logp": personal.base_logp,
                        "style_delta": personal.style_delta,
                        "style_ratio": personal.style_ratio,
                    },
                    "semantic": {
                        "passed": personal.semantic_pass,
                        "verdict": personal.semantic_verdict,
                        "source_coverage": personal.semantic_source_coverage,
                        "candidate_support": personal.semantic_candidate_support,
                        "whole_candidate_to_source": personal.semantic_forward_entailment,
                        "whole_source_to_candidate": personal.semantic_reverse_entailment,
                        "raw": personal.semantic_raw,
                    },
                    "ranked_candidates": [asdict(c) for c in resampled.ranked],
                },
            }
        )
        print(
            f"{item_id}: SynthID weighted mean "
            f"before={_fmt(before_detector.weighted_mean)} "
            f"generic={_fmt(generic_detector.weighted_mean)} "
            f"personal={_fmt(personal_detector.weighted_mean)}"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "watermark_blind_transformation": True,
                "frozen_inputs": str(input_path),
                "frozen_inputs_sha256": input_sha256,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    avg_plain = _avg(results, lambda r: r["plain"]["detector"]["weighted_mean"])
    avg_before = _avg(results, lambda r: r["watermarked_before"]["detector"]["weighted_mean"])
    avg_generic = _avg(results, lambda r: r["generic_paraphrase"]["detector"]["weighted_mean"])
    avg_personal = _avg(results, lambda r: r["personal_resample"]["detector"]["weighted_mean"])
    avg_style_before = _avg(results, lambda r: r["watermarked_before"]["style"]["style_delta"])
    avg_style_generic = _avg(results, lambda r: r["generic_paraphrase"]["style"]["style_delta"])
    avg_style_personal = _avg(results, lambda r: r["personal_resample"]["style"]["style_delta"])
    generic_passes = sum(bool(r["generic_paraphrase"]["semantic"]["passed"]) for r in results)
    personal_passes = sum(bool(r["personal_resample"]["semantic"]["passed"]) for r in results)

    md = [
        "# Watermark robustness under generic paraphrase and personal resampling\n",
        "This experiment is deliberately **watermark-blind during transformation**. The SynthID key and detector score are not passed to either the generic paraphraser or the Paul resampler; detection is measured only before/after.\n",
        f"Frozen input SHA-256: `{input_sha256}`\n",
        "The detector statistic is the simple SynthID weighted-mean g-value score (linearly decreasing depth weights 10→1), not a trained Bayesian detector and not a calibrated yes/no threshold. Compare distributions/changes rather than treating one score as definitive attribution.\n",
        "## Aggregate\n",
        "| Condition | Mean SynthID weighted mean | Mean Paul style Δ | Semantic passes |",
        "|---|---:|---:|---:|",
        f"| Plain generation | {_fmt(avg_plain)} | — | — |",
        f"| Watermarked, before transform | {_fmt(avg_before)} | {_fmt(avg_style_before)} | — |",
        f"| Generic paraphrase | {_fmt(avg_generic)} | {_fmt(avg_style_generic)} | {generic_passes}/{len(results)} |",
        f"| Personal resampling | {_fmt(avg_personal)} | {_fmt(avg_style_personal)} | {personal_passes}/{len(results)} |",
        "\n## Per item\n",
        "| ID | Plain WM | Before WM | Generic WM | Personal WM | Generic semantic | Personal semantic | Personal style Δ |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in results:
        md.append(
            f"| `{r['id']}` | {_fmt(r['plain']['detector']['weighted_mean'])} | "
            f"{_fmt(r['watermarked_before']['detector']['weighted_mean'])} | "
            f"{_fmt(r['generic_paraphrase']['detector']['weighted_mean'])} | "
            f"{_fmt(r['personal_resample']['detector']['weighted_mean'])} | "
            f"{r['generic_paraphrase']['semantic']['verdict']} | "
            f"{r['personal_resample']['semantic']['verdict']} | "
            f"{r['personal_resample']['style']['style_delta']:+.4f} |"
        )

    md.append("\n## Details\n")
    for r in results:
        md.extend(
            [
                f"### {r['id']}\n",
                f"**Prompt:** {r['prompt']}\n",
                f"**Watermarked before:** `{_fmt(r['watermarked_before']['detector']['weighted_mean'])}`  ",
                f"**Generic after:** `{_fmt(r['generic_paraphrase']['detector']['weighted_mean'])}` — semantic `{r['generic_paraphrase']['semantic']['verdict']}`  ",
                f"**Personal after:** `{_fmt(r['personal_resample']['detector']['weighted_mean'])}` — semantic `{r['personal_resample']['semantic']['verdict']}`, style Δ `{r['personal_resample']['style']['style_delta']:+.4f}`\n",
                "<details><summary>Watermarked source</summary>\n\n" + r["watermarked_before"]["text"] + "\n\n</details>\n",
                "<details><summary>Generic paraphrase</summary>\n\n" + r["generic_paraphrase"]["text"] + "\n\n</details>\n",
                "<details><summary>Personal resample</summary>\n\n" + r["personal_resample"]["text"] + "\n\n</details>\n",
            ]
        )

    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {json_path}")
