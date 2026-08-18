from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from .corpus import CorpusDocument, read_corpus_documents, token_chunks
from .modeling import load_scorer, style_delta_score


@dataclass
class CalibrationChunk:
    label: str
    document_index: int
    document: str
    chunk_index: int
    token_count: int
    text: str


@dataclass
class ScoredChunk:
    label: str
    document_index: int
    document: str
    chunk_index: int
    token_count: int
    style_delta: float
    style_ratio: float
    text: str


def _text_key(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_unique_chunks(
    tokenizer,
    docs: list[CorpusDocument],
    *,
    label: str,
    target_tokens: int,
) -> tuple[list[CalibrationChunk], int]:
    """Create unique, non-overlapping chunks without replacement.

    Returns (chunks, duplicates_removed). Duplicates are detected after token
    decoding and whitespace normalization within a group.
    """
    out: list[CalibrationChunk] = []
    seen: set[str] = set()
    duplicates = 0
    for doc_index, doc in enumerate(docs):
        for chunk_index, (text, token_count) in enumerate(
            token_chunks(tokenizer, doc.text, target_tokens=target_tokens)
        ):
            key = _text_key(text)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            out.append(
                CalibrationChunk(
                    label=label,
                    document_index=doc_index,
                    document=doc.name,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    text=text,
                )
            )
    return out, duplicates


def limit_chunks(
    chunks: list[CalibrationChunk],
    *,
    max_chunks: int | None,
    seed: int,
) -> list[CalibrationChunk]:
    if max_chunks is None or max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    rng = random.Random(seed)
    picked = rng.sample(range(len(chunks)), max_chunks)
    return [chunks[i] for i in sorted(picked)]


def auc_from_scores(positive: list[float], negative: list[float]) -> float:
    # Probability that a random positive scores above a random negative;
    # ties count half. Equivalent to ROC AUC for a scalar score.
    wins = 0.0
    total = len(positive) * len(negative)
    if total == 0:
        return float("nan")
    for p in positive:
        for n in negative:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / total


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = q * (len(ordered) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "q10": quantile(0.10),
        "q90": quantile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def document_means(rows: list[ScoredChunk]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.label, row.document_index, row.document)].append(row.style_delta)

    out: list[dict] = []
    for (label, document_index, document), values in sorted(grouped.items()):
        out.append(
            {
                "label": label,
                "document_index": document_index,
                "document": document,
                "chunks": len(values),
                "mean_style_delta": statistics.fmean(values),
                "min_style_delta": min(values),
                "max_style_delta": max(values),
            }
        )
    return out


def _fmt_auc(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "not reported"
    return f"`{value:.3f}`"


def run_calibration(
    *,
    paul_corpus: str,
    generic_corpus: str,
    adapter_path: str,
    base_model_name: str,
    output_path: str,
    chunk_tokens: int = 200,
    max_chunks_per_group: int | None = None,
    score_max_length: int = 1024,
    seed: int = 2026,
):
    paul_docs = read_corpus_documents(paul_corpus)
    generic_docs = read_corpus_documents(generic_corpus)
    scorer = load_scorer(base_model_name, adapter_path)

    paul_chunks, paul_dupes = build_unique_chunks(
        scorer.tokenizer,
        paul_docs,
        label="paul",
        target_tokens=chunk_tokens,
    )
    generic_chunks, generic_dupes = build_unique_chunks(
        scorer.tokenizer,
        generic_docs,
        label="generic",
        target_tokens=chunk_tokens,
    )

    paul_chunks = limit_chunks(
        paul_chunks,
        max_chunks=max_chunks_per_group,
        seed=seed,
    )
    generic_chunks = limit_chunks(
        generic_chunks,
        max_chunks=max_chunks_per_group,
        seed=seed + 1,
    )

    if not paul_chunks or not generic_chunks:
        raise ValueError("Both calibration groups must produce at least one non-empty chunk")

    # Detect exact cross-group contamination. We report it rather than silently
    # dropping evidence, because identical text in both groups is itself useful
    # information about the corpus construction.
    paul_keys = {_text_key(c.text) for c in paul_chunks}
    generic_keys = {_text_key(c.text) for c in generic_chunks}
    cross_group_duplicates = len(paul_keys & generic_keys)

    rows: list[ScoredChunk] = []
    for chunk in [*paul_chunks, *generic_chunks]:
        score = style_delta_score(scorer, chunk.text, max_length=score_max_length)
        rows.append(
            ScoredChunk(
                label=chunk.label,
                document_index=chunk.document_index,
                document=chunk.document,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                style_delta=score["style_delta"],
                style_ratio=score["style_ratio"],
                text=chunk.text,
            )
        )
        print(
            f"{chunk.label:7s} doc={chunk.document_index:02d} chunk={chunk.chunk_index:02d} "
            f"tokens={chunk.token_count:4d}: style_delta={score['style_delta']:+.4f}"
        )

    paul_scores = [r.style_delta for r in rows if r.label == "paul"]
    generic_scores = [r.style_delta for r in rows if r.label == "generic"]
    chunk_auc = auc_from_scores(paul_scores, generic_scores)
    separation_margin = min(paul_scores) - max(generic_scores)

    doc_rows = document_means(rows)
    paul_doc_scores = [d["mean_style_delta"] for d in doc_rows if d["label"] == "paul"]
    generic_doc_scores = [d["mean_style_delta"] for d in doc_rows if d["label"] == "generic"]
    # With one document per author/group, a document AUC is mathematically
    # computable but scientifically vacuous. Require at least two per group.
    document_auc = None
    if len(paul_doc_scores) >= 2 and len(generic_doc_scores) >= 2:
        document_auc = auc_from_scores(paul_doc_scores, generic_doc_scores)

    stats = {
        "paul": describe(paul_scores),
        "generic": describe(generic_scores),
        "chunk_auc": chunk_auc,
        "document_auc": document_auc,
        "separation_margin": separation_margin,
    }

    group_meta = {
        "paul": {
            "documents": len(paul_docs),
            "chunks": len(paul_chunks),
            "unique_chunks": len({_text_key(c.text) for c in paul_chunks}),
            "duplicates_removed": paul_dupes,
        },
        "generic": {
            "documents": len(generic_docs),
            "chunks": len(generic_chunks),
            "unique_chunks": len({_text_key(c.text) for c in generic_chunks}),
            "duplicates_removed": generic_dupes,
        },
        "cross_group_duplicate_chunks": cross_group_duplicates,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# Paul-style scorer calibration\n",
        "> Use a Paul corpus that was **not used to train the adapter** if you want this to mean more than an overfitting sanity check.\n",
        "## Corpus accounting\n",
        "| Group | Documents | Chunks scored | Unique chunks | Duplicates removed |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Held-out Paul | {group_meta['paul']['documents']} | {group_meta['paul']['chunks']} | "
            f"{group_meta['paul']['unique_chunks']} | {group_meta['paul']['duplicates_removed']} |"
        ),
        (
            f"| Generic | {group_meta['generic']['documents']} | {group_meta['generic']['chunks']} | "
            f"{group_meta['generic']['unique_chunks']} | {group_meta['generic']['duplicates_removed']} |"
        ),
        "",
        f"Chunks are deterministic, non-overlapping token partitions targeting about **{chunk_tokens} tokens** each. No chunk is duplicated to reach a sample count.",
        "",
    ]

    if cross_group_duplicates:
        report.extend(
            [
                f"> **Warning:** {cross_group_duplicates} exact normalized chunk(s) occur in both groups. Treat this as corpus contamination.",
                "",
            ]
        )

    report.extend(
        [
            "## Chunk-level summary\n",
            "| Group | Mean delta | Median | Stdev | 10–90% | Min–max |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| Held-out Paul | {stats['paul']['mean']:+.4f} | {stats['paul']['median']:+.4f} | "
                f"{stats['paul']['stdev']:.4f} | {stats['paul']['q10']:+.4f} to {stats['paul']['q90']:+.4f} | "
                f"{stats['paul']['min']:+.4f} to {stats['paul']['max']:+.4f} |"
            ),
            (
                f"| Generic | {stats['generic']['mean']:+.4f} | {stats['generic']['median']:+.4f} | "
                f"{stats['generic']['stdev']:.4f} | {stats['generic']['q10']:+.4f} to {stats['generic']['q90']:+.4f} | "
                f"{stats['generic']['min']:+.4f} to {stats['generic']['max']:+.4f} |"
            ),
            "",
            f"**Chunk-level pairwise AUC:** `{chunk_auc:.3f}`",
            "",
            f"**Separation margin (min Paul − max generic):** `{separation_margin:+.4f}` nats/token",
            "",
            "The chunk AUC is an ordering statistic, not an uncertainty estimate: chunks from the same document are correlated and are **not independent authors/documents**.",
            "",
            "## Document-level summary\n",
            "| Label | Document | Chunks | Mean delta | Min–max |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for d in doc_rows:
        report.append(
            f"| {d['label']} | `{d['document']}` | {d['chunks']} | {d['mean_style_delta']:+.4f} | "
            f"{d['min_style_delta']:+.4f} to {d['max_style_delta']:+.4f} |"
        )

    report.extend(["", f"**Document-level pairwise AUC:** {_fmt_auc(document_auc)}", ""])
    if document_auc is None:
        report.extend(
            [
                "Document-level AUC is only reported when there are at least **2 documents in each group**. With one Paul post and one generic post, the document means are descriptive only.",
                "",
            ]
        )
    else:
        report.extend(
            [
                "Document-level AUC compares per-document mean style scores. It is usually the more honest unit when evaluating author discrimination.",
                "",
            ]
        )

    report.extend(
        [
            "## Raw chunk scores\n",
            "| Label | Document | Chunk | Tokens | Style delta | Ratio |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        report.append(
            f"| {row.label} | `{row.document}` | {row.chunk_index} | {row.token_count} | "
            f"{row.style_delta:+.4f} | {row.style_ratio:.3f}x |"
        )

    out.write_text("\n".join(report) + "\n", encoding="utf-8")

    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "summary": stats,
                "corpus_accounting": group_meta,
                "document_scores": doc_rows,
                "config": {
                    "paul_corpus": paul_corpus,
                    "generic_corpus": generic_corpus,
                    "chunk_tokens": chunk_tokens,
                    "max_chunks_per_group": max_chunks_per_group,
                    "score_max_length": score_max_length,
                    "seed": seed,
                },
                "chunks": [asdict(r) for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.json')}")
