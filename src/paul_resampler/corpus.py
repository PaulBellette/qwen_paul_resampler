from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random

TEXT_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class CorpusDocument:
    path: str
    name: str
    text: str


def read_corpus_documents(root: str | Path) -> list[CorpusDocument]:
    root = Path(root)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        )

    docs: list[CorpusDocument] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            try:
                name = str(path.relative_to(root)) if root.is_dir() else path.name
            except ValueError:
                name = path.name
            docs.append(CorpusDocument(path=str(path), name=name, text=text))
    if not docs:
        raise ValueError(f"No .txt/.md text found under {root}")
    return docs


def read_corpus(root: str | Path) -> list[str]:
    return [doc.text for doc in read_corpus_documents(root)]


def token_chunks(
    tokenizer,
    text: str,
    *,
    target_tokens: int = 200,
) -> list[tuple[str, int]]:
    """Split a document into deterministic, non-overlapping token chunks.

    We balance chunk sizes rather than taking fixed windows so a document that
    is only just over the target does not leave a tiny tail. Every source token
    belongs to exactly one chunk, and no chunk is duplicated to satisfy a
    requested sample count.
    """
    if target_tokens < 2:
        raise ValueError("target_tokens must be >= 2")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    n_chunks = max(1, math.ceil(len(token_ids) / target_tokens))
    base = len(token_ids) // n_chunks
    remainder = len(token_ids) % n_chunks

    chunks: list[tuple[str, int]] = []
    cursor = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        ids = token_ids[cursor : cursor + size]
        cursor += size
        chunk_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append((chunk_text, len(ids)))
    return chunks


def sample_style_excerpts(
    docs: list[str],
    *,
    count: int = 8,
    chars_per_excerpt: int = 600,
    seed: int = 0,
) -> list[str]:
    """Sample corpus excerpts for the prompting-null condition.

    Sampling from arbitrary offsets avoids always feeding document intros,
    which are often less representative of conversational style.
    """
    rng = random.Random(seed)
    excerpts: list[str] = []
    pool = [d for d in docs if len(d.strip()) >= 80]
    if not pool:
        pool = docs

    for _ in range(count):
        doc = rng.choice(pool)
        if len(doc) <= chars_per_excerpt:
            excerpt = doc
        else:
            start = rng.randint(0, len(doc) - chars_per_excerpt)
            excerpt = doc[start : start + chars_per_excerpt]
            # Nudge inward to word boundaries where practical.
            if start > 0 and " " in excerpt[:80]:
                excerpt = excerpt.split(" ", 1)[1]
            if start + chars_per_excerpt < len(doc) and " " in excerpt[-80:]:
                excerpt = excerpt.rsplit(" ", 1)[0]
        excerpts.append(excerpt.strip())
    return excerpts
