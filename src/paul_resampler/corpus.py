from __future__ import annotations

from pathlib import Path
import random

TEXT_SUFFIXES = {".txt", ".md"}


def read_corpus(root: str | Path) -> list[str]:
    root = Path(root)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
        )

    docs: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            docs.append(text)
    if not docs:
        raise ValueError(f"No .txt/.md text found under {root}")
    return docs


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
