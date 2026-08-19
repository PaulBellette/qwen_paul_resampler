#!/usr/bin/env python3
"""Render social-media visuals from a watermark_benchmark.json artifact.

The script deliberately consumes only recorded experiment outputs. It does not
rerun generation, style scoring, NLI, or watermark detection.

Typical usage:

    uv sync --extra media
    uv run python scripts/render_demo.py \
        results/poc_v1/watermark_benchmark.json \
        --out-dir media

Outputs:
    media/demo_card.png   static 16:9 summary card
    media/demo.gif        short looping animation
    media/demo.mp4        MP4 when ffmpeg is available

Use --item ID to pin the candidate-cloud example. Otherwise the script chooses
an item that passed both semantic gates and showed strong watermark attenuation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import json
import math
from pathlib import Path
import re
import shutil
import textwrap
from typing import Any, Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


BG = "#0d1117"
PANEL = "#151b23"
TEXT = "#f0f3f6"
MUTED = "#9ba7b4"
GRID = "#2c3642"
ACCENT = "#4cc9f0"
ACCENT_2 = "#80ed99"
WARN = "#ffb86b"
FAIL = "#ff6b6b"
PLAIN = "#788391"
WM = "#f7c948"
GENERIC = "#a78bfa"
PERSONAL = "#4cc9f0"


@dataclass(frozen=True)
class DemoItem:
    raw: dict[str, Any]

    @property
    def item_id(self) -> str:
        return str(self.raw.get("id", "item"))

    @property
    def source(self) -> str:
        return str(self.raw["watermarked_before"]["text"])

    @property
    def personal_text(self) -> str:
        return str(self.raw["personal_resample"]["text"])

    @property
    def generic_text(self) -> str:
        return str(self.raw["generic_paraphrase"]["text"])

    @property
    def personal_pass(self) -> bool:
        return bool(self.raw["personal_resample"]["semantic"]["passed"])

    @property
    def generic_pass(self) -> bool:
        return bool(self.raw["generic_paraphrase"]["semantic"]["passed"])

    @property
    def plain_wm(self) -> float:
        return float(self.raw["plain"]["detector"]["weighted_mean"])

    @property
    def before_wm(self) -> float:
        return float(self.raw["watermarked_before"]["detector"]["weighted_mean"])

    @property
    def generic_wm(self) -> float:
        return float(self.raw["generic_paraphrase"]["detector"]["weighted_mean"])

    @property
    def personal_wm(self) -> float:
        return float(self.raw["personal_resample"]["detector"]["weighted_mean"])

    @property
    def personal_style(self) -> float:
        return float(self.raw["personal_resample"]["style"]["style_delta"])

    @property
    def candidates(self) -> list[dict[str, Any]]:
        return list(self.raw["personal_resample"].get("ranked_candidates", []))


# ---------- Data helpers ----------


def load_benchmark(path: Path) -> tuple[dict[str, Any], list[DemoItem]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected watermark benchmark JSON with a non-empty 'results' list")
    return data, [DemoItem(r) for r in rows]


def semantic_subset(items: Iterable[DemoItem]) -> list[DemoItem]:
    return [x for x in items if x.personal_pass and x.generic_pass]


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def aggregate_wm(items: list[DemoItem]) -> dict[str, float]:
    return {
        "plain": mean(x.plain_wm for x in items),
        "before": mean(x.before_wm for x in items),
        "generic": mean(x.generic_wm for x in items),
        "personal": mean(x.personal_wm for x in items),
    }


def retained_fraction(plain: float, before: float, after: float) -> float:
    lift = before - plain
    if abs(lift) < 1e-12:
        return float("nan")
    return (after - plain) / lift


def choose_item(items: list[DemoItem], requested: str | None) -> DemoItem:
    if requested:
        for item in items:
            if item.item_id == requested:
                return item
        choices = ", ".join(x.item_id for x in items)
        raise ValueError(f"Unknown --item {requested!r}; choices: {choices}")

    passing = semantic_subset(items)
    pool = passing or items

    # Prefer an item with candidate data and a strong, watermark-blind drop.
    def key(x: DemoItem) -> tuple[int, float, float]:
        has_candidates = int(bool(x.candidates))
        attenuation = x.before_wm - x.personal_wm
        style_gain = x.personal_style
        return (has_candidates, attenuation, style_gain)

    return max(pool, key=key)


def _normalise_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def surface_distance(a: str, b: str) -> float:
    """Cheap, dependency-free surface distance in [0, 1].

    SequenceMatcher is deliberately not called a semantic metric. It is only a
    visual coordinate showing how aggressively wording changed.
    """
    aa = " ".join(_normalise_words(a))
    bb = " ".join(_normalise_words(b))
    if not aa and not bb:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, aa, bb).ratio()


def candidate_points(item: DemoItem) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    winner_text = item.personal_text.strip()
    for c in item.candidates:
        text = str(c.get("text", ""))
        if not text:
            continue
        out.append(
            {
                "index": int(c.get("index", len(out))),
                "mode": str(c.get("mode", "candidate")),
                "text": text,
                "distance": surface_distance(item.source, text),
                "style_delta": float(c.get("style_delta", 0.0)),
                "semantic_pass": c.get("semantic_pass"),
                "is_winner": text.strip() == winner_text,
            }
        )
    return out


def short_text(text: str, width: int = 62, lines: int = 5) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    wrapped = textwrap.wrap(clean, width=width)
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1].rstrip(" .") + "…"
    return "\n".join(wrapped)


# ---------- Plot helpers ----------


def apply_dark_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BG)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(color=GRID, alpha=0.45, linewidth=0.7)


def draw_candidate_cloud(ax: plt.Axes, item: DemoItem, *, title: str = "Rewrite search") -> None:
    style_axis(ax)
    pts = candidate_points(item)
    if not pts:
        ax.text(0.5, 0.5, "No ranked candidate data in artifact", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return

    for p in pts:
        state = p["semantic_pass"]
        if p["is_winner"]:
            ax.scatter(p["distance"], p["style_delta"], s=190, marker="*", color=PERSONAL, edgecolor=TEXT, linewidth=0.9, zorder=8)
        elif state is False:
            ax.scatter(p["distance"], p["style_delta"], s=52, marker="x", color=FAIL, alpha=0.9, zorder=4)
        elif state is True:
            ax.scatter(p["distance"], p["style_delta"], s=64, color=ACCENT_2, alpha=0.85, zorder=5)
        else:
            ax.scatter(p["distance"], p["style_delta"], s=38, color=PLAIN, alpha=0.60, zorder=3)

    before_style = float(item.raw["watermarked_before"]["style"]["style_delta"])
    ax.scatter(0.0, before_style, s=90, facecolors="none", edgecolors=WM, linewidth=1.7, zorder=7)

    winner = next((p for p in pts if p["is_winner"]), None)
    if winner:
        ax.annotate(
            "selected",
            (winner["distance"], winner["style_delta"]),
            xytext=(8, 10),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            weight="bold",
        )

    ax.set_title(title, color=TEXT, fontsize=13, weight="bold", loc="left")
    ax.set_xlabel("Surface rewrite distance")
    ax.set_ylabel("Personal style Δ (nats/token)")
    ax.axhline(0, color=GRID, linewidth=0.8)

    xs = [p["distance"] for p in pts] + [0.0]
    ys = [p["style_delta"] for p in pts] + [before_style]
    xpad = max(0.03, (max(xs) - min(xs)) * 0.08)
    ypad = max(0.005, (max(ys) - min(ys)) * 0.12)
    ax.set_xlim(max(-0.02, min(xs) - xpad), min(1.0, max(xs) + xpad))
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    ax.text(
        0.01,
        0.99,
        "○ source   × semantic fail   ★ selected",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=8.5,
    )


def draw_watermark_bars(ax: plt.Axes, stats: dict[str, float], *, n: int, title: str = "SynthID signal") -> None:
    style_axis(ax)
    labels = ["Plain", "Watermarked", "Generic\nparaphrase", "Personal\nresample"]
    values = [stats["plain"], stats["before"], stats["generic"], stats["personal"]]
    colors = [PLAIN, WM, GENERIC, PERSONAL]
    bars = ax.bar(np.arange(4), values, color=colors, width=0.66)

    lo = min(values)
    hi = max(values)
    pad = max(0.008, (hi - lo) * 0.22)
    ax.set_ylim(lo - pad, hi + pad * 1.7)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Weighted-mean g-value")
    ax.set_title(title, color=TEXT, fontsize=13, weight="bold", loc="left")

    for rect, val in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + pad * 0.20, f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=10, weight="bold")

    gp = retained_fraction(stats["plain"], stats["before"], stats["generic"])
    pp = retained_fraction(stats["plain"], stats["before"], stats["personal"])
    if math.isfinite(gp) and math.isfinite(pp):
        ax.text(
            0.98,
            0.96,
            f"watermark lift retained\ngeneric {gp*100:.0f}%  ·  personal {pp*100:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=MUTED,
            fontsize=9.2,
        )
    ax.text(0.01, 0.02, f"semantic-pass subset · n={n}", transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")


def render_static(data: dict[str, Any], items: list[DemoItem], item: DemoItem, out_path: Path, *, dpi: int = 180) -> None:
    subset = semantic_subset(items) or items
    stats = aggregate_wm(subset)

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    apply_dark_figure(fig)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.25, 0.75], width_ratios=[1.1, 0.9], hspace=0.08, wspace=0.13)

    header = fig.add_subplot(gs[0, :])
    header.set_axis_off()
    header.text(0.0, 0.82, "Same idea. Different realization.", color=TEXT, fontsize=29, weight="bold", va="top")
    header.text(
        0.0,
        0.46,
        "Watermark-blind personalization searches over rewrites, keeps meaning, then selects for personal style.",
        color=MUTED,
        fontsize=14,
        va="top",
    )
    header.text(
        1.0,
        0.82,
        "POC · SynthID weighted-mean statistic",
        color=WM,
        fontsize=11,
        ha="right",
        va="top",
        weight="bold",
    )

    ax_cloud = fig.add_subplot(gs[1, 0])
    draw_candidate_cloud(ax_cloud, item, title=f"Rewrite search · {item.item_id}")

    ax_bar = fig.add_subplot(gs[1, 1])
    draw_watermark_bars(ax_bar, stats, n=len(subset), title="Watermark signal before / after")

    fig.text(
        0.012,
        0.012,
        "Transformation never receives watermark keys or detector scores. Surface distance is lexical, not semantic.",
        color=MUTED,
        fontsize=8.5,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ---------- Animation ----------


def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _phase(t: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    return _ease((t - start) / (end - start))


def _draw_gauge(ax: plt.Axes, label: str, value: float, *, color: str, baseline: float, max_value: float) -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.82, label, color=MUTED, fontsize=10, transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.0, 0.38), 1.0, 0.18, color=GRID, transform=ax.transAxes, clip_on=False))
    frac = (value - baseline) / max(1e-9, max_value - baseline)
    frac = max(0.0, min(1.0, frac))
    ax.add_patch(plt.Rectangle((0.0, 0.38), frac, 0.18, color=color, transform=ax.transAxes, clip_on=False))
    ax.text(1.0, 0.82, f"{value:.3f}", color=TEXT, fontsize=13, ha="right", weight="bold", transform=ax.transAxes)


def render_animation(
    data: dict[str, Any],
    items: list[DemoItem],
    item: DemoItem,
    out_gif: Path,
    out_mp4: Path | None,
    *,
    fps: int = 20,
    seconds: float = 8.0,
    dpi: int = 120,
) -> None:
    pts = candidate_points(item)
    subset = semantic_subset(items) or items
    agg = aggregate_wm(subset)

    total_frames = max(40, int(fps * seconds))
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    apply_dark_figure(fig)

    def draw(frame: int):
        fig.clear()
        apply_dark_figure(fig)
        t = frame / max(1, total_frames - 1)

        # Timeline: intro 0-.18, explode .12-.48, filter .42-.66, watermark .60-.86, hold .86-1
        explode = _phase(t, 0.12, 0.48)
        filtering = _phase(t, 0.42, 0.66)
        watermark = _phase(t, 0.60, 0.86)

        # Header
        if t < 0.24:
            title = "Start with a watermarked AI answer"
        elif t < 0.58:
            title = "Sample many ways to say the same thing"
        elif t < 0.78:
            title = "Keep meaning. Select for personal style."
        else:
            title = "The rewrite never saw the watermark."
        fig.text(0.055, 0.93, title, color=TEXT, fontsize=28, weight="bold", va="top")
        fig.text(0.055, 0.885, "A tiny Qwen + LoRA personalization POC", color=MUTED, fontsize=13, va="top")

        # Left: source / selected text
        text_ax = fig.add_axes([0.055, 0.17, 0.39, 0.64])
        text_ax.set_axis_off()
        text_ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=PANEL, transform=text_ax.transAxes, zorder=-2))
        text_ax.text(0.055, 0.93, item.item_id.upper(), color=WM if watermark < 0.5 else PERSONAL, fontsize=10, weight="bold", transform=text_ax.transAxes)

        mix = _phase(t, 0.54, 0.72)
        if mix < 0.5:
            body = short_text(item.source, width=54, lines=10)
            label = "watermarked source"
        else:
            body = short_text(item.personal_text, width=54, lines=10)
            label = "selected rewrite"
        text_ax.text(0.055, 0.85, label, color=MUTED, fontsize=10, transform=text_ax.transAxes)
        text_ax.text(0.055, 0.77, body, color=TEXT, fontsize=14, linespacing=1.55, va="top", transform=text_ax.transAxes)

        source_cov = item.raw["personal_resample"]["semantic"].get("source_coverage")
        cand_sup = item.raw["personal_resample"]["semantic"].get("candidate_support")
        if t > 0.55:
            text_ax.text(
                0.055,
                0.08,
                f"semantic gate  {'PASS' if item.personal_pass else 'FAIL'}  ·  source coverage {source_cov:.2f}  ·  candidate support {cand_sup:.2f}",
                color=ACCENT_2 if item.personal_pass else FAIL,
                fontsize=9.5,
                transform=text_ax.transAxes,
            )

        # Right top: candidate cloud
        cloud = fig.add_axes([0.50, 0.43, 0.45, 0.38])
        style_axis(cloud)
        cloud.set_title("Rewrite search", color=TEXT, fontsize=13, weight="bold", loc="left")
        cloud.set_xlabel("Surface rewrite distance")
        cloud.set_ylabel("Personal style Δ")
        cloud.axhline(0, color=GRID, linewidth=0.8)

        if pts:
            xs = [p["distance"] for p in pts] + [0.0]
            ys = [p["style_delta"] for p in pts] + [float(item.raw["watermarked_before"]["style"]["style_delta"])]
            xpad = max(0.03, (max(xs) - min(xs)) * 0.08)
            ypad = max(0.005, (max(ys) - min(ys)) * 0.12)
            cloud.set_xlim(max(-0.02, min(xs) - xpad), min(1.0, max(xs) + xpad))
            cloud.set_ylim(min(ys) - ypad, max(ys) + ypad)

            visible_n = int(math.ceil(len(pts) * explode))
            for j, p in enumerate(pts[:visible_n]):
                sem = p["semantic_pass"]
                alpha = 0.65
                marker = "o"
                color = PLAIN
                size = 34
                if filtering > 0:
                    if p["is_winner"]:
                        marker, color, size, alpha = "*", PERSONAL, 190, 1.0
                    elif sem is False:
                        marker, color, size, alpha = "x", FAIL, 48, max(0.12, 1.0 - 0.80 * filtering)
                    elif sem is True:
                        color, size, alpha = ACCENT_2, 64, 0.9
                    else:
                        alpha = max(0.18, 0.65 - 0.25 * filtering)
                cloud.scatter(p["distance"], p["style_delta"], s=size, marker=marker, color=color, alpha=alpha, zorder=5)

            cloud.scatter(0.0, float(item.raw["watermarked_before"]["style"]["style_delta"]), s=85, facecolors="none", edgecolors=WM, linewidth=1.6, zorder=7)

        # Right bottom: watermark gauge / aggregate context
        gauge = fig.add_axes([0.50, 0.15, 0.45, 0.20])
        gauge.set_axis_off()
        gauge.text(0.0, 0.97, "SynthID weighted-mean statistic", color=TEXT, fontsize=13, weight="bold", transform=gauge.transAxes, va="top")
        before = item.before_wm
        after = before + (item.personal_wm - before) * watermark
        base = min(item.plain_wm, item.personal_wm, item.generic_wm) - 0.005
        vmax = max(item.before_wm, item.generic_wm, item.personal_wm) + 0.003

        g1 = fig.add_axes([0.50, 0.22, 0.45, 0.08])
        _draw_gauge(g1, "watermarked", before, color=WM, baseline=base, max_value=vmax)
        g2 = fig.add_axes([0.50, 0.145, 0.45, 0.08])
        _draw_gauge(g2, "after personalization", after, color=PERSONAL, baseline=base, max_value=vmax)

        if t > 0.72:
            retained = retained_fraction(item.plain_wm, item.before_wm, item.personal_wm)
            if math.isfinite(retained):
                fig.text(0.945, 0.115, f"{retained*100:.0f}% of watermark lift retained", ha="right", color=MUTED, fontsize=11)

        fig.text(
            0.055,
            0.06,
            "watermark-blind transformation · detector measured only before/after · POC, not a calibrated attribution test",
            color=MUTED,
            fontsize=9.5,
        )
        return []

    ani = animation.FuncAnimation(fig, draw, frames=total_frames, interval=1000 / fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=animation.PillowWriter(fps=fps), dpi=dpi)

    if out_mp4 is not None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            ani.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2500), dpi=dpi)
        else:
            print("ffmpeg not found; skipped MP4 (GIF was still written)")
    plt.close(fig)


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a social-media card and animation from watermark_benchmark.json")
    p.add_argument("benchmark", type=Path, help="JSON artifact produced beside watermark_benchmark.md")
    p.add_argument("--out-dir", type=Path, default=Path("media"))
    p.add_argument("--item", help="Per-item ID for the rewrite-search panel; auto-selects a good passing example by default")
    p.add_argument("--prefix", default="demo", help="Output basename prefix")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data, items = load_benchmark(args.benchmark)
    item = choose_item(items, args.item)
    subset = semantic_subset(items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / f"{args.prefix}_card.png"
    render_static(data, items, item, png, dpi=args.dpi)
    print(f"Wrote {png}")

    print(
        f"Selected item: {item.item_id}; semantic-pass subset: "
        f"{len(subset)}/{len(items)}"
    )

    if not args.no_animation:
        gif = args.out_dir / f"{args.prefix}.gif"
        mp4 = None if args.no_mp4 else args.out_dir / f"{args.prefix}.mp4"
        render_animation(data, items, item, gif, mp4, fps=args.fps, seconds=args.seconds, dpi=max(90, int(args.dpi * 0.75)))
        print(f"Wrote {gif}")
        if mp4 is not None and mp4.exists():
            print(f"Wrote {mp4}")


if __name__ == "__main__":
    main()
