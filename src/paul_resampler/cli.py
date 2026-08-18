from __future__ import annotations

import argparse
from pathlib import Path

from .calibrate import run_calibration
from .rewrite import run_rewrite
from .semantic import DEFAULT_NLI_MODEL
from .train import train_style_adapter

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
GEN_MODEL = "Qwen/Qwen3-0.6B"


def read_source(args) -> str:
    if args.text:
        return args.text.strip()
    if args.source:
        return Path(args.source).read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --source FILE or --text '...' ")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paul-resampler",
        description="Personal-style resampling with tiny Qwen and a LoRA likelihood-ratio scorer.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train a tiny personal-style LoRA scorer")
    tr.add_argument("--corpus", required=True, help="Directory/file of your .txt/.md writing")
    tr.add_argument("--out", default="adapters/paul")
    tr.add_argument("--base-model", default=BASE_MODEL)
    tr.add_argument("--block-size", type=int, default=512)
    tr.add_argument("--epochs", type=float, default=2.0)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--batch-size", type=int, default=1)
    tr.add_argument("--grad-accum", type=int, default=8)
    tr.add_argument("--lora-r", type=int, default=8)
    tr.add_argument("--lora-alpha", type=int, default=16)
    tr.add_argument("--seed", type=int, default=42)

    rw = sub.add_parser("rewrite", help="Generate, semantic-gate, rerank, and compare with the prompting null")
    src = rw.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="File containing text to rewrite")
    src.add_argument("--text", help="Inline text to rewrite")
    rw.add_argument("--corpus", required=True, help="Same corpus; used only for the prompting-null examples")
    rw.add_argument("--adapter", default="adapters/paul")
    rw.add_argument("--out", default="runs/resample.md")
    rw.add_argument("--generator-model", default=GEN_MODEL)
    rw.add_argument("--base-model", default=BASE_MODEL)
    rw.add_argument("-n", type=int, default=32)
    rw.add_argument("--temperature", type=float, default=1.05)
    rw.add_argument("--top-p", type=float, default=0.95)
    rw.add_argument("--max-new-tokens", type=int, default=500)
    rw.add_argument("--score-max-length", type=int, default=1024)
    rw.add_argument("--seed", type=int, default=1234)
    rw.add_argument("--nli-model", default=DEFAULT_NLI_MODEL, help="Dedicated NLI model used only for semantic preservation")
    rw.add_argument("--nli-device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    rw.add_argument("--nli-entailment-threshold", type=float, default=0.50)
    rw.add_argument("--nli-max-contradiction", type=float, default=0.20)
    rw.add_argument("--nli-max-length", type=int, default=512)
    rw.add_argument("--nli-source-coverage", type=float, default=1.0, help="Fraction of source sentence/claim units that must be entailed by the candidate")
    rw.add_argument("--nli-candidate-support", type=float, default=1.0, help="Fraction of candidate sentence/claim units that must be entailed by the source")
    rw.add_argument(
        "--no-semantic-gate",
        action="store_true",
        help="Disable NLI semantic-preservation checks and rank by style only",
    )

    cal = sub.add_parser("calibrate", help="Check whether the style score separates held-out personal text from generic text")
    cal.add_argument("--paul-corpus", required=True, help="Prefer text NOT used to train the adapter")
    cal.add_argument("--generic-corpus", required=True, help="Comparison .txt/.md corpus")
    cal.add_argument("--adapter", default="adapters/paul")
    cal.add_argument("--out", default="runs/calibration.md")
    cal.add_argument("--base-model", default=BASE_MODEL)
    cal.add_argument("--chunk-tokens", type=int, default=200, help="Target tokens per deterministic non-overlapping chunk")
    cal.add_argument("--max-chunks-per-group", "--samples", dest="max_chunks_per_group", type=int, default=None, help="Optional cap; samples unique chunks without replacement (legacy --samples alias supported)")
    cal.add_argument("--score-max-length", type=int, default=1024)
    cal.add_argument("--seed", type=int, default=2026)
    return p


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train_style_adapter(
            corpus_path=args.corpus,
            output_dir=args.out,
            base_model_name=args.base_model,
            block_size=args.block_size,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
        )
    elif args.command == "rewrite":
        run_rewrite(
            source=read_source(args),
            adapter_path=args.adapter,
            corpus_path=args.corpus,
            output_path=args.out,
            generator_model_name=args.generator_model,
            base_model_name=args.base_model,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            score_max_length=args.score_max_length,
            seed=args.seed,
            semantic_gate=not args.no_semantic_gate,
            nli_model_name=args.nli_model,
            nli_device=args.nli_device,
            nli_entailment_threshold=args.nli_entailment_threshold,
            nli_max_contradiction=args.nli_max_contradiction,
            nli_max_length=args.nli_max_length,
            nli_source_coverage=args.nli_source_coverage,
            nli_candidate_support=args.nli_candidate_support,
        )
    elif args.command == "calibrate":
        run_calibration(
            paul_corpus=args.paul_corpus,
            generic_corpus=args.generic_corpus,
            adapter_path=args.adapter,
            base_model_name=args.base_model,
            output_path=args.out,
            chunk_tokens=args.chunk_tokens,
            max_chunks_per_group=args.max_chunks_per_group,
            score_max_length=args.score_max_length,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
