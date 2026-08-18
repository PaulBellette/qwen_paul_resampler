from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


@dataclass
class DirectionalNLI:
    premise: str
    hypothesis: str
    entailment: float
    neutral: float
    contradiction: float


@dataclass
class SemanticCheck:
    passed: bool
    verdict: str
    forward: DirectionalNLI
    reverse: DirectionalNLI
    source_claims: list[DirectionalNLI]
    candidate_claims: list[DirectionalNLI]
    threshold: float
    max_contradiction: float
    required_source_coverage: float
    required_candidate_support: float

    @staticmethod
    def _ok(score: DirectionalNLI, threshold: float, max_contradiction: float) -> bool:
        return score.entailment >= threshold and score.contradiction <= max_contradiction

    @property
    def source_coverage(self) -> float:
        if not self.source_claims:
            return 1.0
        return sum(self._ok(x, self.threshold, self.max_contradiction) for x in self.source_claims) / len(self.source_claims)

    @property
    def candidate_support(self) -> float:
        if not self.candidate_claims:
            return 1.0
        return sum(self._ok(x, self.threshold, self.max_contradiction) for x in self.candidate_claims) / len(self.candidate_claims)

    @property
    def raw(self) -> str:
        lines = [
            "Sentence/claim coverage NLI semantic gate",
            (
                f"whole candidate -> source: entail={self.forward.entailment:.4f}, "
                f"neutral={self.forward.neutral:.4f}, contradiction={self.forward.contradiction:.4f}"
            ),
            (
                f"whole source -> candidate: entail={self.reverse.entailment:.4f}, "
                f"neutral={self.reverse.neutral:.4f}, contradiction={self.reverse.contradiction:.4f}"
            ),
            "",
            f"SOURCE CLAIM COVERAGE: {self.source_coverage:.3f} (required {self.required_source_coverage:.3f})",
        ]
        for i, x in enumerate(self.source_claims, 1):
            ok = "PASS" if self._ok(x, self.threshold, self.max_contradiction) else "FAIL"
            lines.append(
                f"  S{i} {ok}: entail={x.entailment:.4f}, neutral={x.neutral:.4f}, "
                f"contradiction={x.contradiction:.4f} :: {x.hypothesis}"
            )
        lines.extend([
            "",
            f"CANDIDATE CLAIM SUPPORT: {self.candidate_support:.3f} (required {self.required_candidate_support:.3f})",
        ])
        for i, x in enumerate(self.candidate_claims, 1):
            ok = "PASS" if self._ok(x, self.threshold, self.max_contradiction) else "FAIL"
            lines.append(
                f"  C{i} {ok}: entail={x.entailment:.4f}, neutral={x.neutral:.4f}, "
                f"contradiction={x.contradiction:.4f} :: {x.hypothesis}"
            )
        lines.extend([
            "",
            f"thresholds: entailment>={self.threshold:.3f}, contradiction<={self.max_contradiction:.3f}",
            f"VERDICT: {self.verdict}",
        ])
        return "\n".join(lines)


@dataclass
class LoadedNLI:
    tokenizer: object
    model: object
    device: torch.device
    entailment_id: int
    neutral_id: int
    contradiction_id: int


def split_semantic_units(text: str) -> list[str]:
    """Cheap sentence-ish segmentation for coverage checks.

    Deliberately avoids an NLP dependency. Newlines are treated as boundaries and
    ordinary sentence punctuation is split when followed by a plausible sentence
    start. This is a coverage heuristic, not a claim parser.
    """
    text = re.sub(r"\s*\n+\s*", "\n", text.strip())
    units: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=["\'\(\[]?[A-Z0-9])', para)
        units.extend(p.strip() for p in parts if p.strip())
    return units or ([text.strip()] if text.strip() else [])


def _resolve_label_id(model, wanted: str) -> int:
    wanted = wanted.lower()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in id2label.items():
        if label == wanted or wanted in label:
            return idx
    raise RuntimeError(f"NLI model does not expose a '{wanted}' label: {id2label}")


def _choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_nli(model_name: str = DEFAULT_NLI_MODEL, *, device: str = "auto") -> LoadedNLI:
    target = _choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(target)
    return LoadedNLI(
        tokenizer=tokenizer,
        model=model,
        device=target,
        entailment_id=_resolve_label_id(model, "entailment"),
        neutral_id=_resolve_label_id(model, "neutral"),
        contradiction_id=_resolve_label_id(model, "contradiction"),
    )


def directional_nli(nli: LoadedNLI, premise: str, hypothesis: str, *, max_length: int = 512) -> DirectionalNLI:
    encoded = nli.tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(nli.device) for k, v in encoded.items()}
    with torch.inference_mode():
        logits = nli.model(**encoded).logits[0].float()
        probs = torch.softmax(logits, dim=-1)
    return DirectionalNLI(
        premise=premise,
        hypothesis=hypothesis,
        entailment=float(probs[nli.entailment_id].item()),
        neutral=float(probs[nli.neutral_id].item()),
        contradiction=float(probs[nli.contradiction_id].item()),
    )


def semantic_decision(
    source_claims: list[DirectionalNLI],
    candidate_claims: list[DirectionalNLI],
    *,
    entailment_threshold: float = 0.50,
    max_contradiction: float = 0.20,
    required_source_coverage: float = 1.0,
    required_candidate_support: float = 1.0,
) -> bool:
    """Coverage-based equivalence test.

    Each source sentence must be entailed by the *whole candidate* (retention),
    and each candidate sentence must be entailed by the *whole source* (no
    unsupported additions). Fractions are tunable so semantic rubberiness can be
    studied explicitly instead of hidden in one whole-passage similarity score.
    """
    def ok(x: DirectionalNLI) -> bool:
        return x.entailment >= entailment_threshold and x.contradiction <= max_contradiction

    source_fraction = 1.0 if not source_claims else sum(ok(x) for x in source_claims) / len(source_claims)
    candidate_fraction = 1.0 if not candidate_claims else sum(ok(x) for x in candidate_claims) / len(candidate_claims)
    return source_fraction >= required_source_coverage and candidate_fraction >= required_candidate_support


def verify_candidate(
    nli: LoadedNLI,
    *,
    source: str,
    candidate: str,
    entailment_threshold: float = 0.50,
    max_contradiction: float = 0.20,
    required_source_coverage: float = 1.0,
    required_candidate_support: float = 1.0,
    max_length: int = 512,
) -> SemanticCheck:
    # Keep whole-passage NLI for diagnostics, but do not use its entailment score
    # as the gate: high lexical overlap can hide a missing claim.
    forward = directional_nli(nli, candidate, source, max_length=max_length)
    reverse = directional_nli(nli, source, candidate, max_length=max_length)

    source_units = split_semantic_units(source)
    candidate_units = split_semantic_units(candidate)

    # Whole candidate -> each source unit: did the rewrite retain every source claim?
    source_claims = [directional_nli(nli, candidate, unit, max_length=max_length) for unit in source_units]
    # Whole source -> each candidate unit: did the rewrite invent anything unsupported?
    candidate_claims = [directional_nli(nli, source, unit, max_length=max_length) for unit in candidate_units]

    passed = semantic_decision(
        source_claims,
        candidate_claims,
        entailment_threshold=entailment_threshold,
        max_contradiction=max_contradiction,
        required_source_coverage=required_source_coverage,
        required_candidate_support=required_candidate_support,
    )
    return SemanticCheck(
        passed=passed,
        verdict="PASS" if passed else "FAIL",
        forward=forward,
        reverse=reverse,
        source_claims=source_claims,
        candidate_claims=candidate_claims,
        threshold=entailment_threshold,
        max_contradiction=max_contradiction,
        required_source_coverage=required_source_coverage,
        required_candidate_support=required_candidate_support,
    )
