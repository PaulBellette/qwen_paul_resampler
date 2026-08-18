from __future__ import annotations

from dataclasses import dataclass

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
    threshold: float
    max_contradiction: float

    @property
    def raw(self) -> str:
        return (
            "Bidirectional NLI semantic gate\n"
            f"candidate -> source: entail={self.forward.entailment:.4f}, "
            f"neutral={self.forward.neutral:.4f}, contradiction={self.forward.contradiction:.4f}\n"
            f"source -> candidate: entail={self.reverse.entailment:.4f}, "
            f"neutral={self.reverse.neutral:.4f}, contradiction={self.reverse.contradiction:.4f}\n"
            f"thresholds: entailment>={self.threshold:.3f}, "
            f"contradiction<={self.max_contradiction:.3f}\n"
            f"VERDICT: {self.verdict}"
        )


@dataclass
class LoadedNLI:
    tokenizer: object
    model: object
    device: torch.device
    entailment_id: int
    neutral_id: int
    contradiction_id: int


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
    forward: DirectionalNLI,
    reverse: DirectionalNLI,
    *,
    entailment_threshold: float = 0.50,
    max_contradiction: float = 0.20,
) -> bool:
    """Require approximate equivalence rather than one-way implication.

    candidate -> source penalizes dropped content; source -> candidate penalizes
    substantive additions. The contradiction ceiling catches direct reversals
    even when entailment calibration is imperfect.
    """
    return (
        forward.entailment >= entailment_threshold
        and reverse.entailment >= entailment_threshold
        and forward.contradiction <= max_contradiction
        and reverse.contradiction <= max_contradiction
    )


def verify_candidate(
    nli: LoadedNLI,
    *,
    source: str,
    candidate: str,
    entailment_threshold: float = 0.50,
    max_contradiction: float = 0.20,
    max_length: int = 512,
) -> SemanticCheck:
    # If candidate entails source, the candidate has retained the source claims.
    forward = directional_nli(nli, candidate, source, max_length=max_length)
    # If source entails candidate, the candidate has not added unsupported claims.
    reverse = directional_nli(nli, source, candidate, max_length=max_length)
    passed = semantic_decision(
        forward,
        reverse,
        entailment_threshold=entailment_threshold,
        max_contradiction=max_contradiction,
    )
    return SemanticCheck(
        passed=passed,
        verdict="PASS" if passed else "FAIL",
        forward=forward,
        reverse=reverse,
        threshold=entailment_threshold,
        max_contradiction=max_contradiction,
    )
