from paul_resampler.semantic import DirectionalNLI, semantic_decision


def score(entailment: float, neutral: float = 0.1, contradiction: float = 0.05):
    return DirectionalNLI(
        premise="p",
        hypothesis="h",
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
    )


def test_bidirectional_entailment_passes():
    assert semantic_decision(score(0.85), score(0.80))


def test_one_way_entailment_fails():
    assert not semantic_decision(score(0.90), score(0.30))


def test_contradiction_ceiling_fails_closed():
    assert not semantic_decision(score(0.80, contradiction=0.25), score(0.85))
