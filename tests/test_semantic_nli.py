from paul_resampler.semantic import DirectionalNLI, semantic_decision, split_semantic_units


def score(entailment: float, neutral: float = 0.1, contradiction: float = 0.05):
    return DirectionalNLI(
        premise="p",
        hypothesis="h",
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
    )


def test_all_claims_supported_passes():
    assert semantic_decision([score(0.85), score(0.80)], [score(0.90), score(0.75)])


def test_missing_source_claim_fails_strict_coverage():
    assert not semantic_decision([score(0.90), score(0.30)], [score(0.90)])


def test_unsupported_candidate_claim_fails():
    assert not semantic_decision([score(0.90)], [score(0.90), score(0.25)])


def test_rubberiness_is_explicitly_tunable():
    assert semantic_decision(
        [score(0.90), score(0.90), score(0.20)],
        [score(0.90), score(0.90)],
        required_source_coverage=2 / 3,
    )


def test_contradiction_ceiling_fails_claim():
    assert not semantic_decision([score(0.80, contradiction=0.25)], [score(0.85)])


def test_sentence_splitter_keeps_three_source_sentences():
    text = "First claim. Second claim? Third claim!"
    assert split_semantic_units(text) == ["First claim.", "Second claim?", "Third claim!"]
