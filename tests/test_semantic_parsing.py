from paul_resampler.semantic import parse_claims


def test_claim_parser_prefixed_lines():
    raw = "CLAIM: One thing is true.\nCLAIM: A second thing is qualified."
    assert parse_claims(raw, "fallback") == ["One thing is true.", "A second thing is qualified."]


def test_claim_parser_accepts_bullets():
    raw = "- First substantive statement here.\n- Second substantive statement here."
    assert parse_claims(raw, "fallback") == [
        "First substantive statement here.",
        "Second substantive statement here.",
    ]
