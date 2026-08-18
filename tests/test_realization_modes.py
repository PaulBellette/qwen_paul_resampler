from paul_resampler.rewrite import REALIZATION_MODES, realization_messages


def test_modes_round_robin_and_do_not_mention_paul():
    names = []
    for i in range(len(REALIZATION_MODES) * 2):
        name, messages = realization_messages("A source sentence.", i)
        names.append(name)
        joined = "\n".join(m["content"] for m in messages).lower()
        assert "paul" not in joined
        assert "source sentence" in joined
    assert names[: len(REALIZATION_MODES)] == names[len(REALIZATION_MODES) :]
