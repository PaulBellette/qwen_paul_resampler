from paul_resampler.corpus import CorpusDocument, token_chunks


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_token_chunks_are_balanced_and_non_overlapping():
    tok = FakeTokenizer()
    words = [f"w{i}" for i in range(401)]
    chunks = token_chunks(tok, " ".join(words), target_tokens=200)
    assert [n for _, n in chunks] == [134, 134, 133]
    reconstructed = " ".join(text for text, _ in chunks).split()
    assert reconstructed == words


def test_short_document_is_returned_once_not_replicated():
    tok = FakeTokenizer()
    chunks = token_chunks(tok, "one two three", target_tokens=200)
    assert chunks == [("one two three", 3)]
