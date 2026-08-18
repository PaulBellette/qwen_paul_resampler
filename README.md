# Tiny Qwen personal-style resampler

A deliberately small proof-of-concept for the hypothesis:

> A pretrained LM already contains the ingredients of a person's writing style; style transfer can be treated partly as **selection among semantically equivalent realizations**, rather than only as direct prompting.

This demo compares two conditions using Qwen3-0.6B:

1. **Prompting null:** show the model examples of your writing and ask it to “sound like me”.
2. **Resampling:** tell Qwen to express the same content *from scratch*, sample many realizations, rank them with a personal-style likelihood ratio, and take the highest-ranked candidate that passes a separate claim-preservation gate.

The style score is:

```text
mean_log_p(Paul-LoRA, candidate) - mean_log_p(base-Qwen, candidate)
```

The semantic gate does **not** contribute to the style score. It uses a dedicated bidirectional NLI model, then checks candidates from highest style score downward. The first candidate that is entailed by the source *and* entails the source wins. This implements:

```text
argmax style(candidate)
subject to meaning(candidate) ~= meaning(source)
```

without spending semantic-judge calls on candidates that cannot possibly win.

## Setup

```bash
uv sync
```

Put your own writing into `corpus/`. Plain `.txt` and `.md` files are read recursively. More is better, but a few tens of thousands of words is enough for a first poke at the idea.

Avoid filling the corpus with quoted text from other people. The LoRA learns whatever it sees.

## 1. Train the personal-style delta

```bash
uv run paul-resampler train \
  --corpus corpus \
  --out adapters/paul \
  --epochs 2
```

Defaults use `Qwen/Qwen3-0.6B-Base`, LoRA rank 8, 512-token blocks, and bf16 on a supporting CUDA GPU.

The aim is **not** a good standalone Paul generator. The adapter only needs to learn which token sequences become more likely after seeing Paul's text.

## 2. Resample a normal assistant response

Save a source response in `source.txt`, then:

```bash
uv run paul-resampler rewrite \
  --corpus corpus \
  --adapter adapters/paul \
  --source source.txt \
  --out runs/demo.md
```

The rewrite path defaults to 32 candidates at temperature 1.05 / top-p 0.95. Candidates are distributed round-robin across four generic realization modes: conversational, informal discussion, reconstruct-from-memory, and compressed/rebuilt. None mentions Paul or sees Paul examples.

The command writes:

- `runs/demo.md` — source, candidate-search modes, NLI-gated winner, explicit prompting null, all ranked candidates, and directional semantic-check details for candidates that had to be tested.
- `runs/demo.json` — machine-readable scores, claims, checks, outputs, and config.

To reproduce the old pure style-reranking behavior:

```bash
uv run paul-resampler rewrite \
  --corpus corpus \
  --adapter adapters/paul \
  --source source.txt \
  --no-semantic-gate
```

### Semantic preservation: dedicated NLI gate

The generative Qwen model no longer judges its own rewrites. By default the demo loads `cross-encoder/nli-deberta-v3-small` and runs the source/candidate pair in both directions:

```text
candidate -> source   # catches dropped source content
source -> candidate   # catches unsupported additions
```

A candidate passes only when both directions clear the entailment threshold and both stay below the contradiction ceiling. Defaults are deliberately configurable:

```bash
uv run paul-resampler rewrite \
  --corpus corpus \
  --adapter adapters/paul \
  --source source.txt \
  --nli-entailment-threshold 0.50 \
  --nli-max-contradiction 0.20 \
  --out runs/demo.md
```

This is still a POC gate rather than a proof of semantic equivalence, especially for long or highly technical passages, but it is independent of the style generator and should reject spectacular topic drift much more reliably than asking the 0.6B model to reason about its own output.

## 3. Calibrate the style scorer

The first run showed the LoRA could assign very different style deltas, but blog topic and style are entangled. Check whether the scalar score actually discriminates your writing:

```bash
uv run paul-resampler calibrate \
  --paul-corpus corpus_heldout \
  --generic-corpus generic_corpus \
  --adapter adapters/paul \
  --out runs/calibration.md
```

**Use Paul text that was not used to train the adapter** if you want this to be more than an overfitting sanity check. A simple way is to reserve whole old posts before retraining.

Calibration now partitions each document into deterministic, **non-overlapping ~200-token chunks**. It does not resample or duplicate a short document to manufacture a requested `N`. The report includes:

- document, chunk, and unique-chunk counts for each group;
- chunk score distributions and a chunk-level pairwise AUC;
- the separation margin `min(Paul) - max(generic)`;
- per-document mean scores;
- document-level AUC only when there are at least two documents in each group.

The chunk AUC is descriptive: chunks from the same post are correlated, so 20 chunks from one post are not 20 independent examples of an author. With one held-out Paul post and one Kristian post, the per-document means are the honest headline and any chunk-level separation is a useful smoke test rather than a population estimate.

You can change the target chunk size or cap work without replacement:

```bash
uv run paul-resampler calibrate \
  --paul-corpus corpus_heldout \
  --generic-corpus generic_corpus \
  --adapter adapters/paul \
  --chunk-tokens 200 \
  --max-chunks-per-group 100 \
  --out runs/calibration.md
```

The raw chunk texts, document names, token counts, and scores are retained in JSON.

## What would count as interesting?

There are now three separate questions:

1. **Does the scorer discriminate?** Held-out Paul text should tend to have higher style delta than generic text.
2. **Does broader generation expose useful stylistic variation?** The 32 candidates should span more than tiny synonym substitutions.
3. **Can selection recover style without content drift?** The semantic-gated resampling winner should beat the explicit “sound like me” null in blind judgement while preserving the source claims.

The most informative failure modes are also useful:

- **Null high style / semantic FAIL:** direct prompting is retrieving Paul-ish subject matter rather than merely style.
- **Human-best candidate exists but scorer ranks it badly:** the generator contains the style but the likelihood-ratio scorer is poor.
- **All candidates sound alike:** generation is not exploring enough of the semantic equivalence class.
- **High-style candidates systematically fail semantics:** the scorer is still exploiting topic/content correlations.

## Important confound

The LoRA likelihood ratio can learn **subject matter** as well as style. Comparing candidates for the *same source* reduces this because candidate content is mostly held constant; the semantic gate further constrains content drift. It does not make the score magically style-only.

That confound is part of the experiment: old blog posts are a fairly hostile corpus because personal subject matter, vocabulary, and surface style are naturally entangled.
