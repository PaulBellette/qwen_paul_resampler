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

The semantic gate does **not** contribute to the style score. It uses a dedicated NLI model with sentence/claim coverage checks, then checks candidates from highest style score downward. The first candidate that preserves enough source units without adding unsupported candidate units wins. This implements:

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

### Semantic preservation: sentence/claim coverage NLI

The generative Qwen model no longer judges its own rewrites. By default the demo loads `cross-encoder/nli-deberta-v3-small`, splits source and candidate into sentence-like semantic units, and asks two coverage questions:

```text
whole candidate -> each source sentence   # did we retain this source claim?
whole source    -> each candidate sentence # did we invent this candidate claim?
```

Whole-passage bidirectional NLI is still recorded for diagnostics, but no longer decides the gate: the previous version showed that high lexical overlap could hide a missing claim. By default every source and candidate unit must pass, but the *fraction* required is explicitly configurable so semantic rubberiness can be studied rather than silently baked into the judge:

```bash
uv run paul-resampler rewrite \
  --corpus corpus \
  --adapter adapters/paul \
  --source source.txt \
  --nli-entailment-threshold 0.50 \
  --nli-max-contradiction 0.20 \
  --nli-source-coverage 1.0 \
  --nli-candidate-support 1.0 \
  --out runs/demo.md
```

For a deliberately looser human-ish experiment you can, for example, allow one source sentence in five to be weakly covered with `--nli-source-coverage 0.8`. This is still a POC heuristic rather than a proof of semantic equivalence, especially for long or highly technical passages, but its failures should now be visible at the individual source/candidate sentence level.

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

## 4. Watermark robustness experiment

This layer was added **after** the personal resampler. It deliberately keeps watermark generation/detection outside the resampling API so the transformation cannot adapt to the watermark key or score.

The first target is the open SynthID Text implementation shipped in Hugging Face Transformers. We use Transformers for watermark generation and for computing SynthID g-values, then reproduce the simple **Mean / Weighted Mean** detector statistic from Google DeepMind's reference implementation. The weighted score uses linearly decreasing watermark-depth weights from 10 to 1. This is a research statistic, **not** a calibrated yes/no provenance threshold.

### Phase A — generate and freeze source texts

Start with the included prompt set or copy it and edit it:

```bash
cp experiments/prompts.example.jsonl experiments/prompts.jsonl

uv run paul-resampler watermark-generate \
  --prompts experiments/prompts.jsonl \
  --out runs/watermark_sources.jsonl
```

For every prompt this generates a controlled pair from the same Qwen model and RNG seed:

```text
prompt
  ├── plain Qwen output
  └── Qwen + SynthID output
```

The JSONL records the prompt, texts, generation settings, SynthID configuration, and detector scores. **Freeze this file once generated.** The command refuses to overwrite an existing source file unless `--overwrite` is passed deliberately. Do not regenerate it while tuning the downstream resampler; otherwise it is easy to select lucky/unlucky watermark instances by accident.

The default public SynthID keys use the 30-key configuration published in Google DeepMind's reference implementation. They are intentionally not secret: this experiment asks about incidental robustness under a watermark-blind transformation, not production security.

### Phase B — transform without watermark feedback

```bash
uv run paul-resampler watermark-test \
  --inputs runs/watermark_sources.jsonl \
  --adapter adapters/paul \
  --out runs/watermark_benchmark.md
```

For each frozen **watermarked** source the benchmark runs two transformations:

```text
watermarked source
  ├── generic semantic-preserving paraphrase
  └── existing Paul resampler (style rank + NLI gate)
```

Only after each transformation is complete does the benchmark measure the SynthID detector statistic again. The resampler function receives the source text, generator, style scorer and semantic gate; it receives **no watermark configuration, keys, score, or detector callback**.

The generic paraphrase is an important control. If both generic rewriting and personal resampling destroy the signal equally, the result is simply "rewriting damages this watermark." If personal resampling behaves differently, that is the more interesting observation.

Outputs:

- `runs/watermark_benchmark.md` — compact aggregate and per-item comparison.
- `runs/watermark_benchmark.json` — all source/control/personal texts, SynthID statistics, style scores, semantic checks, and ranked personal candidates.

The headline columns are:

```text
plain detector statistic
watermarked-before detector statistic
generic-paraphrase-after detector statistic
personal-resample-after detector statistic
semantic preservation
personal style delta
```

For a cheap smoke test before running the whole prompt set:

```bash
uv run paul-resampler watermark-generate \
  --prompts experiments/prompts.example.jsonl \
  --limit 2 \
  --out runs/watermark_sources_smoke.jsonl

uv run paul-resampler watermark-test \
  --inputs runs/watermark_sources_smoke.jsonl \
  --adapter adapters/paul \
  -n 8 \
  --out runs/watermark_smoke.md
```

### Interpretation warning

The simple SynthID weighted-mean statistic is useful for comparing distributions under a fixed key/configuration, but a single raw score is not a universal detection probability. Google explicitly recommends calibration at the desired false-positive rate when using the Mean/Weighted Mean detector across lengths. If the first experiment is interesting, the next step is to collect enough plain/watermarked controls to calibrate score-vs-length before making claims about "watermark removed" or "watermark survives."

## 5. Render a social-media demo from the saved results

The renderer is deliberately downstream-only: it reads `watermark_benchmark.json` and **does not rerun any model, semantic gate, style scorer, or watermark detector**.

Install the optional media dependencies:

```bash
uv sync --extra media
```

Then render a static 16:9 card plus an 8-second looping GIF and (when `ffmpeg` is available) MP4:

```bash
uv run python scripts/render_demo.py \
  results/poc_v1/watermark_benchmark.json \
  --out-dir media
```

The script auto-selects a semantically valid example with a strong watermark attenuation, or you can pin one:

```bash
uv run python scripts/render_demo.py \
  results/poc_v1/watermark_benchmark.json \
  --item mundane \
  --out-dir media
```

The static card combines two views generated entirely from the recorded artifact:

- **Rewrite search:** each personal-resampling candidate is plotted by lexical/surface rewrite distance from the watermarked source versus personal-style likelihood-ratio delta. Semantic failures are marked separately and the selected candidate is highlighted. Surface distance is intentionally labelled as such; it is not an embedding or semantic metric.
- **Watermark attenuation:** aggregate plain, watermarked, generic-paraphrase and personal-resample weighted-mean SynthID statistics on the subset where both transformations passed the semantic gate. It also reports the fraction of watermark lift retained relative to the plain-generation baseline.

The animation tells the same story for one example: source text → candidate cloud → semantic filtering → selected rewrite → before/after SynthID statistic. The renderer never receives anything beyond the saved benchmark JSON.
