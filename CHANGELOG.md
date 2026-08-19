# Changelog

## 0.6.0

- Add a two-phase SynthID Text robustness experiment.
- `watermark-generate` creates paired plain/watermarked Qwen outputs and freezes them to JSONL.
- `watermark-test` measures the fixed source, a generic-paraphrase control, and the existing personal resampler.
- Reuse the same personal-resampling candidate selection via a watermark-blind in-memory helper.
- Add simple Mean/Weighted Mean SynthID g-value scoring based on the Google DeepMind reference detector statistic.
- Add an example 10-prompt benchmark set and watermark-statistic tests.
- Refuse to overwrite frozen source JSONL unless `--overwrite` is explicitly passed; benchmark output records its SHA-256.
- Require Transformers >= 4.52.3 for the public SynthID watermarking API.


## 0.5.0

- Replaced whole-passage NLI as the semantic decision rule with sentence/claim coverage NLI.
- Each source sentence must be entailed by the whole candidate; each candidate sentence must be supported by the whole source.
- Kept whole-passage bidirectional NLI scores as diagnostics only.
- Added explicit `--nli-source-coverage` and `--nli-candidate-support` controls so semantic looseness can be tuned rather than hidden.
- Reports now show per-unit entailment/neutral/contradiction scores and aggregate coverage fractions.

## 0.4.0

- Replaced the tiny generative semantic judge with a dedicated bidirectional NLI gate.
- Added four generic realization modes and round-robin candidate generation to broaden the semantic-equivalence search.
- Added configurable NLI model/device/thresholds and surfaced directional entailment scores in reports.
- Added a project `.gitignore` covering virtualenvs, model artefacts, runs, editor junk, and private corpora while preserving placeholder READMEs.

## 0.3.0

- Reworked calibration to use deterministic, non-overlapping token chunks rather than random character windows sampled with replacement.
- Calibration never duplicates a short document just to hit a requested sample count.
- Added document/chunk/unique-chunk accounting and duplicate detection.
- Added per-document mean style scores and only report document-level AUC with at least two documents in each group.
- Relabelled chunk AUC as a descriptive ordering statistic because chunks from one document are correlated.
- Added min-Paul minus max-generic separation margin.
- Added `--chunk-tokens` and optional `--max-chunks-per-group` calibration controls. The old `--samples` flag remains as an alias for the latter.

## 0.2.0

- Broadened the weak rewrite prompt from close paraphrasing to fresh realization of the same claims.
- Increased default exploration to 32 candidates, temperature 1.05, top-p 0.95.
- Added claim extraction and a conservative semantic-preservation gate.
- Style-rank first, then semantic-check from the top until the first feasible candidate passes.
- Kept the explicit “sound like me” prompt unchanged as the null condition.
- Added raw semantic-check output and extracted claims to reports/JSON.
- Added `--no-semantic-gate` to reproduce v0.1 style-only reranking.
- Added `calibrate` to compare held-out personal prose against a generic corpus and report score distributions plus pairwise AUC.

## 0.7.0

- Added `scripts/render_demo.py` for repo-generated social-media visuals from `watermark_benchmark.json`.
- Generates a static 16:9 PNG card and looping GIF; also MP4 when `ffmpeg` is installed.
- Candidate cloud uses dependency-free surface rewrite distance versus recorded personal-style delta and marks semantic failures/selected candidate.
- Watermark panel reports semantic-pass-subset aggregate statistics and watermark-lift retention.
- Added optional `media` dependencies and tracked `media/README.md`.
