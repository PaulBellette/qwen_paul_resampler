# Changelog

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
