# Tiny Qwen personal-style resampler

A deliberately small proof-of-concept for the hypothesis:

> A pretrained LM already contains the ingredients of a person's writing style; style transfer can be treated partly as **selection among semantically similar realizations**, rather than only as direct prompting.

This demo compares two conditions using Qwen3-0.6B:

1. **Prompting null:** show the model examples of your writing and ask it to “sound like me”.
2. **Resampling:** give Qwen only a weak rewrite prompt, sample N rewrites, and pick the candidate with the largest personal-style likelihood-ratio score.

The score is:

```text
mean_log_p(Paul-LoRA, candidate) - mean_log_p(base-Qwen, candidate)
```

Because every candidate expresses the same source content, topic is mostly held constant while the scorer chooses among surface realizations.

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
  -n 16 \
  --out runs/demo.md
```

The command writes:

- `runs/demo.md` — source, winning resample, explicit prompting null, and all ranked candidates.
- `runs/demo.json` — machine-readable scores and outputs.

## What would count as interesting?

The key comparison is not whether the winner “sounds good”. It is whether the resampling condition consistently beats the prompting null in blind human judgement.

A useful next experiment is to take 20–50 source responses and, without looking at the method labels, choose:

- which output sounds more like you;
- which better preserves the source meaning;
- whether either is objectionably mannered.

If resampling wins on style while preserving meaning, the scorer is finding useful structure that the direct prompt is not exploiting.

## Important confound

The LoRA likelihood ratio can learn **subject matter** as well as style. Comparing candidates for the *same source* reduces this substantially, because all candidates discuss the same thing, but it does not eliminate it. A later version should hold out topics or train/evaluate across distinct domains.

## Why no automated semantic judge in v0?

A 0.6B model judging subtle semantic preservation is likely to add more noise than insight. For a POC, retain every candidate and inspect the ranked list. Once the style effect is real, add a stronger semantic gate or claim-level verifier.
