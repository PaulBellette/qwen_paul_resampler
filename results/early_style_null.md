# Qwen personal-style resampling demo

## Source

AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship. Those regularities can arise from post-training, decoding choices, and standardized assistant behavior. As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

## Candidate search

Realization modes: `conversational`, `informal-discussion`, `from-memory`, `compressed-rebuild`. Candidates are distributed round-robin across modes; none are shown Paul examples.

## Semantic gate

Sentence/claim coverage NLI using `cross-encoder/nli-deberta-v3-small`. Each source sentence is checked against the whole candidate, and each candidate sentence against the whole source. Entailment threshold `0.50`, contradiction ceiling `0.20`, required source coverage `1.00`, candidate support `1.00`.

## Winner: resampling (semantic-gated)

**Mode:** `from-memory`  
**Style delta:** `+0.0058` nats/token  (**likelihood ratio:** `1.006x`)  
**Semantic gate:** `PASS` (source coverage `1.00`, candidate support `1.00`; whole cand→src `0.998`, src→cand `0.999`)

Machine-generated text detectors primarily rely on statistical patterns found in widely used model outputs, rather than identifying the unique signature of an authorship-based system. These patterns often develop through post-training training, during decoding choices, and the standardized behavior of standardized assistants. As language models enhance their ability to reproduce the distribution of human writing, purely text-based detection will increasingly become less reliable in practice.

## Null: explicit ‘sound like me’ prompt

**Style delta:** `+0.0952` nats/token  (**likelihood ratio:** `1.100x`)  
**Semantic gate:** `FAIL` (source coverage `0.00`, candidate support `0.00`; whole cand→src `0.000`, src→cand `0.000`)

I've been learning about physics, and I think it's like looking into the mind of god. Whatever nihilistic temperament I may be currently nursing, it's hard to deny that physics can click everything into place so sublimely that the feeling that "there must something must be going on!" is very, very hard to dispel. I learned this as a lad going to the park and playing with my friends.

## Original source score

`+0.0050` nats/token

## Ranked candidates

### 1. candidate 18 · from-memory

Style delta: `+0.0546` nats/token; ratio `1.056x`; semantic: `FAIL`

AI systems tend to use statistical patterns in model outputs that emerge from training, decoding choices, and consistent behavior in assistants. As language models grow more proficient at mimicking the distribution of human writing, text-based detection will become increasingly less effective in practice.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.9925, neutral=0.0070, contradiction=0.0005
whole source -> candidate: entail=0.9980, neutral=0.0018, contradiction=0.0002

SOURCE CLAIM COVERAGE: 1.000 (required 1.000)
  S1 PASS: entail=0.9726, neutral=0.0235, contradiction=0.0039 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 PASS: entail=0.9793, neutral=0.0175, contradiction=0.0032 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 PASS: entail=0.9859, neutral=0.0137, contradiction=0.0004 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 0.500 (required 1.000)
  C1 PASS: entail=0.9887, neutral=0.0099, contradiction=0.0014 :: AI systems tend to use statistical patterns in model outputs that emerge from training, decoding choices, and consistent behavior in assistants.
  C2 FAIL: entail=0.4139, neutral=0.5849, contradiction=0.0012 :: As language models grow more proficient at mimicking the distribution of human writing, text-based detection will become increasingly less effective in practice.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```

</details>

### 2. candidate 12 · conversational

Style delta: `+0.0433` nats/token; ratio `1.044x`; semantic: `FAIL`

A text detector usually learns from the statistical patterns in the outputs of commonly used models, rather than focusing on a unique signature of machine authorship. These patterns can come from decisions made during training or how assistants behave when decoding outputs. As language models improve their ability to reproduce human writing distributions, pure text-based detection should become less effective in principle.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.9794, neutral=0.0205, contradiction=0.0002
whole source -> candidate: entail=0.9982, neutral=0.0016, contradiction=0.0002

SOURCE CLAIM COVERAGE: 0.333 (required 1.000)
  S1 PASS: entail=0.9926, neutral=0.0073, contradiction=0.0001 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 FAIL: entail=0.0032, neutral=0.9960, contradiction=0.0008 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 FAIL: entail=0.0025, neutral=0.9972, contradiction=0.0003 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 0.333 (required 1.000)
  C1 PASS: entail=0.9971, neutral=0.0028, contradiction=0.0001 :: A text detector usually learns from the statistical patterns in the outputs of commonly used models, rather than focusing on a unique signature of machine authorship.
  C2 FAIL: entail=0.4629, neutral=0.5303, contradiction=0.0069 :: These patterns can come from decisions made during training or how assistants behave when decoding outputs.
  C3 FAIL: entail=0.0022, neutral=0.9971, contradiction=0.0008 :: As language models improve their ability to reproduce human writing distributions, pure text-based detection should become less effective in principle.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```

</details>

### 3. candidate 13 · informal-discussion

Style delta: `+0.0221` nats/token; ratio `1.022x`; semantic: `FAIL`

In principle, text detectors should become less reliable as AI models become better at replicating human writing patterns, which may arise from post-training decisions and standardized behavior.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.0150, neutral=0.9829, contradiction=0.0021
whole source -> candidate: entail=0.9981, neutral=0.0014, contradiction=0.0005

SOURCE CLAIM COVERAGE: 0.667 (required 1.000)
  S1 FAIL: entail=0.0012, neutral=0.9973, contradiction=0.0014 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 PASS: entail=0.9291, neutral=0.0649, contradiction=0.0061 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 PASS: entail=0.9881, neutral=0.0098, contradiction=0.0021 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 1.000 (required 1.000)
  C1 PASS: entail=0.9981, neutral=0.0014, contradiction=0.0005 :: In principle, text detectors should become less reliable as AI models become better at replicating human writing patterns, which may arise from post-training decisions and standardized behavior.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```

</details>

### 4. candidate 28 · conversational

Style delta: `+0.0162` nats/token; ratio `1.016x`; semantic: `FAIL`

AI text detectors often rely on statistical patterns in the output rather than an inherent signature of machine authorship. These patterns can arise from post-training model decisions or consistent behavior across standardized assistants. As language models improve at generating similar distributions to human writing, purely text-based methods should become less effective in the long run.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.0466, neutral=0.9531, contradiction=0.0003
whole source -> candidate: entail=0.9970, neutral=0.0029, contradiction=0.0001

SOURCE CLAIM COVERAGE: 0.667 (required 1.000)
  S1 PASS: entail=0.9853, neutral=0.0145, contradiction=0.0002 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 FAIL: entail=0.0956, neutral=0.8996, contradiction=0.0048 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 PASS: entail=0.9672, neutral=0.0324, contradiction=0.0004 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 0.333 (required 1.000)
  C1 PASS: entail=0.8554, neutral=0.1441, contradiction=0.0005 :: AI text detectors often rely on statistical patterns in the output rather than an inherent signature of machine authorship.
  C2 FAIL: entail=0.3018, neutral=0.6974, contradiction=0.0008 :: These patterns can arise from post-training model decisions or consistent behavior across standardized assistants.
  C3 FAIL: entail=0.0266, neutral=0.9730, contradiction=0.0003 :: As language models improve at generating similar distributions to human writing, purely text-based methods should become less effective in the long run.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```

</details>

### 5. candidate 4 · conversational

Style delta: `+0.0155` nats/token; ratio `1.016x`; semantic: `FAIL`

AI detectors typically leverage statistical patterns in the output distributions of widely used machine models rather than relying on unique signs of machine authorship. These patterns may emerge from post-training processing, decoding decisions, and the standardized behavior of assistants. As language models improve at aligning with the distribution of human writing, purely text-based detection should become increasingly less effective in practice.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.9979, neutral=0.0017, contradiction=0.0003
whole source -> candidate: entail=0.9958, neutral=0.0036, contradiction=0.0006

SOURCE CLAIM COVERAGE: 0.667 (required 1.000)
  S1 PASS: entail=0.9926, neutral=0.0055, contradiction=0.0019 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 PASS: entail=0.8092, neutral=0.1888, contradiction=0.0019 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 FAIL: entail=0.1005, neutral=0.8958, contradiction=0.0037 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 0.333 (required 1.000)
  C1 FAIL: entail=0.0795, neutral=0.9192, contradiction=0.0013 :: AI detectors typically leverage statistical patterns in the output distributions of widely used machine models rather than relying on unique signs of machine authorship.
  C2 PASS: entail=0.9888, neutral=0.0111, contradiction=0.0002 :: These patterns may emerge from post-training processing, decoding decisions, and the standardized behavior of assistants.
  C3 FAIL: entail=0.0033, neutral=0.9957, contradiction=0.0010 :: As language models improve at aligning with the distribution of human writing, purely text-based detection should become increasingly less effective in practice.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```

</details>

### 6. candidate 22 · from-memory

Style delta: `+0.0087` nats/token; ratio `1.009x`; semantic: `FAIL`

Language models increasingly rely on statistical patterns observed in human writing, rather than a direct signature of machine authorship. These patterns emerge from post-training adjustments and consistent behavior among standardized assistants. As models become more accurate at capturing the distribution of human writing, text-based detection should become less robust in practice.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.5910, neutral=0.4075, contradiction=0.0015
whole source -> candidate: entail=0.9984, neutral=0.0012, contradiction=0.0004

SOURCE CLAIM COVERAGE: 1.000 (required 1.000)
  S1 PASS: entail=0.9701, neutral=0.0280, contradiction=0.0019 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 PASS: entail=0.6466, neutral=0.3052, contradiction=0.0482 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 PASS: entail=0.9886, neutral=0.0105, contradiction=0.0009 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 0.667 (required 1.000)
  C1 PASS: entail=0.9918, neutral=0.0073, contradiction=0.0010 :: Language models increasingly rely on statistical patterns observed in human writing, rather than a direct signature of machine authorship.
  C2 FAIL: entail=0.1564, neutral=0.8423, contradiction=0.0014 :: These patterns emerge from post-training adjustments and consistent behavior among standardized assistants.
  C3 PASS: entail=0.9902, neutral=0.0093, contradiction=0.0005 :: As models become more accurate at capturing the distribution of human writing, text-based detection should become less robust in practice.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```

</details>

### 7. candidate 10 · from-memory

Style delta: `+0.0058` nats/token; ratio `1.006x`; semantic: `PASS`

Machine-generated text detectors primarily rely on statistical patterns found in widely used model outputs, rather than identifying the unique signature of an authorship-based system. These patterns often develop through post-training training, during decoding choices, and the standardized behavior of standardized assistants. As language models enhance their ability to reproduce the distribution of human writing, purely text-based detection will increasingly become less reliable in practice.

<details><summary>Semantic check</summary>

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.9984, neutral=0.0015, contradiction=0.0001
whole source -> candidate: entail=0.9987, neutral=0.0011, contradiction=0.0002

SOURCE CLAIM COVERAGE: 1.000 (required 1.000)
  S1 PASS: entail=0.9971, neutral=0.0028, contradiction=0.0001 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 PASS: entail=0.9248, neutral=0.0747, contradiction=0.0005 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 PASS: entail=0.9926, neutral=0.0064, contradiction=0.0010 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 1.000 (required 1.000)
  C1 PASS: entail=0.9959, neutral=0.0041, contradiction=0.0001 :: Machine-generated text detectors primarily rely on statistical patterns found in widely used model outputs, rather than identifying the unique signature of an authorship-based system.
  C2 PASS: entail=0.9849, neutral=0.0144, contradiction=0.0007 :: These patterns often develop through post-training training, during decoding choices, and the standardized behavior of standardized assistants.
  C3 PASS: entail=0.5961, neutral=0.4001, contradiction=0.0038 :: As language models enhance their ability to reproduce the distribution of human writing, purely text-based detection will increasingly become less reliable in practice.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: PASS
```

</details>

### 8. candidate 2 · from-memory

Style delta: `+0.0014` nats/token; ratio `1.001x`; semantic: `not checked`

AI text detectors typically rely on statistical patterns in the outputs of widely used model outputs, rather than leveraging a unique signature characteristic of machine authorship. These statistical patterns may emerge from post-training training decisions and standard assistant behaviors. As language models improve in their ability to generate and reproduce the statistical distributions of human writing, text-based detection approaches are expected to become less effective in practical applications.

### 9. candidate 5 · informal-discussion

Style delta: `+0.0013` nats/token; ratio `1.001x`; semantic: `not checked`

Text-based detection systems often rely on statistical regularities in common model outputs, not on an inherent signature of machine authorship. These regularities may stem from post-training, decoding decisions, or standard assistant behaviors. As language models become more adept at replicating human writing distributions, purely text-based detection should become less effective in principle.

### 10. candidate 23 · compressed-rebuild

Style delta: `-0.0003` nats/token; ratio `1.000x`; semantic: `not checked`

AI text detectors often leverage statistical patterns found in the outputs of commonly trained and decoded models, rather than relying on a unique signature of machine authorship. These regularities may stem from post-training adjustments, decoding decisions, and the standardized behavior of assistants. As language models get better at replicating the linguistic distribution of human writing, text-based detection methods will eventually lose their reliability in general.

### 11. candidate 15 · compressed-rebuild

Style delta: `-0.0007` nats/token; ratio `0.999x`; semantic: `not checked`

Text detection models currently rely on statistical patterns within model outputs rather than an inherent signature of machine authorship. These statistical patterns often emerge from training data, during decoding decisions, and standardized behavior of assistants. As language models become better at replicating the distribution of human writing, the reliability of purely text-based detection should gradually decrease.

### 12. candidate 27 · compressed-rebuild

Style delta: `-0.0020` nats/token; ratio `0.998x`; semantic: `not checked`

AI text detectors primarily rely on statistical patterns in model outputs rather than a unique signature of machine authorship. These patterns often develop during training and during the decoding process, as well as due to standardized assistant behavior. As language models improve at replicating the distribution of human writing, purely text-based detection may become less reliable in practice.

### 13. candidate 30 · from-memory

Style delta: `-0.0045` nats/token; ratio `0.995x`; semantic: `not checked`

Language models increasingly rely on statistical patterns that emerge from training data rather than relying on the model's inherent characteristics. These patterns can result from post-training adjustments, during decoding processes, and from the standardized behavior of assistants. As the model becomes more capable of replicating the distribution of human writing, text-based detection methods may become less effective in practice.

### 14. candidate 11 · compressed-rebuild

Style delta: `-0.0067` nats/token; ratio `0.993x`; semantic: `not checked`

AI text detectors often rely on statistical patterns found in common model outputs instead of identifying the authorship of the text itself. These patterns can emerge from post-training training data, during decoding decisions, or from standardized behavior of assistants. As language models become better at mimicking the statistical distribution of human writing, the use of purely text-based detection might eventually lose its effectiveness in principle.

### 15. candidate 24 · conversational

Style delta: `-0.0105` nats/token; ratio `0.990x`; semantic: `not checked`

AI text detectors usually rely on statistical patterns in the output of commonly used language models rather than detecting an inherent signature of machine authorship. These patterns may emerge during training, decoding decisions, or as part of standardized assistant behavior. With language models improving in their ability to reproduce the distribution of human writing, it’s becoming increasingly less reliable in principle for purely text-based detection.

### 16. candidate 17 · informal-discussion

Style delta: `-0.0110` nats/token; ratio `0.989x`; semantic: `not checked`

AI text detectors rely on statistical patterns in common model outputs rather than an inherent signature of machine authorship. These patterns emerge from post-training decoding choices and standardized assistant behavior. As language models improve at capturing human writing patterns, purely text-based detection should become less accurate in theory.

### 17. candidate 7 · compressed-rebuild

Style delta: `-0.0133` nats/token; ratio `0.987x`; semantic: `not checked`

AI text detectors rely on statistical patterns in common model outputs rather than an inherent signature of machine authorship. These patterns can emerge from post-training training, decoding choices, and standardizing how assistants behave. With language models becoming better at replicating human writing distributions, purely text-based detection will become less reliable in principle.

### 18. candidate 14 · from-memory

Style delta: `-0.0136` nats/token; ratio `0.986x`; semantic: `not checked`

AI text detectors often rely on statistical patterns in the outputs of trained models rather than detecting machine authorship directly. These patterns can emerge from post-training decoding decisions and consistent behavior among assistants. As language models improve at replicating the distribution of human writing, purely text-based detection will become less accurate in principle.

### 19. candidate 20 · conversational

Style delta: `-0.0147` nats/token; ratio `0.985x`; semantic: `not checked`

The text detectors typically use patterns found in common model outputs rather than a signature associated with machine authors. These patterns can stem from post-training decisions, decoding choices, and standard behavior when interacting with assistants. As language models improve at replicating how humans write, it is likely that purely text-based detection will become less precise in the future.

### 20. candidate 29 · informal-discussion

Style delta: `-0.0154` nats/token; ratio `0.985x`; semantic: `not checked`

AI text detectors rely on statistical patterns in model outputs, not an author's inherent signature. These patterns often originate from post-training model decisions and standardized assistant behavior. As models improve at replicating human writing distributions, purely text-based detection may become increasingly inaccurate in practical applications.

### 21. candidate 9 · informal-discussion

Style delta: `-0.0171` nats/token; ratio `0.983x`; semantic: `not checked`

AI text detectors tend to analyze patterns that emerge in the outputs of common model outputs rather than detecting an inherent signature of machine authorship. These patterns can stem from post-training training choices, decoding behaviors, and standardized assistant behavior. As language models become more adept at replicating the distribution of human writing, text-based detection should become less reliable in principle.

### 22. candidate 3 · compressed-rebuild

Style delta: `-0.0225` nats/token; ratio `0.978x`; semantic: `not checked`

AI text detectors typically focus on statistical regularities in outputs rather than detecting machine authorship. These regularities may stem from post-training model adjustments, decoding strategies, and consistent assistant behavior across different generations. As language models become more adept at replicating human writing distributions, the reliability of purely text-based detection techniques should gradually decrease.

### 23. candidate 1 · informal-discussion

Style delta: `-0.0236` nats/token; ratio `0.977x`; semantic: `not checked`

AI text detectors can use statistical regularities in model outputs rather than relying on an inherent signature of machine authorship. These regularities may originate from post-training adjustments, decoding strategies, or standardized behavior. As model capabilities improve, the reliability of purely text-based detection will decrease in practice.

### 24. candidate 25 · informal-discussion

Style delta: `-0.0256` nats/token; ratio `0.975x`; semantic: `not checked`

Text detectors increasingly rely on statistical patterns in model outputs, not a unique signature of machine authorship. These patterns often stem from post-training decoding decisions and standard assistant behaviors. As language models evolve to better replicate human writing distributions, the reliability of purely text-based detection will gradually decrease in principle.

### 25. candidate 19 · compressed-rebuild

Style delta: `-0.0301` nats/token; ratio `0.970x`; semantic: `not checked`

AI text detectors typically use statistical patterns in model outputs to identify writing, rather than relying on machine authorship. These patterns stem from post-training model decisions and consistent behavior of the assistants. As language models refine their ability to mirror the distribution of human writing, the necessity of purely text-based detection will diminish.

### 26. candidate 26 · from-memory

Style delta: `-0.0322` nats/token; ratio `0.968x`; semantic: `not checked`

AI text detectors typically rely on statistical patterns found in the outputs of widely used model systems rather than a specific feature that distinguishes a machine from human creation. These statistical patterns can emerge from post-training learning and decoding choices as well as from the behavior of standard assistants. As language models improve at generating texts closer to human writing styles, the accuracy of purely text-based detection should gradually decrease.

### 27. candidate 6 · from-memory

Style delta: `-0.0365` nats/token; ratio `0.964x`; semantic: `not checked`

Language models tend to leverage statistical patterns found in widely used model outputs rather than detect individual authorship. These patterns are often shaped by post-training decisions, decoding choices, and standard behaviors of assistants. As models become better at replicating the distribution of human text, the reliability of purely text-based detection will decrease in practice.

### 28. candidate 31 · compressed-rebuild

Style delta: `-0.0449` nats/token; ratio `0.956x`; semantic: `not checked`

Language models increasingly rely on statistical patterns in output distributions rather than on an inherent signature of machine authorship. These patterns emerge from post-training training, during decoding, and from the standardized behavior of assistants. As models become better at replicating the distribution of human writing, purely text-based detection will increasingly become less reliable in practice.

### 29. candidate 21 · informal-discussion

Style delta: `-0.0451` nats/token; ratio `0.956x`; semantic: `not checked`

AI text detectors rely on statistical patterns in the model's outputs, which are shaped by post-training behaviors and standardizing assistant outputs, rather than an inherent signature of the author. As language models become better at replicating human writing styles, purely text-based detection should eventually lose its effectiveness in principle.

### 30. candidate 0 · conversational

Style delta: `-0.0517` nats/token; ratio `0.950x`; semantic: `not checked`

What I'm trying to say is that AI text detectors tend to rely on statistical patterns found in the outputs of the models rather than something inherent to the author’s work. These regularities often come from post-training decisions made by the models, as well as the standard ways assistants are trained or designed to interact. As models become better at mimicking how humans write, I think pure text-based detection might become less effective in the long run.

### 31. candidate 16 · conversational

Style delta: `-0.0656` nats/token; ratio `0.937x`; semantic: `not checked`

A few AI text detectors rely on statistical patterns found in common model outputs rather than a signature of machine authorship. These patterns emerge from post-training decisions during decoding and standardized assistant behavior. As language models improve their ability to replicate the diversity of human writing, it becomes less likely to be reliable in principle for purely text-based detection.

### 32. candidate 8 · conversational

Style delta: `-0.0701` nats/token; ratio `0.932x`; semantic: `not checked`

You see, AI tools often use patterns from how models are trained and how they make decisions, rather than relying on what someone wrote directly. These patterns might be from the model’s learning process or how it behaves in general when talking to people. As models get better at predicting how humans write, we expect text-based detection to get less accurate over time.

## Null semantic check

```text
Sentence/claim coverage NLI semantic gate
whole candidate -> source: entail=0.0005, neutral=0.9975, contradiction=0.0021
whole source -> candidate: entail=0.0002, neutral=0.9977, contradiction=0.0022

SOURCE CLAIM COVERAGE: 0.000 (required 1.000)
  S1 FAIL: entail=0.0005, neutral=0.9927, contradiction=0.0068 :: AI text detectors generally exploit statistical regularities in common model outputs rather than an intrinsic signature of machine authorship.
  S2 FAIL: entail=0.0002, neutral=0.9965, contradiction=0.0033 :: Those regularities can arise from post-training, decoding choices, and standardized assistant behavior.
  S3 FAIL: entail=0.0004, neutral=0.9941, contradiction=0.0055 :: As language models improve at reproducing the distribution of human writing, purely text-based detection should become less reliable in principle.

CANDIDATE CLAIM SUPPORT: 0.000 (required 1.000)
  C1 FAIL: entail=0.0002, neutral=0.9985, contradiction=0.0013 :: I've been learning about physics, and I think it's like looking into the mind of god.
  C2 FAIL: entail=0.0002, neutral=0.9974, contradiction=0.0024 :: Whatever nihilistic temperament I may be currently nursing, it's hard to deny that physics can click everything into place so sublimely that the feeling that "there must something must be going on!" is very, very hard to dispel.
  C3 FAIL: entail=0.0002, neutral=0.9989, contradiction=0.0010 :: I learned this as a lad going to the park and playing with my friends.

thresholds: entailment>=0.500, contradiction<=0.200
VERDICT: FAIL
```
