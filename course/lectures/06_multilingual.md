# Lecture 06 — Multilingual and low-resource language modelling

[Course home](../../README.md) · Week 2, day 1 · 120 minutes

## Learning outcomes

- Compare micro, macro, and per-language performance.
- Explain transfer, continued pretraining, and language sampling tradeoffs.
- Design an evaluation with community and linguistic context.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/06_multilingual.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### Transfer is conditional
A shared model may reuse syntax, semantics, or subword patterns across languages, but transfer depends on training exposure, script, domain, and supervision. Related languages are not interchangeable. Code-switching and spelling variation can invalidate assumptions in a monolingual tokenizer or label schema. Translation-based augmentation can introduce translationese and change culturally situated sentiment.

A multilingual pipeline can use a frozen encoder with a task head, task-specific fine-tuning, language adapters, continued pretraining, or instruction tuning. Compare an inexpensive classifier before investing in generative fine-tuning. The existing AfriSenti notebook in this repository is an optional historical bridge; its environment needs separate verification.

### Sampling and representation
If language l has n_l examples, a temperature-smoothed sampling distribution can be $q_l=n_l^\alpha/\sum_j n_j^\alpha$, where alpha=1 follows the corpus and alpha=0 samples languages uniformly. Reducing alpha increases exposure for smaller languages, but repeated examples may overfit. Count unique examples and effective repetitions, not only optimizer steps.

### Metrics with explicit denominators
For a class, precision=TP/(TP+FP), recall=TP/(TP+FN), and F1 is their harmonic mean. State how undefined terms are treated. Macro-F1 averages class F1 values and highlights minority-class failures. Overall accuracy weights examples; equal-language accuracy averages language accuracies. These answer different questions. A language with 20 examples has much wider uncertainty than one with 2,000.

Use paired bootstrap resampling when comparing predictions on the same evaluation units. If speakers or documents induce dependence, resample clusters rather than individual rows. A confidence interval quantifies a sampling procedure under assumptions; it does not correct an unrepresentative sample.

### Data and participation
AfriSenti provides a concrete African-language sentiment benchmark. Read its annotation and access conditions before using it. Do not assume a web-hosted dataset is unrestricted or still available in the same form. For this course, synthetic examples make all mandatory labs independent of downloads; a real-data extension uses student-supplied permitted TSV files.

A dataset statement should identify language varieties, collection context, annotators, consent/access conditions, exclusions, and known limitations. Ask speakers to interpret errors, with credit and appropriate compensation in real projects. Do not make students disclose sensitive language or identity information to participate.

## Worked example

A model gets 90/100 English examples and 3/10 Hausa examples correct. Overall accuracy is 93/110=84.55%, while equal-language accuracy is (0.90+0.30)/2=60%. Neither should be reported without the per-language counts.

## Guided tutorial

**Question:** Expose aggregate metric failures, bootstrap a paired comparison, and inspect language sampling weights.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Run `extensions/afrisenti_baseline.py` on permitted train/dev/test TSV exports. It fits a character TF-IDF classifier without downloading or redistributing tweets.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why might uniform language sampling hurt a very small language?
2. Can translated evaluation data establish performance on naturally occurring code-switching?
3. Why is macro-F1 useful but insufficient for a multilingual system?

## After class

Complete the [exercise sheet](../exercises/06_multilingual.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [AfriSenti](https://aclanthology.org/2023.emnlp-main.862/)
- [Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)](https://arxiv.org/abs/1911.02116)
