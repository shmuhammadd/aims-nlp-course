# Lecture 03 — Pretraining: data, objectives, scaling, and small-model training

[Course home](../../README.md) · Week 1, day 3 · 120 minutes

## Learning outcomes

- Train a next-token model with a correctly shifted objective.
- Diagnose overfitting using held-out loss.
- Reason about compute allocation and data mixture design.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/03_pretraining.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### What the objective rewards
For causal pretraining, minimize \(L=-\frac1N\sum_t\log p_\theta(x_t\mid x_{<t})\). The model is rewarded for prediction on the training distribution, including its omissions and errors. Masked language modelling predicts selected missing tokens with bidirectional context; encoder-decoder denoising reconstructs corrupted sequences. Choose an objective appropriate to representation learning, generation, or conditional generation.

### A tractable training derivation
Our small model stores a logit vector \(W_{i,:}\) for every preceding character i. Its prediction is \(p_j=\mathrm{softmax}(W_{i,:})_j\). For target y, \(\partial L/\partial W_{i,j}=p_j-\mathbf1[j=y]\). Average gradients across examples and apply gradient descent. This neural bigram model has no long-range context; its value is exposing the full train/evaluate loop before using a decoder implementation.

Initialize from a controlled seed. Shift inputs and targets once, compute training loss, update weights, then evaluate without updates on a separate document set. Do not concatenate documents without explicit boundary handling: an artificial transition between documents changes the task. Repeated epochs can drive down training loss while held-out loss rises.

### Data engineering is part of modelling
A pretraining pipeline needs provenance, permission to use data, language identification, quality filters, deduplication, and contamination checks. Every filter changes representation: for example, an English-oriented quality classifier may discard useful code-switched text. Record token volume before and after filtering for each language and domain. Deduplication reduces memorization pressure but does not remove all near duplicates.

### Scaling under constraints
A rough dense-transformer training estimate is \(C\approx6ND\) floating-point operations for N parameters and D training tokens; attention, embeddings, optimizer, and hardware overhead can matter. It is an estimate, not a runtime promise. Compute-optimal allocation depends on the loss regime and inference demand. The Chinchilla study challenged training larger models on too few tokens under its setting; it is not a universal fixed ratio for every architecture or application.

Mixture-of-experts routes tokens to a subset of feed-forward experts. Total parameters affect storage; active parameters better describe some compute costs. Load balancing, communication, routing stability, and expert capacity still matter. Continued pretraining on a target domain can help but may forget previous capabilities; compare both target and general-domain evaluations.

## Worked example

If N=100 million parameters and D=2 billion tokens, 6ND≈1.2×10¹⁸ FLOPs. At a sustained 10 TFLOP/s this idealized estimate is 120,000 seconds (33.3 hours), before omitted costs. Peak device throughput is not sustained throughput.

## Guided tutorial

**Question:** Train a neural character bigram model from scratch, print learning curves, and compare held-out loss with a uniform baseline.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Run `extensions/train_tiny_decoder.py` for an actual randomly initialized causal Transformer trained on the same authored corpus. The tiny run demonstrates training, not useful language generation.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why can training perplexity improve while useful performance gets worse?
2. How would duplicated test documents distort a scaling comparison?
3. Why is a randomly initialized bigram model not a small replica of a frontier LLM?

## After class

Complete the [exercise sheet](../exercises/03_pretraining.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
