# Lecture 03 exercises — Pretraining: data, objectives, scaling, and small-model training

[Lecture](../lectures/03_pretraining.md) · [Tutorial](../tutorials/03_pretraining.ipynb)

Budget: 45–60 minutes. Submit your calculations, changed code or notebook, measured results, and 150–250 words interpreting the result. This sheet is worth 10 points; the optional model extension is not required for full credit.

## 1. Derive (2 points)

Derive softmax cross-entropy gradients for one preceding-character row and verify that their sum is zero.

## 2. Implement (4 points)

Add validation-based checkpoint selection to the notebook. Train with three learning rates and restore the best development checkpoint before a single test evaluation.

## 3. Investigate (4 points)

Design a 1-billion-token mixture for two languages and a specialist domain. Give proportions, deduplication rules, and two forgetting checks.

## Submission checklist

- Identify data provenance and split; record seed and software versions.
- Include a baseline and the requested comparison, with failures retained.
- Distinguish observations from hypotheses and identify a limitation.
- Disclose collaboration and any AI assistance; you must explain your code.
