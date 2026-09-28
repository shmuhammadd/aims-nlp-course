# Lecture 04 exercises — Instruction tuning and parameter-efficient adaptation

[Lecture](../lectures/04_adaptation.md) · [Tutorial](../tutorials/04_adaptation.ipynb)

Budget: 45–60 minutes. Submit your calculations, changed code or notebook, measured results, and 150–250 words interpreting the result. This sheet is worth 10 points; the optional model extension is not required for full credit.

## 1. Derive (2 points)

Derive gradients of mean squared error for A and B under Y=X(W+sAB).

## 2. Implement (4 points)

Sweep ranks 1, 2, 4, and 8 in the notebook; compare held-out MSE and parameter counts over three seeds.

## 3. Investigate (4 points)

Prepare six original prompt-response examples and manually mark supervised tokens, including an empty response and a padded example. Explain your handling of EOS.

## Submission checklist

- Identify data provenance and split; record seed and software versions.
- Include a baseline and the requested comparison, with failures retained.
- Distinguish observations from hypotheses and identify a limitation.
- Disclose collaboration and any AI assistance; you must explain your code.
