# Lecture 02 exercises — Build and inspect a causal Transformer

[Lecture](../lectures/02_transformers.md) · [Tutorial](../tutorials/02_transformers.ipynb)

Budget: 45–60 minutes. Submit your calculations, changed code or notebook, measured results, and 150–250 words interpreting the result. This sheet is worth 10 points; the optional model extension is not required for full credit.

## 1. Derive (2 points)

Compute attention for Q=K=I₂ and V=[[1,0],[0,2]] with and without a causal mask.

## 2. Implement (4 points)

Add two independent attention heads, concatenate, and project. Assert shapes and test causal invariance at every prefix.

## 3. Investigate (4 points)

Remove positional embeddings and compare a sequence with a permutation under unmasked attention. Explain the observed equivariance and why a causal mask complicates the comparison.

## Submission checklist

- Identify data provenance and split; record seed and software versions.
- Include a baseline and the requested comparison, with failures retained.
- Distinguish observations from hypotheses and identify a limitation.
- Disclose collaboration and any AI assistance; you must explain your code.
