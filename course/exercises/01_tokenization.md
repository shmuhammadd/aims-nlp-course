# Lecture 01 exercises — Language models, tokens, and experimental baselines

[Lecture](../lectures/01_tokenization.md) · [Tutorial](../tutorials/01_tokenization.ipynb)

Budget: 45–60 minutes. Submit your calculations, changed code or notebook, measured results, and 150–250 words interpreting the result. This sheet is worth 10 points; the optional model extension is not required for full credit.

## 1. Derive (2 points)

Calculate NLL, perplexity, and bits per byte for the worked example when the original text is 12 bytes. State all logarithm bases.

## 2. Implement (4 points)

Extend the notebook with an NFC-normalized character tokenizer and report counts before/after on five original examples. Include a combining-accent example.

## 3. Investigate (4 points)

Create a 20-example tokenization audit across two languages you can assess, matched by domain. Document authorship, counts, limitations, and a document-level split.

## Submission checklist

- Identify data provenance and split; record seed and software versions.
- Include a baseline and the requested comparison, with failures retained.
- Distinguish observations from hypotheses and identify a limitation.
- Disclose collaboration and any AI assistance; you must explain your code.
