# Lecture 09 exercises — Efficient inference, compression, and deployment tradeoffs

[Lecture](../lectures/09_efficiency.md) · [Tutorial](../tutorials/09_efficiency.ipynb)

Budget: 45–60 minutes. Submit your calculations, changed code or notebook, measured results, and 150–250 words interpreting the result. This sheet is worth 10 points; the optional model extension is not required for full credit.

## 1. Derive (2 points)

Compute KV memory for B=2, T=8192, L=32, H_kv=4, d_h=128, s=2 in GiB.

## 2. Implement (4 points)

Compare per-tensor and per-output-channel 4-bit/8-bit quantization on weights with one high-range output channel.

## 3. Investigate (4 points)

Design an inference benchmark with short/long prompts and batch sizes 1/4. Define warm-up, synchronization, memory measurement, and quality checks.

## Submission checklist

- Identify data provenance and split; record seed and software versions.
- Include a baseline and the requested comparison, with failures retained.
- Distinguish observations from hypotheses and identify a limitation.
- Disclose collaboration and any AI assistance; you must explain your code.
