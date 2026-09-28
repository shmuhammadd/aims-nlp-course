# Lecture 09 — Efficient inference, compression, and deployment tradeoffs

[Course home](../../README.md) · Week 2, day 4 · 120 minutes

## Learning outcomes

- Calculate model and KV-cache memory separately.
- Implement per-channel quantization and measure error.
- Explain prefill, decoding, batching, and speculative decoding.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/09_efficiency.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### What consumes memory
Weights are one component. At inference, include activations, temporary workspaces, the key/value cache, and serving overhead. During training, also include gradients and optimizer state. For a dense model with N parameters at b bits per weight, idealized weight storage is Nb/8 bytes, before scales, metadata, and unquantized layers.

For batch B, cached length T, layers L, H_kv key/value heads of dimension d_h, and s bytes per cached element, a simple cache estimate is
\[M_{KV}=2BTLH_{kv}d_hs.\]
The leading 2 counts keys and values. Grouped-query attention reduces H_kv. Sliding-window layers can bound part of the cache; implementation details determine actual allocation.

### Prefill versus decode
Prefill processes the prompt in parallel. Decode generates tokens sequentially, often becoming limited by memory movement and cache reads at small batch sizes. Report time to first token separately from output tokens per second. Include warm-up, synchronization on accelerators, batch size, input length, and output length in timing. Bigger batches can raise throughput while increasing latency.

FlashAttention reorganizes exact attention to reduce memory traffic; it is not simply a sparse or approximate attention rule. Paged cache management reduces allocation waste and supports serving many sequences. Neither guarantees a fixed speedup for every device and workload.

### Quantization and distillation
For symmetric per-channel quantization, choose \(s_c=\max|W_c|/(2^{b-1}-1)\), quantize \(q=\mathrm{clip}(\mathrm{round}(W/s_c))\), then reconstruct \(\hat W=s_cq\). Handle all-zero channels. Per-channel scaling often reduces error when channel ranges differ, but introduces metadata. Quantization error in weights is not task accuracy; evaluate outputs too.

Knowledge distillation trains a student from teacher outputs or distributions. Temperature-smoothed KL divergence can convey alternatives, but the teacher can transmit errors and data restrictions. Speculative decoding drafts tokens with a cheaper model and verifies them with a target model; correct acceptance/rejection can preserve the target distribution. Speed depends on acceptance, draft cost, and hardware.

### Deployment as an experiment
Choose a quality-latency-memory operating point, not a single fastest number. Measure on representative inputs, including long prompts and underrepresented languages. Quantization kernels may be unavailable on a laptop even if theoretical storage looks attractive. The notebook calculates estimates and reconstruction errors; it does not claim hardware speedups.

## Worked example

For B=1, T=4096, L=24, H_kv=8, d_h=64 and s=2 bytes, the KV cache estimate is 201,326,592 bytes=192 MiB. With 2 KV heads it becomes 48 MiB. Four-bit storage for 1 billion weights is ideally 500 MB, excluding metadata.

## Guided tutorial

**Question:** Quantize a matrix at two bit widths and estimate KV-cache growth with context length and head sharing.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Time actual runs of `extensions/hf_text.py` on a declared device. Its wall time includes model loading unless you instrument generation separately; do not call it decode throughput.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why does four-bit weight storage not make all inference memory four-bit?
2. Why can a throughput improvement worsen user latency?
3. What additional evidence is needed after measuring low weight MSE?

## After class

Complete the [exercise sheet](../exercises/09_efficiency.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
