# Lecture 13 — Efficient inference and quantisation

[Course home](../../README.md) · Week 3, day 3 · 120 minutes · [Slides](../slides/13_efficiency.pptx) · [Notebook](../tutorials/13_efficiency.ipynb) · [Exercises](../exercises/13_efficiency.md) · [Quiz](../quizzes/13_efficiency.md)

## Learning outcomes

- Work out weight memory and KV-cache memory for a model, and decide which model and precision fit a free Colab GPU or a laptop.
- Implement symmetric per-channel quantisation and measure its error on weights and on outputs.
- Explain prefill, decode, batching, speculative decoding and distillation, and state what must be measured before a compressed model is used.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 12 (retrieval, tools and agents) |
| 10–35 | Slides 1–10: the Task 7 memory problem, weight memory, the fit table, the KV cache on the board |
| 35–60 | Slides 11–21: prefill and decode, FlashAttention, paged attention, speculative decoding, quantisation, distillation, the transfer check |
| 60–65 | Break |
| 65–105 | [Notebook 13](../tutorials/13_efficiency.ipynb): sections 1–6 together, section 7 in pairs |
| 105–115 | Slides 22–23: deployment as an experiment; discuss the Hausa question |
| 115–120 | Exit ticket: the memory line of your plan, one measured number, one thing the arithmetic leaves out |

## Teaching notes

### The problem
Task 7 ranks `meta-llama/Llama-3.1-8B-Instruct`. It has 8.03 billion weights. At 16 bits each that is 16.1 GB, and a free Colab T4 has 15 GB. At 4 bits the weights take 4.0 GB, and about 5.5 GB once loaded with overhead. A team with a free GPU works on the 4-bit model, and the organisers run the 16-bit one. A steering vector found at 4 bits may not transfer, so the team has to measure the transfer.

### What consumes memory
Weights are one part. At inference, add activations, temporary workspace, the key/value (KV) cache and the serving runtime. Training also holds gradients and optimiser state. A dense model with $N$ weights at $b$ bits each needs $Nb/8$ bytes for weights, before scales, metadata and any layers left unquantised.

For batch $B$, cached length $T$, $L$ layers, $H_{kv}$ key/value heads of dimension $d_h$ and $s$ bytes per number, the cache holds

$$
M_{KV}=2\,B\,T\,L\,H_{kv}\,d_h\,s \text{ bytes.}
$$

The leading 2 counts keys and values. Grouped-query attention lets several query heads share one KV head, which lowers $H_{kv}$. Sliding-window layers bound part of the cache. The library decides the real allocation, so measure it.

Llama-3.1-8B has 32 layers and 32 query heads that share 8 KV heads of dimension 128. At 16 bits and 4,096 tokens the cache is $2\times4096\times32\times8\times128\times2=2^{29}$ bytes, or 0.5 GiB. At the full 131,072-token context it is 16 GiB, as large as the weights. In common 4-bit set-ups the weights are 4-bit and the cache stays at 16 bits.

### What fits
Ideal weight memory in GB, from $N b/8$. The fit rule is the course's assumption: keep 3 GB free on a 15 GB T4 for activations, cache, runtime and quantisation overhead, and allow the model 4 GB on a laptop with 8 GB of memory.

| Model (nominal size) | 16-bit | 8-bit | 4-bit | Free Colab T4 | 8 GB laptop |
|---|---|---|---|---|---|
| SmolVLM-256M, SmolVLM-500M | 0.5, 1.0 | 0.26, 0.5 | 0.13, 0.25 | Any precision | Any precision |
| Gemma 3 1B | 2.0 | 1.0 | 0.5 | Any precision | 16-bit or lower |
| SmolVLM 2.2B | 4.4 | 2.2 | 1.1 | Any precision | 8-bit or 4-bit |
| Gemma 3 4B | 8.0 | 4.0 | 2.0 | 16-bit or lower | 4-bit |
| Llama-3.1-8B | 16.1 | 8.0 | 4.0 | 8-bit or 4-bit | Does not fit |
| Gemma 3 12B | 24.0 | 12.0 | 6.0 | 4-bit | Does not fit |

A CPU runs 32-bit floats by default, which doubles the 16-bit column. Low-bit inference on a CPU needs a runtime with kernels for it, such as llama.cpp. A model that fits may still be too slow to finish a development run, and the table says nothing about speed.

### Prefill and decode
Prefill processes the whole prompt in parallel and fills the cache. Decode then produces one token at a time, each step reading the whole cache; at small batch sizes its speed is set by how fast memory can be read. Report time to first token apart from output tokens per second. A timing needs the device, batch size, input length, output length, a warm-up run, and a synchronisation call on a GPU before the clock is read. A larger batch raises throughput and makes each request wait longer.

FlashAttention (Dao et al., 2022) computes exact attention in blocks that fit in the GPU's fast on-chip memory, so the full $T\times T$ attention matrix is never written to main GPU memory. It is the same function as standard attention, with less memory traffic. PagedAttention (Kwon et al., 2023) stores the KV cache in fixed-size blocks that need not be contiguous, as an operating system pages memory. It wastes less memory when many sequences of different lengths are served together. Neither gives a fixed speedup on every device and workload.

### Quantisation
Symmetric per-channel quantisation at $b$ bits takes each channel $c$ of a weight matrix and computes

$$
s_c=\frac{\max|W_c|}{2^{b-1}-1},\qquad q=\mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{W}{s_c}\right)\right),\qquad \hat W=s_c\,q .
$$

A channel of zeros would give $s_c=0$, so set its scale to 1. Each reconstructed weight is within $s_c/2$ of the original. One scale for the whole tensor is set by its largest channel. In the lab, one output channel is 30 times larger than the rest; at 4 bits with a shared scale every weight in the other 31 channels rounds to zero. Per-channel scales lower the weight error from 1.20 to 0.23 at 4 bits and from 0.0205 to 0.0006 at 8 bits. They cost one stored scale per channel.

Weight error is not task accuracy. Evaluate the outputs. Libraries that load 4-bit models, such as bitsandbytes, use non-uniform levels (NF4, from the QLoRA paper) and compute in 16 bits; the principle of a few levels and a scale is the same.

### Distillation
Knowledge distillation trains a small student on a large teacher's output distributions (Hinton et al., 2015). The loss is the KL divergence between teacher and student distributions, both softened by a temperature $T$, usually mixed with the ordinary loss on labels. A higher temperature shows the student which wrong answers the teacher finds plausible. The teacher also passes on its errors, and its licence may restrict use of its outputs.

### Speculative decoding
A cheap draft model proposes several tokens. The target model scores them all in one parallel pass. Each drafted token $x$ is accepted with probability $\min(1, p(x)/q(x))$, where $p$ is the target's probability and $q$ the draft's. At the first rejection the token is resampled from $\max(p-q,0)$, renormalised, and drafting restarts. This rule returns samples distributed as the target model's (Leviathan et al., 2022). With acceptance rate $\alpha$ and $\gamma$ drafted tokens, one target pass yields $(1-\alpha^{\gamma+1})/(1-\alpha)$ tokens on average. Speed depends on the acceptance rate, the cost of the draft model and the hardware.

### A steering vector across precisions
For Task 7, first test the method on a model small enough to load at both 4 and 16 bits. Compute the steering vector at each precision and report the cosine between them. Then apply the 4-bit vector to the 16-bit model and score compliance and relevance on the same development questions. In the lab's two-layer toy the cosine is 0.988 at 4 bits and 0.528 at 2 bits. A 32-layer model can differ.

### Deployment as an experiment
Choose an operating point on quality, latency and memory. One fastest number is not a result. Measure on inputs like the ones the system will see, with long prompts and with the lower-resource languages included: a model quantised without loss on English may lose more on Hausa, and only a per-language evaluation shows it. The notebook computes memory estimates and reconstruction errors. It claims no speedup on any hardware.

## Worked example

**KV cache.** For $B=1$, $T=4096$, $L=24$, $H_{kv}=8$, $d_h=64$ and $s=2$ bytes, the cache is 201,326,592 bytes, or 192 MiB. With 2 KV heads it is 48 MiB. For Llama-3.1-8B at the same length ($L=32$, $H_{kv}=8$, $d_h=128$) it is 536,870,912 bytes, or 0.5 GiB; a batch of 8 needs 4 GiB.

**Weights.** Four-bit storage for 1 billion weights is 500 MB before metadata. For 8.03 billion weights: 16.1 GB at 16 bits, 8.0 GB at 8 bits, 4.0 GB at 4 bits.

**Quantising one channel.** Weights $(0.52, -1.4, 0.71)$ at 4 bits: $q_{\max}=7$ and $s=1.4/7=0.2$. Dividing gives $(2.6, -7, 3.55)$, which rounds to $(3, -7, 4)$, and $\hat W=(0.6, -1.4, 0.8)$. The errors are 0.08, 0 and 0.09, each under $s/2=0.1$. If another channel of the same tensor has a largest weight of 30 and the scale is shared, $s=30/7=4.29$ and all three weights round to 0.

**Speculative acceptance.** The draft gives a token $q=0.8$ and the target gives it $p=0.6$: accept with probability 0.75. With $\alpha=0.8$ and $\gamma=4$, one target pass yields $(1-0.8^5)/0.2=3.36$ tokens.

**Temperature.** Teacher logits $(2, 1, 0)$ give probabilities $(0.665, 0.245, 0.090)$ at $T=1$ and $(0.506, 0.307, 0.186)$ at $T=2$.

## Guided tutorial

**Question:** Which model and precision fit your hardware, and what does 4-bit storage do to the weights, the outputs and a steering direction?

The notebook prints weight memory and a fit table for seven model sizes, writes out the KV cache for Llama-3.1-8B, quantises a matrix per tensor and per channel at 4 and 8 bits, compares ideal and actual storage, tracks a difference-of-means direction through a quantised two-layer network, and checks by simulation that speculative sampling returns the target distribution. Students write `kv_cache_bytes` and row-wise `quantise_rows` and `dequantise` against tests. The matrices are random and no model is timed.

**Shared-task path:** every team writes the memory line of its plan. Task 7 teams run the precision-transfer check on the smaller ranked model, `CohereLabs/tiny-aya-global`, after reading its size from the model card. For a real timing, use `extensions/hf_text.py` on a declared device: the time it reports includes prefill and excludes model loading, so it is not decode throughput alone. See the [extension guide](../extensions/README.md).

## Practical questions

1. Why does four-bit weight storage not make all inference memory four-bit?
2. Why can a throughput improvement worsen the latency a user sees?
3. What else must you measure after finding a low weight error?
4. A Task 7 vector was found on a 4-bit model. What check comes before submission?

## After class

Complete the [exercise sheet](../exercises/13_efficiency.md) in 45–60 minutes. Read sections 1–3 of the QLoRA paper before the next session (20 minutes). [Quiz 13](../quizzes/13_efficiency.md) opens day 14.

## Readings

**Required**
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023). Sections 1–3: 4-bit storage with 16-bit computation, the set-up most teams will load.

**Memory and speed**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022). Section 3.1: why the attention matrix need not be stored.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023). Sections 3–4: where KV-cache memory is wasted and how paging recovers it.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022). Section 2: the acceptance rule and the expected number of tokens.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et al., 2023). The same rule, found independently; compare the two proofs.

**Compression**
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022). Quantisation that corrects the remaining weights for each rounding error.
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022). Outlier features, the real-model version of the lab's large channel.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015). Section 2: temperature and soft targets.
