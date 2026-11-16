# Quiz 13 — Efficient inference and quantisation

[Course home](../../README.md) · [Lecture notes](../lectures/13_efficiency.md) · Sat at the start of day 14 · 10 minutes · 10 points · closed book

**1. (1 point)** A model has 4 billion weights. About how much memory do its weights need at 4 bits each, before overhead?

- A. 0.5 GB
- B. 2 GB
- C. 4 GB
- D. 16 GB

**2. (1 point)** A team quantises a model's weights from 16 bits to 4 bits and keeps 16-bit computation. What happens to the KV cache for a 4,096-token prompt?

- A. It shrinks by a factor of four.
- B. It disappears, because quantised models need no cache.
- C. It stays the same size.
- D. It grows by a factor of four.

**3. (1 point)** Speculative decoding accepts a drafted token with probability $\min(1, p/q)$ and resamples from the leftover on rejection. Compared with sampling from the target model alone, the distribution of the output is

- A. the draft model's distribution.
- B. an average of the draft and target distributions.
- C. the target model's distribution.
- D. the target's distribution at a lower temperature.

**4. (1 point)** A weight matrix has one channel 30 times larger than the rest. It is quantised at 4 bits with one scale for the whole matrix. What happens to the other channels?

- A. They are stored exactly.
- B. Most of their weights round to zero.
- C. They are clipped to the largest integer.
- D. They use more bits than the large channel.

**5. (3 points)** A model has $L=16$ layers and $H_{kv}=4$ key/value heads of dimension $d_h=64$. The cache holds 16-bit numbers.

a. Compute the KV-cache memory for a batch of 4 at 2,048 tokens, in MiB. (2)
b. The batch is halved and the context is doubled. What is the cache now? (1)

**6. (3 points)** Quantise the channel $(0.6, -2.1, 1.0)$ symmetrically at 4 bits.

a. Give the scale and the three stored integers. (2)
b. Give the largest reconstruction error and the bound it must respect. (1)
