# Exercise sheet 13 — Efficient inference and quantisation

[Course home](../../README.md) · [Lecture notes](../lectures/13_efficiency.md) · [Notebook](../tutorials/13_efficiency.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. Compute the KV-cache memory in GiB for $B=2$, $T=8192$, $L=32$, $H_{kv}=4$, $d_h=128$ and $s=2$ bytes.

b. Quantise the channel $(1.0, -0.35, 2.1)$ symmetrically at 4 bits. Give the scale, the stored integers, the reconstructed weights and the largest error. Check the error against the bound $s/2$.

## 2. Implement (4 points)

Section 4 of notebook 13 stores each 4-bit integer in a whole byte. Write `pack4(q)` and `unpack4(packed, shape)` so that two integers share one byte: add 8 to each integer to make it non-negative, then put one in the high four bits and the next in the low four bits.

Apply them to the 4-bit per-channel quantisation of the lab's matrix `W`. Assert that `unpack4(pack4(q), q.shape)` equals `q`. Report the packed bytes, the bytes of float32 scales, and their sum beside the lab's "ideal packed bytes". State what your functions do when the number of integers is odd.

## 3. Investigate (4 points)

Use the model your team plans to run for its own task.

a. Write the memory line of your plan. Give the parameter count from the model card, the precision, and the weight memory from $Nb/8$. Find the layers, KV heads and head dimension in the model's configuration, measure or estimate the longest development input in tokens, and compute the KV cache with your `kv_cache_bytes`.

b. Say whether the model fits your hardware under the course's 3 GB rule. If you can load the model, record the peak memory the library reports and compare it with your estimate. If you cannot, say so and leave the comparison out.

c. Write the timing protocol you would use for one result in your paper: prompt lengths, batch sizes, output length, warm-up, synchronisation, what you record, and which quality metric is scored on the same run.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output, a log or a calculation you show.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
