type: title
title: Efficient inference and quantisation
subtitle: Day 13 · Week 3
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: The model needs 16.1 GB. Your GPU has 15 GB. What are your options?

Notes: Take answers for a minute. Expect: a smaller model, fewer bits per weight, a CPU, a paid GPU. Each has a price, and today puts numbers on the first two. Keep the list on the board.

---
title: Task 7 hands every team this problem
subtitle: Days 11 and 12 chose a method. Today asks what it costs to run.

| Fact | Number |
|---|---|
| Ranked model | `Llama-3.1-8B-Instruct`, 8.03 billion weights |
| Weights at 16 bits | 8.03 × 2 = 16.1 GB |
| Free Colab T4 | 15 GB |
| Weights at 4 bits | 4.0 GB; about 5.5 GB loaded, with overhead |

- You prototype at 4 bits. The organisers run your code at full precision.
- A steering vector found at 4 bits may not transfer. You have to measure it.
- Tasks 6, 9 and 12: the same arithmetic decides which vision-language model you can fine-tune

Notes: Model and submission facts are from the Task 7 card. The second ranked model, tiny-aya-global, is smaller; read its size from the model card.

---
type: section
title: Part 1: Memory
subtitle: What has to fit

---
title: Inference memory has four parts

| Part | Grows with | Day-13 tool |
|---|---|---|
| Weights | Parameters × bits per weight | Quantisation, a smaller model |
| KV cache | Batch × tokens × layers × KV heads | Shorter context, shared heads |
| Activations and workspace | Batch × tokens × width | Smaller batch |
| Runtime overhead | The library | Measure it |

- Training adds gradients and optimiser state, which is why LoRA on day 4 trained few weights
- Quantising the weights to 4 bits leaves the other three rows as they were

Notes: Ask: why does 4-bit weight storage not make all inference memory 4-bit? The cache and activations usually stay at 16 bits, and scales and unquantised layers add to the weights.

---
title: Weight memory is parameters × bits / 8

$$
M_{\mathrm{weights}} = \frac{N\,b}{8}\ \mathrm{bytes}
$$

| Llama-3.1-8B, N = 8.03 billion | Weights |
|---|---|
| 32-bit | 32.1 GB |
| 16-bit | 16.1 GB |
| 8-bit | 8.0 GB |
| 4-bit | 4.0 GB |

- This is the ideal figure. Scales, metadata and layers left unquantised add to it.
- 1 billion weights at 4 bits is 500 MB

Notes: 8.03 × 10^9 × 16 / 8 = 16.06 × 10^9 bytes. GB here is 10^9 bytes; the cache slides use GiB, 2^30 bytes. Say so once.

---
title: Halving the bits halves the weights
subtitle: Ideal weight memory from N × b / 8, nominal model sizes
layout: stack

![](figures/13_weights.png)

- Llama-3.1-8B crosses the 15 GB line only at 16 bits, before any cache or overhead is added

Notes: Computed in tutorial 13, section 1. Sizes are nominal, from the model names, except Llama's 8.03 billion.

---
title: What fits a free Colab GPU or a laptop
subtitle: Course rule: keep 3 GB free on the T4 and 4 GB free on the laptop

| Model | 16-bit | 8-bit | 4-bit | T4, 15 GB | Laptop, 8 GB |
|---|---|---|---|---|---|
| SmolVLM-500M | 1.0 | 0.5 | 0.25 | Any | Any |
| Gemma 3 1B | 2.0 | 1.0 | 0.5 | Any | 16-bit or lower |
| SmolVLM 2.2B | 4.4 | 2.2 | 1.1 | Any | 8-bit or 4-bit |
| Gemma 3 4B | 8.0 | 4.0 | 2.0 | 16-bit or lower | 4-bit |
| Llama-3.1-8B | 16.1 | 8.0 | 4.0 | 8-bit or 4-bit | No |
| Gemma 3 12B | 24.0 | 12.0 | 6.0 | 4-bit | No |

- Weights in GB. "Fits" says nothing about speed.

Notes: From the notebook's table. A CPU runs 32-bit floats by default, which doubles the 16-bit column; low-bit CPU inference needs a runtime with kernels for it, such as llama.cpp. The 3 GB and 4 GB allowances are our assumption: have teams change them for their own machine.

---
title: The KV cache: keys and values for every past token

$$
M_{KV} = 2 \cdot B \cdot T \cdot L \cdot H_{kv} \cdot d_h \cdot s
$$

- B batch, T tokens, L layers, H_kv key/value heads of dimension d_h, s bytes per number
- The 2 counts keys and values

### Llama-3.1-8B: L = 32, H_kv = 8, d_h = 128, s = 2 bytes, T = 4,096

$$
2 \cdot 1 \cdot 4096 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 2^{29}\ \mathrm{bytes} = 0.5\ \mathrm{GiB}
$$

- A batch of 8 needs 4 GiB

Notes: Do the powers of two on the board: 2 · 2^12 · 2^5 · 2^3 · 2^7 · 2 = 2^29. The model has 32 query heads sharing 8 KV heads (grouped-query attention); the configuration is on the model's Hugging Face page.

---
title: The cache grows with context, and can pass the weights
layout: split

- Double the context, double the cache
- At 16,384 tokens: 2 GiB
- At the full 131,072-token context: 16 GiB, as large as the 16-bit weights
- Sharing 8 KV heads among 32 query heads cuts the cache by four
- In common 4-bit set-ups the cache stays at 16 bits
- Cap the context and the batch in your plan, and write both down

![](figures/13_kv.png)

Notes: Tutorial section 2. Task 7 prompts are single questions, so the cache is small there. Retrieval systems from day 12 fill the context and pay this cost.

---
type: section
title: Part 2: Time
subtitle: Where the seconds go

---
title: Prefill and decode are different workloads

| | Prefill | Decode |
|---|---|---|
| Does | Reads the whole prompt in parallel | Writes one token per step |
| Fills or reads | Fills the KV cache | Reads the whole cache each step |
| Report as | Time to first token | Output tokens per second |

- A timing needs: device, batch size, prompt length, output length, a warm-up run
- On a GPU, synchronise before reading the clock
- A larger batch raises throughput and makes each request wait longer

Notes: Ask: why can a throughput improvement worsen the latency a user sees? Requests wait for the batch to fill and share the device while it runs.

---
title: Two systems ideas, and what each changes

| | FlashAttention | Paged attention |
|---|---|---|
| Problem | The T × T attention matrix is written to GPU memory | Each sequence reserves one long block for its cache |
| Change | Compute attention in blocks that stay in fast on-chip memory | Store the cache in small fixed blocks, allocated as needed |
| Output | Exact: the same numbers | Exact: the same numbers |
| Helps | Long sequences | Serving many sequences of different lengths |

- Neither is an approximation, and neither promises a fixed speedup on every device

Notes: Dao et al. 2022; Kwon et al. 2023 (vLLM). We give no speedup figures: they depend on hardware and workload, and the course has measured none.

---
title: Speculative decoding: draft cheaply, verify in one pass

- A small draft model proposes γ tokens. The target model scores all of them in one parallel pass.
- Accept a drafted token x with probability min(1, p(x) / q(x)): p from the target, q from the draft
- At the first rejection, resample from max(p − q, 0), renormalised
- The output is distributed exactly as the target model's

$$
\mathrm{tokens\ per\ target\ pass} = \frac{1-\alpha^{\gamma+1}}{1-\alpha} = \frac{1-0.8^{5}}{0.2} = 3.36
$$

- Draft says 0.8, target says 0.6: accept with probability 0.75

Notes: Leviathan et al. 2022. α is the acceptance rate, here 0.8, with γ = 4 drafted tokens. 3.36 tokens per target pass is not a speedup: the draft model's passes cost time too.

---
title: The speculative output matches the target
layout: split

- Lab: 200,000 draws over a five-token vocabulary
- Every proposal comes from the draft q
- 75% are accepted, the overlap of p and q
- The final tokens follow p to within 0.001
- A draft that agrees with the target more often is accepted more often

![](figures/13_spec.png)

Notes: Tutorial section 6, seed 42. Overlap = sum of min(p, q) = 0.30 + 0.25 + 0.10 + 0.07 + 0.03 = 0.75.

---
type: section
title: Part 3: Compression
subtitle: Fewer bits, smaller models, and what to check

---
title: Symmetric quantisation: a scale and small integers

$$
s_c = \frac{\max|W_c|}{2^{b-1}-1} \qquad q = \mathrm{round}(W / s_c) \qquad \hat{W} = s_c\, q
$$

| | 0.52 | −1.4 | 0.71 |
|---|---|---|---|
| Divide by s = 1.4 / 7 = 0.2 | 2.6 | −7 | 3.55 |
| Round: the stored integers | 3 | −7 | 4 |
| Multiply by s | 0.6 | −1.4 | 0.8 |

- 4 bits: integers from −7 to 7. Every error is at most s / 2 = 0.1.
- A channel of zeros would give s = 0. Set its scale to 1.

Notes: Board work before the table. c indexes the channel; one scale per channel is "per-channel". Errors here are 0.08, 0 and 0.09.

---
title: One large channel ruins a shared scale
layout: split

- Lab: a 64 × 32 matrix; one output channel is 30 times larger
- One scale for the matrix, 4 bits: every weight in the other 31 channels rounds to zero
- One scale per channel: weight error falls from 1.20 to 0.23
- At 8 bits: from 0.0205 to 0.0006
- Cost: one stored scale per channel
- Weight error is not task accuracy. Evaluate the outputs.

![](figures/13_quant.png)

Notes: Tutorial section 3. In the hand example: share a scale with a channel whose largest weight is 30 and s = 4.29, so 0.52, −1.4 and 0.71 all round to 0. LLM.int8 found such outlier dimensions in real models.

---
title: Loading a 4-bit model is a few lines

```
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=config, device_map="auto")
```

- Weights are stored in 4 bits. Computation, activations and the cache run in 16 bits.
- NF4 spaces its 16 levels unevenly (QLoRA). The idea of a few levels and a scale is the same.
- A sketch: it has not been run in this course. It needs a GPU and access to the gated weights.

Notes: Dettmers et al. 2023. Request Llama access early, as on day 11. Record the library versions with every result.

---
title: Distillation trains a small model on a large one's outputs

$$
L = T^2\ \mathrm{KL}\left(\sigma(z_{\mathrm{teacher}}/T)\ \|\ \sigma(z_{\mathrm{student}}/T)\right)
$$

- σ is the softmax. T is a temperature that softens both distributions.
- Teacher logits (2, 1, 0):
  - T = 1 gives (0.665, 0.245, 0.090)
  - T = 2 gives (0.506, 0.307, 0.186)
- The soft targets tell the student which wrong answers the teacher finds plausible
- The student inherits the teacher's errors. Check the teacher's licence before training on its outputs.

Notes: Hinton et al. 2015; the loss is usually mixed with the ordinary cross-entropy on labels. For a capstone team, distilling a large model's labels into a small classifier is a legitimate system if the task rules allow the large model.

---
title: A 4-bit steering vector may not transfer. Measure it.
layout: split

- Toy: a two-layer network, direction by difference of means
- Cosine with the full-precision direction: 0.988 at 4 bits, 0.528 at 2 bits

### The check for Task 7
- Take a model that loads at 4 and at 16 bits
- Compute the vector at both; report the cosine
- Apply the 4-bit vector to the 16-bit model
- Score compliance and relevance on the same development questions

![](figures/13_direction.png)

Notes: Tutorial section 5. Two layers of 128 units; a 32-layer model can behave differently, which is the reason for the check. tiny-aya-global is the natural model for it if it fits at both precisions.

---
title: Deployment is an experiment

- Choose an operating point on three axes: quality, latency, memory
- Measure on inputs like the real ones: long prompts, and every language you will serve
- Record for each row:
  - model, precision, library versions, device
  - batch size, prompt length, output length
  - the task metric, peak memory, time to first token, tokens per second
- Today's notebook computes memory and reconstruction error. It claims no speedup.

Notes: Ask: you measured a low weight error; what else is needed? Task quality, per-language quality, real memory and run time on the target device.

---
type: big
title: The 4-bit model matches the 16-bit model's score on English. Can you use it for Hausa?

Notes: Not yet. An English score says nothing about Hausa: a language with less training data may sit on weights that quantisation disturbs more, or may not. Nobody in the room knows until the per-language scores are measured. The answer we want is the experiment: both precisions, each language, same items, a paired test from day 10.

---
title: Lab: 40 minutes

- Open `course/tutorials/13_efficiency.ipynb`. NumPy only.
- Sections 1 to 6, together: the fit table, the KV cache, quantisation, storage, a direction under quantisation, speculative sampling
  - Predict each output before you run the cell
- Section 7, in pairs: write the functions until the tests pass
  - `kv_cache_bytes(B, T, L, H_kv, d_h, bytes_per)`
  - `quantise_rows(W, bits)` and `dequantise(q, scale)`, with an all-zero row handled
- Then write the memory line of your team's plan

---
title: Readings

### Before the next session (20 minutes)
- Dettmers et al., 2023. QLoRA. Sections 1 to 3.

### Memory and speed
- Dao et al., 2022. FlashAttention.
- Kwon et al., 2023. PagedAttention (vLLM).
- Leviathan et al., 2022. Speculative Decoding.

### Compression
- Frantar et al., 2022. GPTQ.
- Dettmers et al., 2022. LLM.int8().
- Hinton et al., 2015. Distilling the Knowledge in a Neural Network.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- The memory line of your plan: model, precision, weight memory, longest context, cache
- One number you measured in the lab
- One thing the arithmetic leaves out

Quiz 13 opens the next session: six questions, ten minutes.
