type: title
title: The causal Transformer
subtitle: Day 2 · Week 1
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: You change the last word of a sentence. Which of the model's predictions are allowed to move?

Notes: Take answers for one minute. The answer is only the prediction made at the last position. Every earlier prediction must stay the same to the last decimal. By the end of today students can test this in four lines.

---
title: Where today sits
subtitle: Yesterday we chose the tokens. Today we build what reads them.

| Day | What we do | What it uses from today |
|---|---|---|
| 1 | Tokens, loss, baselines | |
| **2** | **The decoder block: attention, masks, shapes** | |
| 3 | Pretraining and decoding | The shifted loss, the parameter count N |
| 4 | LoRA | The attention matrices that LoRA adapts |
| 11 | Steering (Task 7) | The residual stream, read and edited between layers |
| 13 | Efficiency | The key-value cache and grouped-query attention |

- You built a small language model in the DeepMind path. Today we open it and count every shape.

Notes: Quiz 1 has just been sat. Keep this slide to one minute.

---
type: section
title: Part 1: The block
subtitle: A running sum, and one way to mix positions

---
title: The residual stream is a running sum
subtitle: The pre-norm decoder block

$$
H = X + \mathrm{Attention}(\mathrm{Norm}(X))
$$
$$
Y = H + \mathrm{MLP}(\mathrm{Norm}(H))
$$

- X has one row per token: shape (T, d)
- Attention moves information **between** positions. The MLP works on each position alone.
- Each sublayer reads a normalised copy and **adds** its result to the stream
- After the last block: one more Norm, then a d × V matrix gives the logits
- Day 11 stops between two blocks, reads this stream and writes a changed one back

Notes: Draw one horizontal line for the stream and hang the attention and MLP boxes under it, each with an arrow down and an arrow back up into a plus sign. Keep the drawing on the board for day 11.

---
title: Norm controls the scale of what each sublayer reads
subtitle: LayerNorm and RMSNorm, applied to each row of the stream

$$
\mathrm{LayerNorm}(x) = g \cdot \frac{x - \mathrm{mean}(x)}{\sqrt{\mathrm{var}(x) + \epsilon}} + b
\qquad
\mathrm{RMSNorm}(x) = g \cdot \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}}
$$

- LayerNorm subtracts the mean and divides by the standard deviation: 2d parameters
- RMSNorm only divides by the root mean square: d parameters, one less pass over the vector
- **Pre-norm** puts the Norm inside the branch, so the stream itself is never rescaled
- The 2017 Transformer normalised after the addition. Deep stacks train more easily with pre-norm.

Notes: Xiong et al. 2020 analyse why post-norm needs learning-rate warm-up and pre-norm does not. RMSNorm is Zhang and Sennrich 2019. Llama models use RMSNorm.

---
title: Attention is a weighted average chosen by content
subtitle: One head, one sequence

$$
Q = XW_Q \qquad K = XW_K \qquad V = XW_V
$$
$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) \qquad O = AV
$$

- Row i of QKᵀ scores every key against query i
- The softmax runs along each row, so each row of A sums to 1
- Row i of O is an average of the value rows, weighted by row i of A
- M is the mask: 0 where attention is allowed, −∞ where it is forbidden

Notes: Say "query asks, key advertises, value is what gets copied". Then drop the metaphor and stay with the matrices.

---
title: Worked example: attention for two tokens

Take Q = K = [[2, 0], [1, 1]] and V = [[1, 0], [0, 2]], so d_k = 2.

| Step | Row 0 | Row 1 |
|---|---|---|
| QKᵀ | [4, 2] | [2, 2] |
| Divide by √2 | [2.83, 1.41] | [1.41, 1.41] |
| Softmax, no mask | [0.80, 0.20] | [0.50, 0.50] |
| Output O = AV, no mask | [0.80, 0.39] | [0.50, 1.00] |
| Softmax, causal mask | [1, 0] | [0.50, 0.50] |
| Output, causal mask | [1, 0] | [0.50, 1.00] |

- The mask changes row 0 only. Row 1 is the last position and may see everything.

Notes: Do it on the board. The row-0 scores differ by 1.414, and e^1.414 = 4.11, so the weights are 4.11/5.11 = 0.804 and 0.196. Output row 0 is 0.804·[1,0] + 0.196·[0,2]. Notebook section 1 checks every number.

---
title: Why divide by the square root of the head width?
layout: split

- Suppose the entries of q and k are independent with mean 0 and variance 1
- Then q·k is a sum of d_k terms and has variance d_k
- Wide heads give large scores, and a softmax of large scores puts all its weight on one key
- A saturated softmax passes almost no gradient
- Dividing by √d_k brings the variance back to 1 at every width

![](figures/02_scaling.png)

Notes: The figure draws 500 random queries and 16 random keys at each width, seed 42. Without scaling the largest weight climbs from 0.43 at width 4 to 0.95 at width 1,024. With scaling it stays near 0.24.

---
title: Two masks, two jobs

| | Causal mask | Padding mask |
|---|---|---|
| Hides | Keys after the query | Keys that are padding tokens |
| Shape | (T, T), the same for every sequence | (B, 1, 1, T), one per sequence |
| Why | Position t must not read tokens it will predict | Short sequences are padded to the batch length |

```
causal  = np.tril(np.ones((T, T), dtype=bool))       # (T, T)
not_pad = np.arange(T)[None, :] < lengths[:, None]   # (B, T)
allowed = causal & not_pad[:, None, None, :]         # (B, 1, T, T)
scores  = np.where(allowed, scores, -np.inf)
```

- Also remove padded positions from the loss. The mask does not do that for you.

Notes: A common bug is to use the padding mask and forget the causal one, or the reverse. The lab's test cell builds both.

---
title: The mask goes in before the softmax
layout: split

- Set forbidden scores to −∞, then take the softmax
- The allowed weights then sum to 1 on their own
- Zeroing weights **after** the softmax leaves rows that sum to less than 1
  - early rows shrink the most, so the output scale depends on position
- Row 0 is always [1, 0, 0, …]: the first token can attend only to itself
- Subtract the row maximum before exponentiating to avoid overflow

![](figures/02_attention.png)

Notes: The figure is the attention matrix from notebook section 2: random weights, seed 42, six tokens. With the 2×2 example, zeroing after the softmax leaves row 0 summing to 0.80.

---
type: section
title: Part 2: Shapes and counts
subtitle: B sequences, T tokens, width d, H heads

---
title: Trace the shapes from token IDs to logits

| Step | Operation | Shape |
|---|---|---|
| Token IDs | | (B, T) |
| Embed | look up rows of a V × d table | (B, T, d) |
| Project | X W_Q, X W_K, X W_V | (B, T, d) each |
| Split heads | reshape, then swap two axes | (B, H, T, d/H) |
| Scores | Q Kᵀ / √(d/H), add mask | (B, H, T, T) |
| Mix | softmax, then A V | (B, H, T, d/H) |
| Merge heads | swap back, reshape, × W_O | (B, T, d) |
| MLP | d → d_ff → d | (B, T, d) |
| Logits | Norm, then × a d × V matrix | (B, T, V) |

- Memory for the scores grows with T². Everything else grows with T.

Notes: Ask the class for each shape before revealing the row. With B = 2, T = 6, d = 8, H = 4 the scores are (2, 4, 6, 6), which is what the lab prints.

---
title: Heads are a reshape
subtitle: No new kind of layer: the same three matrices, cut into H slices

```
B, T, d = x.shape
dh = d // H                                    # head width
# split: (B, T, d) -> (B, T, H, dh) -> (B, H, T, dh)
q = (x @ Wq).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
# ... same for k and v; scores and softmax per head ...
# merge: (B, H, T, dh) -> (B, T, H, dh) -> (B, T, d)
out = mixed.transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo
```

- Each head has its own attention pattern over the same tokens
- The scale is the head width d/H, not d
- You write the middle in the lab. Assert the shape after every line.

Notes: Swap the axes before merging. A reshape straight from (B, H, T, dh) to (B, T, d) runs without error and mixes heads with positions. Insist on the shape comments.

---
title: Count the parameters of one block
subtitle: GPT-2 small: d = 768, d_ff = 3,072, biases everywhere

| Part | Formula | Parameters |
|---|---|---|
| Attention: W_Q, W_K, W_V, W_O | 4d² + 4d | 2,362,368 |
| MLP: two matrices | 2·d·d_ff + d_ff + d | 4,722,432 |
| Two LayerNorms | 4d | 3,072 |
| **One block** | | **7,087,872** |

- 12 blocks: 85,054,464
- Token table 50,257 × 768, position table 1,024 × 768, final norm: 39,385,344
- Total: **124,439,808**, the "124M" of GPT-2 small
- The number of heads does not appear. Heads cut the same matrices into slices.

Notes: 4 × 768² = 2,359,296, plus 3,072 biases. 2 × 768 × 3,072 = 4,718,592, plus 3,840 biases. The output layer reuses the token table, so it adds nothing. The lab's second exercise is this count as a function.

---
title: Three changes in today's decoders

| Change | What it replaces | Effect |
|---|---|---|
| RMSNorm | LayerNorm | Half the norm parameters, less arithmetic |
| SwiGLU MLP | ReLU or GELU MLP | A gated activation with three matrices in place of two |
| Grouped-query attention | One key and value head per query head | Several query heads share one key-value head |

- SwiGLU with d_ff = 8d/3 has 3 · d · 8d/3 = 8d² weights, the same as a two-matrix MLP with d_ff = 4d
- Grouped-query attention keeps every query head. It shrinks W_K, W_V and the key-value cache.
- Read these in a model's `config.json` before you download the weights

Notes: SwiGLU computes (Swish(xW1) ⊙ xW3) W2. GLU variants are from Shazeer 2020; grouped-query attention from Ainslie et al. 2023. Day 13 returns to the key-value cache.

---
title: Count a Llama-3.1-8B block the same way
subtitle: d = 4,096, 32 query heads, 8 key-value heads, MLP width 14,336, no biases

| Part | Formula | Parameters |
|---|---|---|
| W_Q and W_O | 2 × 4,096² | 33,554,432 |
| W_K and W_V | 2 × 4,096 × (8 × 128) | 8,388,608 |
| SwiGLU MLP | 3 × 4,096 × 14,336 | 176,160,768 |
| Two RMSNorms | 2 × 4,096 | 8,192 |
| **One block** | | **218,112,000** |

- 32 blocks: 6.98 billion. Two embedding tables of 128,256 × 4,096: 1.05 billion.
- Total: **8.03 billion**. Task 7 teams load this model on day 11.

Notes: The configuration values are in the model's config.json. 8 key-value heads of width 128 make W_K and W_V 4,096 × 1,024. The input and output embedding tables are separate in this model.

---
title: The MLP holds most of a block
layout: stack

![](figures/02_params.png)

- Attention gets the attention. The MLP gets two thirds of a GPT-2 block and four fifths of a Llama block.
- LoRA on day 4 usually adapts attention matrices, the smaller part of the block

Notes: Shares computed from the two tables. The norm parameters are too few to see.

---
title: Position: add a vector, or rotate the query and key
layout: split

- Attention without position treats the input as a set: shuffle the tokens and the outputs shuffle with them
- **Absolute**: add a learned vector for position t to the embedding. Positions beyond the table do not exist.
- **RoPE**: rotate each pair of coordinates of q and k by an angle proportional to the position
  - the score then depends on the distance m − n
  - no parameters, applied in every layer
- RoPE does not promise good behaviour at lengths the model never trained on

![](figures/02_rope.png)

Notes: The figure uses one random query and key of width 64, seed 42, with the query at positions 40, 100 and 400. The three curves lie on top of each other. Notebook section 5 prints the same fact for width 8.

---
type: section
title: Part 3: Train and test
subtitle: Every position at once, and a test that needs no data

---
title: Teacher forcing trains every position in one pass
subtitle: Inputs are the sequence. Targets are the sequence shifted by one.

| Position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Input | `<bos>` | Darasa | linaanza | leo |
| The model may read | `<bos>` | `<bos>` Darasa | `<bos>` Darasa linaanza | all four |
| Target | Darasa | linaanza | leo | `<eos>` |

- The causal mask gives position t its prefix and nothing more
- So one forward pass yields T training examples, and none of them sees its answer
- The loss is the mean of −log p(target) over positions, with padding left out
- An untrained model should start near ln V. A first loss far below that means a leak.

Notes: "Darasa linaanza leo" is yesterday's Swahili sentence, the class starts today, shown here as whole words for readability. In the lab the first loss is 2.77 against ln 12 = 2.48.

---
title: Generation adds one token at a time

- Run the model on the prompt and take the logits at the **last** position
- Choose a token from them. Tomorrow covers how: greedy, temperature, top-p.
- Append it and run again
- The keys and values of earlier positions do not change, because of the causal mask
  - so store them: the **key-value cache**
  - each new token then costs one position of computation, not T
- Training is parallel over positions. Generation is a loop. That is why generating is slow.

Notes: The cache holds two tensors per layer of shape (B, kv heads, T, d/H). Grouped-query attention cuts it by the ratio of query heads to key-value heads: 4 times for Llama-3.1-8B.

---
title: The causal-invariance test
layout: split

```
logits = model(seq)
alt = seq.copy()
alt[cut:] = other_tokens
logits_alt = model(alt)
assert np.allclose(logits[:cut],
                   logits_alt[:cut])
```

- Change the tokens from position `cut` on. Earlier logits must not move at all.
- Needs no trained weights and no labels
- Run it for every `cut`, and again after any change to the attention code

![](figures/02_leak.png)

Notes: Notebook sections 3 and 4. With the mask the largest change is exactly 0 at every cut. Without it the early logits move by 0.08 to 0.44 in this random model.

---
title: An attention map is a picture, not an explanation

- A large weight says a position was averaged in. It does not say what was copied or whether later layers used it.
- The MLP and every later layer can undo or ignore what one head moved
- To show that a model **uses** something, change it and watch the output
- That is day 11: find a direction in the residual stream, add to it or remove it, and measure the answer
- The stream you will edit there is the X and H of today's first equation

Notes: Day 11 makes the same point about probes. Keep the two-line block equation on the board and point at it.

---
type: big
title: By mistake your mask is all zeros. Training loss falls fast. How do you find out?

Notes: An additive mask of zeros blocks nothing, so every position reads its own target and the loss falls far below what the data allow. The loss curve looks like success. The invariance test fails at once. A second clue is generation: the model has learned to copy the next token, which is not there at inference, so samples are poor.

---
title: Lab: 40 minutes

- Open `course/tutorials/02_transformers.ipynb`. NumPy only, random weights.
- Sections 1 to 5, together: two-token attention, one block, the invariance test, the shifted loss, RoPE
  - Do section 1 on paper before you run it
- Section 6, in pairs: write two functions until the tests pass
  - `multi_head_attention` with a shape assertion on every tensor
  - `block_params`, which must return 7,087,872 for a GPT-2 small block
- Teams: find `config.json` for the model you plan to use and read off d, layers, heads and MLP width

---
title: Readings

### Before the next session (20 minutes)
- Vaswani et al., 2017. Attention Is All You Need. Sections 3.1 to 3.3.

### The modern block
- Su et al., 2021. RoFormer: rotary position embedding.
- Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
- Ainslie et al., 2023. GQA: grouped-query attention.
- Touvron et al., 2023. LLaMA. The architecture section: pre-norm, SwiGLU and rotary embeddings.

### Going deeper
- Xiong et al., 2020. On Layer Normalization in the Transformer Architecture.
- Shazeer, 2020. GLU Variants Improve Transformer.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- The shape of the attention scores for B = 4, T = 10, H = 2
- One bug the invariance test catches, and one it does not
- The width and layer count of the model your team plans to use

Quiz 2 opens the next session: six questions, ten minutes.
