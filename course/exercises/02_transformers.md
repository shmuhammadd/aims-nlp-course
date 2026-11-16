# Exercise sheet 2 — The causal Transformer

[Course home](../../README.md) · [Lecture notes](../lectures/02_transformers.md) · [Notebook](../tutorials/02_transformers.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. Let $Q=K=I_2$ and $V=\begin{pmatrix}1&0\\0&2\end{pmatrix}$, with $d_k=2$. Compute the attention weights and the output for both rows, without a mask and with a causal mask.
b. A pre-norm block has $d=512$, MLP width 2,048, biases on every matrix and two LayerNorms. Count its parameters, part by part.

## 2. Implement (4 points)

In notebook 2, build a decoder with two blocks and two heads from your `multi_head_attention`. Use the notebook's `norm`, `emb`, `pos` and `head`, and draw the second block's weights from `rng`.

- It takes a batch of token IDs of shape (B, T) and a vector of lengths, and returns logits of shape (B, T, V).
- Assert causal invariance at every cut point from 1 to T − 1.
- Assert that changing a padding token leaves the logits of the real tokens unchanged.
- Then break the mask on purpose with `np.tril(np.ones((T, T), dtype=bool), 1)` and report which assertion fails first.

## 3. Investigate (4 points)

Count a real model before you download it. Use the model your team plans to run for its SemEval task. If you have not chosen one, use `meta-llama/Llama-3.1-8B-Instruct` (Task 7) or the text encoder of your task's baseline.

- Open the model's `config.json` on its model page. Record the width, the number of layers, the number of heads, the MLP width, the vocabulary size and the kind of norm.
- Count the parameters of one block and of the whole model, showing the formula for each part. Compare your total with the published parameter count and explain any gap above 1%.
- Compute the memory of the weights at 16 bits and at 4 bits per weight. Say whether each fits on a 15 GB GPU.

Report the three results in one table. State which numbers you read from the configuration and which you computed.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
