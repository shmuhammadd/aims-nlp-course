# Quiz 2 — The causal Transformer

[Course home](../../README.md) · [Lecture notes](../lectures/02_transformers.md) · Sat at the start of day 3 · 10 minutes · 10 points · closed book

**1. (1 point)** A batch has B = 3 sequences of T = 5 tokens. The model has width d = 16 and H = 4 heads. What is the shape of the attention scores?

- A. (3, 5, 16)
- B. (3, 4, 5, 4)
- C. (3, 4, 5, 5)
- D. (3, 5, 5)

**2. (1 point)** A decoder uses a causal mask. What are the attention weights of the first position?

- A. Uniform over all T positions.
- B. 1 on itself and 0 elsewhere, whatever the weights of the model.
- C. They depend on the query and key matrices.
- D. 0 everywhere.

**3. (1 point)** A model has 32 query heads and 8 key-value heads (grouped-query attention). Compared with 32 key-value heads, what shrinks?

- A. The number of attention patterns per layer.
- B. The matrices $W_K$ and $W_V$, and the key-value cache.
- C. The MLP.
- D. The vocabulary.

**4. (1 point)** A new decoder with a vocabulary of 1,000 tokens and random weights reports a loss of 0.9 after a few training steps. Note that $\ln 1000\approx6.9$. What is the most likely cause?

- A. The learning rate is too low.
- B. The model is very good.
- C. Positions can see the tokens they are meant to predict.
- D. The vocabulary is too large.

**5. (3 points)** Two tokens have value rows $v_0=(2,0)$ and $v_1=(0,4)$. After scaling, the scores of query 1 are $\ln 3$ for key 0 and $0$ for key 1.

a. Compute the attention weights of query 1. (1)
b. Compute the output row for query 1. (1)
c. With a causal mask, give the weights and the output row for query 0. (1)

**6. (3 points)** A pre-norm block has width $d=100$, MLP width 400, no biases, and two LayerNorms with a gain and a bias each.

a. Count its parameters, part by part. (2)
b. The block changes from 4 heads to 10 heads. How does the count change? (1)
