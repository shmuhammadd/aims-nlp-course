# Quiz 3 — Pretraining and decoding

[Course home](../../README.md) · [Lecture notes](../lectures/03_pretraining.md) · Sat at the start of day 4 · 10 minutes · 10 points · closed book

**1. (1 point)** You need a model that reads a whole sentence and outputs one label. Which pretraining objective gives each token context from both sides?

- A. Causal language modelling.
- B. Masked language modelling.
- C. Greedy decoding.
- D. Beam search.

**2. (1 point)** Estimate the training compute for a model with $10^9$ parameters on $10^{10}$ tokens.

- A. $6\times10^{9}$ operations
- B. $10^{19}$ operations
- C. $6\times10^{19}$ operations
- D. $6\times10^{90}$ operations

**3. (1 point)** As the temperature goes to 0, sampling becomes

- A. uniform over the vocabulary.
- B. greedy decoding.
- C. beam search with width 2.
- D. top-p with p = 1.

**4. (1 point)** The model is nearly certain of the next token at one position and very unsure at the next. Which method keeps a different number of candidate tokens at the two positions?

- A. Top-k with k = 40.
- B. Greedy.
- C. Top-p with p = 0.9.
- D. Temperature 0.7.

**5. (3 points)** A model gives four tokens the probabilities 0.4, 0.3, 0.2 and 0.1.

a. Give the distribution after top-k with $k=2$. (1)
b. Give the distribution after top-p with $p=0.75$. (1)
c. Give the distribution at temperature 0.5. Fractions are fine. (1)

**6. (3 points)** A bigram model predicts $p=(0.6, 0.3, 0.1)$ for the next token. The true next token is the third one.

a. Write the gradient of the loss with respect to the three logits. (1)
b. What do its entries sum to, and why? (1)
c. During training, held-out loss is lowest at step 60 and rises until step 100. Which weights do you keep, and on which data do you report the final score? (1)
