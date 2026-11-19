# Exercise sheet 3 — Pretraining and decoding

[Course home](../../README.md) · [Lecture notes](../lectures/03_pretraining.md) · [Notebook](../tutorials/03_pretraining.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. For one row of logits $z$ with $p=\mathrm{softmax}(z)$ and loss $L=-\log p_y$, derive $\partial L/\partial z_j$ and show that the gradient sums to zero over $j$. Check your formula on $p=(0.6, 0.3, 0.1)$ with $y=2$.
b. A team wants to continue pretraining an 8.03-billion-parameter model on 500 million tokens of Hausa and Swahili text. Estimate the compute with $6ND$, and the time at a sustained 10 TFLOP/s. Name two costs the estimate leaves out.

## 2. Implement (4 points)

In notebook 3, write `generate(method, seed)` for the trained bigram model `W`. It returns one string of at most 40 characters and supports four methods: greedy, temperature 0.7, temperature 1.0, and top-p 0.9 with your `top_p_filter`.

- Draw 200 strings per sampling method, with seeds 0 to 199.
- Report in one table, per method: the share of strings that reach the end symbol, the mean length, the share that are exactly one of the four training sentences, and the mean log-probability per character under the model.
- Add an assertion that temperature 0.01 gives the same string as greedy.

Say which method you would use to show a visitor "what the model writes", and why.

## 3. Investigate (4 points)

Use your team's shared task and its baseline.

- Name the pretrained model the baseline uses, its pretraining objective (causal, masked or denoising) and its parameter count. Give the source of each fact.
- Tasks 6 and 7: find the generation settings the baseline uses (sampling on or off, temperature, top-p, top-k, beams, maximum new tokens, repetition penalty). List each value and where you found it. If a value comes from a library default, say so. Tasks 9 and 12: say how the baseline turns model outputs into labels, and where a threshold or an argmax is applied.
- Contamination: list every public source in which your task's evaluation items, or their labels, could already appear. Say what your team will do to keep them out of training, and quote the task rule that applies if there is one.

Report the answers in one table. Mark anything you could not find as "not found" and say where you looked.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
