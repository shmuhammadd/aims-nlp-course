# Exercise sheet 11 — Interpretability and cross-lingual steering

[Course home](../../README.md) · [Lecture notes](../lectures/11_steering.md) · [Notebook](../tutorials/11_steering.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

Let $h=(3,1)$, the source mean $\mu_s=(2,0)$ and the target mean $\mu_t=(0,2)$.

a. Compute $v$, $\hat v$ and the projection of $h$ on $\hat v$.
b. Compute $h'$ after adding with $\alpha=1$, after clamping to the target mean's projection, and after ablation. Give the projection of each on $\hat v$.

## 2. Implement (4 points)

In notebook 11, add norm-scaled steering: $h'=h+\alpha\,\lVert h\rVert\,\hat v$, so that $\alpha$ is a fraction of each activation's own length. Sweep $\alpha$ over 0, 0.1, 0.2, 0.4, 0.8 on the held-out English activations. Report compliance, relevance and their harmonic mean in one table beside additive steering and clamping. Add an assertion that $\alpha=0$ returns the input unchanged.

## 3. Investigate (4 points)

The lab's language direction leans 0.6 towards topic 0. Change that overlap to 0, 0.3, 0.6 and 0.9, keeping the seed. For each value, find the largest $\alpha$ at which relevance stays above 0.95 under additive steering. Plot or tabulate the result and explain it in terms of the geometry. Then state one way you could detect this kind of overlap in a real model, where you do not know the topic directions.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
