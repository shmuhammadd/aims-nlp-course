# Exercise sheet 10 — Benchmarking and evaluation

[Course home](../../README.md) · [Lecture notes](../lectures/10_evaluation.md) · [Notebook](../tutorials/10_evaluation.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. For $p=[0.8,0.7,0.4,0.1]$ and $y=[1,0,1,0]$, compute the Brier score, the accuracy, and the coverage and selective accuracy at confidence threshold 0.75, where confidence is $\max(p,1-p)$.

b. A reference has three tokens and a candidate has two. The cosines, with reference tokens as rows, are

$$
S=\begin{pmatrix}0.9&0.1\\0.2&0.7\\0.5&0.3\end{pmatrix}.
$$

Compute BERTScore recall, precision and F1 by greedy matching.

## 2. Implement (4 points)

In notebook 10, write `reliability_table(conf, correct, bins)`. It splits confidence from 0.5 to 1 into `bins` equal bins and returns one row per bin: lower edge, upper edge, count, accuracy and mean confidence. An empty bin has count 0 and no accuracy. Also return ECE, computed over the non-empty bins.

Run it on the overconfident model from section 2 with 5, 10 and 20 bins, once on all 5,000 predictions and once on the first 100. Report the six ECE values in one table. Add an assertion that the bin counts sum to the number of predictions.

## 3. Investigate (4 points)

Use your team's own task and its development data.

a. Run the organisers' scorer on your reproduced baseline's development predictions and record the score.

b. Build three systems that need no model and score them with the same scorer. Choose ones that fit your task: the most frequent label for every item; a constant share of 0.5; an empty answer; the question copied back as the answer; the same fixed sentence for every question.

c. Put the four scores in one table. Say which of the three the metric treats most generously and why, with reference to how the metric is computed. Name one second number you would report beside the official metric to expose that system.

If your baseline predictions are not ready, do parts b and c and say so.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output or a scorer log.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
