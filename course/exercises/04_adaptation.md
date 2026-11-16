# Exercise sheet 04 — Post-training I: prompting, instruction tuning and LoRA

[Course home](../../README.md) · [Lecture notes](../lectures/04_adaptation.md) · [Notebook](../tutorials/04_adaptation.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show. The optional model extension is not needed for full marks.

## 1. Derive (2 points)

Let $Y=X(W+sAB)$ with targets $T$ and loss $L=\frac{1}{n\,d_{out}}\sum (Y-T)^2$.

a. Derive $\partial L/\partial A$ and $\partial L/\partial B$. Show that with $B=0$ the first is zero and the second is not.
b. An adapter of rank 16 sits on a 768×3072 matrix. Count its parameters and give them as a percentage of the matrix.

## 2. Implement (4 points)

In notebook 04 the true change has rank 2. Keep it at rank 2 and train adapters of rank 1, 2, 4 and 8, for seeds 42, 43 and 44. Report held-out MSE and the adapter's parameter count in one table, with the frozen base as the first row. Assert in every run that `Wbase` is unchanged. Say which rank you would choose and why.

## 3. Investigate (4 points)

a. In section 4 of the notebook, score all 24 orderings for $\gamma$ = 1, 0.8, 0.5 and 0.2. Tabulate the minimum, mean and maximum test accuracy. For $\gamma=0.5$, list the label sequences of the three best and three worst orderings and state the pattern.

b. For your team's SemEval task, write the few-shot prompt you would try first, using four demonstrations taken from the training split. Give the label list, the demonstrations in order, the rule that maps a reply to a label, and what you do with a reply that fits no label. Then take one of your examples, split it into tokens by hand (words will do), and mark which positions would carry loss in response-only fine-tuning.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
