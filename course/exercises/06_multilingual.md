# Exercise sheet 06 — Multilingual and low-resource modelling; building data for African languages

[Course home](../../README.md) · [Lecture notes](../lectures/06_multilingual.md) · [Notebook](../tutorials/06_multilingual.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show. The optional extension is not needed for full marks.

## 1. Derive (2 points)

a. A system gets 45/50, 8/20 and 3/5 items right in three languages. Compute overall accuracy and equal-language accuracy.
b. Two annotators label 100 posts. Both say "offensive" for 30 and both say "not offensive" for 55. A alone says "offensive" for 10, and B alone for 5. Compute $p_o$, $p_e$ and $\kappa$.

## 2. Implement (4 points)

Add this cell to notebook 06, after your `macro_f1_by_language` passes its test. It simulates two systems on three languages; the language codes are labels on simulated data.

```python
rng = np.random.default_rng(42)
sizes = {"ha": 300, "ig": 100, "yo": 30}
acc_a = {"ha": .80, "ig": .70, "yo": .50}; acc_b = {"ha": .78, "ig": .72, "yo": .70}
lang = np.concatenate([[l] * n for l, n in sizes.items()])
y = rng.integers(0, 3, size=len(lang))
def predict(acc):
    wrong = rng.random(len(lang)) > np.array([acc[l] for l in lang])
    return np.where(wrong, (y + rng.integers(1, 3, size=len(lang))) % 3, y)
pred_a = predict(acc_a); pred_b = predict(acc_b)
```

Report macro-F1 per language for both systems and the mean over languages. Then compute a paired bootstrap 95% interval, with 2,000 resamples, for the difference in the mean over languages. Resample items within each language, with the same indices for both systems. State the rule your code applies when a resample of the 30 "yo" items contains no item of some class. Say whether you would claim that B is better, and for which languages.

## 3. Investigate (4 points)

Use your team's shared task data.

a. For each language your team has chosen (three at most), tabulate the number of training and development items per class, or per track if your task has no classes. Name the smallest cell. Compute the score of a system that always predicts the largest class, with the task's own metric.

b. Run a pilot annotation. Two team members each label the same 30 development items without seeing the gold labels or each other's labels. Compute Cohen's kappa between the two of you, and between each of you and the gold labels. Read your disagreements together and write down the two commonest causes. If your task has no categorical labels, judge 30 system or reference outputs on one yes-or-no question that your task's scoring depends on.

c. Write half a page of data statement for the data you used: language varieties, who wrote and labelled it, the licence, and one limit that matters for your system.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
