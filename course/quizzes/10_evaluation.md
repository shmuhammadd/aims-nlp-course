# Quiz 10 — Benchmarking and evaluation

[Course home](../../README.md) · [Lecture notes](../lectures/10_evaluation.md) · Sat at the start of day 11 · 10 minutes · 10 points · closed book

**1. (1 point)** A team scores macro-F1 0.61 on a development split it made itself and compares it with 0.58 on the task leaderboard. Why does the comparison fail?

- A. Macro-F1 cannot be compared across systems.
- B. The data and the protocol differ, so the two numbers come from different benchmarks.
- C. A difference of 0.03 is always too small to matter.
- D. Leaderboard scores are always lower than development scores.

**2. (1 point)** Two classifiers make the same decision on every test item. One has expected calibration error 0.01 and the other 0.13. What differs between them?

- A. Their accuracy.
- B. Their macro-F1.
- C. How closely their stated confidence matches how often they are right.
- D. The number of items they were tested on.

**3. (1 point)** Fifty systems with the same true accuracy are scored on one 200-item test set. The highest score is reported. On fresh items that system will most likely score

- A. higher, because it has been shown to be the best.
- B. the same as on the leaderboard.
- C. lower, because the maximum of fifty noisy scores overstates the true accuracy.
- D. lower, because the fresh items are harder.

**4. (1 point)** A model refuses 0% of adversarial prompts written in Yoruba. Which conclusion do the data support?

- A. The model is unsafe in Yoruba.
- B. The model is safe in Yoruba.
- C. The model was not safety-trained.
- D. None yet: the responses must be labelled, because the model may not have understood the prompts.

**5. (3 points)** A multi-label system is scored on two frames.

| Frame | TP | FP | FN |
|---|---|---|---|
| A | 30 | 10 | 10 |
| B | 5 | 5 | 15 |

a. Compute F1 for each frame. (1)
b. Compute macro-F1. (1)
c. Compute micro-F1. (1)

**6. (3 points)** A reference answer has two tokens and a candidate has three. The cosines, with reference tokens as rows, are

$$
\begin{pmatrix}0.8&0.5&0.2\\0.1&0.6&0.3\end{pmatrix}.
$$

a. Compute BERTScore recall and precision by greedy matching. (2)
b. Give one reason a correct answer with extra words can score below a wrong one-word answer. (1)
