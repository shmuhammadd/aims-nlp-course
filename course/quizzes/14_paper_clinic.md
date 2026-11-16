# Quiz 14 — Paper clinic and peer review

[Course home](../../README.md) · [Lecture notes](../lectures/14_paper_clinic.md) · Sat at the start of day 15 · 10 minutes · 10 points · closed book

**1. (1 point)** A team writes its abstract on day 14, before the official evaluation opens. Which sentence can it contain?

- A. "Our system ranks in the top five."
- B. "On the development set our system scores macro-F1 0.58 against 0.55 for the baseline."
- C. "Our system will outperform the baseline on the test set."
- D. "Our system achieves the best results on the task."

**2. (1 point)** Why does a bootstrap interval for macro-F1 recompute the metric on every resample?

- A. Resampling changes the gold labels.
- B. Macro-F1 is built from per-class counts and is not a mean of per-item scores.
- C. Recomputing makes the interval narrower.
- D. McNemar's test requires it.

**3. (1 point)** The 95% paired bootstrap interval for macro-F1 of B minus macro-F1 of A is −0.045 to +0.014. Which sentence does it support?

- A. B is worse than A.
- B. A and B perform equally well.
- C. These items do not separate A and B.
- D. B would win with more data.

**4. (1 point)** Which of these review comments names a reproducibility gap?

- A. "The introduction is too long."
- B. "The method is not new."
- C. "Table 2 does not say which split or seed produced its scores."
- D. "The paper has several typing errors."

**5. (3 points)** A Task 12 team runs its multimodal system twice on the same rows, once with the right painting and once with the decoy. On 6 rows only the right-painting run is correct. On 1 row only the decoy run is correct.

a. Compute the exact one-sided McNemar p-value. Show the sum. (2)
b. Does the run pass the image-use check at p < 0.05? (1)

**6. (3 points)** A draft says: "Our model clearly outperforms the baseline (0.61 against 0.58)." Name three things the sentence, or the sentences after it, must add before a reader can trust the claim. (1 each)
