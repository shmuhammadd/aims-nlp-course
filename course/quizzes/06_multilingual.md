# Quiz 6 — Multilingual and low-resource modelling; building data for African languages

[Course home](../../README.md) · [Lecture notes](../lectures/06_multilingual.md) · Sat at the start of day 7 · 10 minutes · 10 points · closed book

**1. (1 point)** Two systems have the same overall accuracy on a test set of 100 English and 10 Hausa items. Which statement must be true?

- A. They have the same Hausa accuracy.
- B. They have the same equal-language accuracy.
- C. They got the same number of items right in total.
- D. They make the same errors.

**2. (1 point)** Languages are sampled with probability $q_l=n_l^\alpha/\sum_j n_j^\alpha$. What does $\alpha=0$ do?

- A. It samples only the largest language.
- B. It samples every language equally often, so examples of the smallest language repeat most.
- C. It samples in proportion to corpus size.
- D. It removes repeated examples.

**3. (1 point)** Your test items come from 40 speakers, about 25 items each. How should you bootstrap, and what happens to the interval compared with resampling items?

- A. Resample items; the interval is the same.
- B. Resample speakers; the interval is usually wider.
- C. Resample speakers; the interval is usually narrower.
- D. Resample classes; the interval is wider.

**4. (1 point)** Two annotators agree on 92% of posts, and Cohen's kappa is 0.29. Which explanation fits?

- A. The annotators disagree on most posts.
- B. Kappa was computed on too few items.
- C. One class is rare, so most of the agreement is expected by chance.
- D. Kappa cannot be used with two classes.

**5. (3 points)** Two annotators label 100 tweets as positive or not. Both say positive for 20 and both say not for 50. A alone says positive for 10, and B alone for 20.

a. Compute the observed agreement $p_o$. (1)
b. Compute the expected agreement $p_e$. (1)
c. Compute $\kappa$ to two decimal places. (1)

**6. (3 points)** A system gets 160 of 200 English items, 30 of 50 Swahili items and 10 of 50 Yoruba items right.

a. Compute overall accuracy. (1)
b. Compute equal-language accuracy. (1)
c. Which would you report in a paper about a system for all three languages, and what must sit beside it? (1)
