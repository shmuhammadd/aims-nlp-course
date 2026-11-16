# Exercise sheet 14 — Paper clinic and peer review

[Course home](../../README.md) · [Lecture notes](../lectures/14_paper_clinic.md) · [Notebook](../tutorials/14_paper_clinic.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and the paragraph asked for in part 3. Your 300–500-word peer review is handed in separately tonight.

## 1. Derive (2 points)

Eight items have gold labels 1 1 1 0 0 0 0 0. System A predicts 1 1 0 0 0 0 0 1. System B predicts 1 1 1 1 1 0 0 0.

a. Compute accuracy and macro-F1 for each system. Show TP, FP and FN for each class.
b. One bootstrap draw picks items 1 1 2 4 5 6 7 7. Compute macro-F1 for A and for B on that draw, and the difference B − A. Compare it with the difference on the full set.

## 2. Implement (4 points)

Task 12 ranks macro-F1 computed per language and then averaged. In notebook 14, give every item a language with `lang = np.random.default_rng(14).integers(0, 4, n_items)`.

Write `macro_f1_by_language(y_true, y_pred, lang)`, which returns the mean over languages of that language's macro-F1. Then write a stratified paired bootstrap that resamples items within each language, so that every resample keeps each language's item count. Use 1,000 resamples and `np.random.default_rng(0)`.

Report the per-language macro-F1 of both systems, the two averaged scores, and the 95% interval for B − A. Add an assertion that with one language `macro_f1_by_language` equals `macro_f1`.

## 3. Investigate (4 points)

Use your team's own development predictions for the baseline and your main system, on the same items.

a. Compute your task's metric for both systems and a paired bootstrap interval for the difference with 1,000 resamples. Task 7 teams bootstrap per-item compliance and relevance. Say what unit you resampled and why.
b. Build an error table for your main system: at least three categories, with a count for each, covering at least 30 errors or all the errors if there are fewer.
c. Write the results paragraph for your paper in four moves (comparison, size with interval, one mechanism, one alternative explanation), in 150–250 words. Give the run IDs from `experiment_log.md` for every number.

If your team has no saved predictions for two systems yet, say so, do parts a to c on the notebook's synthetic data, and add the command you will run to save predictions. The mark for part 3 is then capped at 3.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
