# Exercise sheet 15 — Defence and concept check

[Course home](../../README.md) · [Lecture notes](../lectures/15_capstone.md) · [Notebook](../tutorials/15_capstone.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points and is the last sheet of portfolio 3. Submit your working, your changed notebook, the numbers you measured and the script asked for in part 3.

## 1. Derive (2 points)

A full system and an ablated system are scored on the same 12 items. Both are right on 6 items. Only the full system is right on 4. Only the ablated system is right on 1. Both are wrong on 1.

a. Compute each system's accuracy, the paired difference, and the exact one-sided McNemar p-value that the full system is better.
b. A third system answers 8 of the 12 items and gets 6 right. Compute its coverage, its selective accuracy, and its overall accuracy with abstentions counted as wrong.

## 2. Implement (4 points)

In notebook 15, add a mismatched-policy condition: case $i$ keeps its chart and gold answer and takes the policy of case $(i+1) \bmod 12$. Run the system on the 12 mismatched cases and report accuracy, coverage and the number of answers that changed.

Write the full, decoy-chart and mismatched-policy predictions to one JSONL file with the fields from section 3 of the notebook, one line per item per condition. Add an assertion that every answer that is not an abstention carries both evidence IDs.

Explain in two sentences why the mismatched policy still gives some right answers, and what that says about a decoy test with three options.

## 3. Investigate (4 points)

Use your team's own SemEval results.

a. Write the script of your "one result" slide in 150–250 words: the comparison, the metric and split, the difference with its interval, one failure with its count, and the next experiment.
b. Write the three hardest questions an examiner could ask about that result, and your answer to each in two or three sentences. One answer must name a result that would change your conclusion.
c. Add to `participation.md` the status of your official submission, the next action, its date and the person who does it.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output or in `experiment_log.md`.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
