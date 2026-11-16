# Exercise sheet 8 — Vision-language models

[Course home](../../README.md) · [Lecture notes](../lectures/08_vlm.md) · [Notebook](../tutorials/08_vlm.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. Compute the patch counts for 336×336 and 672×336 inputs with 14×14 patches, and the token counts after 2×2 pooling. Give two reasons a real processor's count may differ.
b. On a development set, a system is right with the real image and wrong with the decoy on 9 rows, and the reverse on 2 rows. Compute the one-sided exact McNemar p-value. Does it pass at 0.05?

## 2. Implement (4 points)

In notebook 8:

a. Write `text_only_check(real_labels, decoy_labels)`, which returns the share of rows with the same label and whether it reaches 0.99. Assert that `prior` passes and `parser` fails.
b. Give `weak` a parameter for the probability that it reads the chart. For probabilities 0.05, 0.15, 0.30 and 0.50, report accuracy with the real image, $b$, $c$ and the p-value on the 300 questions.
c. Keep the probability at 0.15 and raise the number of questions to 600 and 1,200. Report the p-value at each size and say how many questions this system needs to pass.

## 3. Investigate (4 points)

Choose one. State which numbers you measured.

**A. Task 12 teams.** Take 100 development rows. Build a decoy for each by pairing its caption with a different painting. Run your current system, or the organisers' baseline, on both sets. Report accuracy with real and decoy paintings, $b$, $c$, the p-value, and the share of rows whose label is unchanged. Say whether the run would pass as multimodal, as text-only, or as neither.

**B. Task 6 and Task 9 teams.** Take 50 development items. Run your current system once with the real image and once with a blank image of the same size. Report the task metric for both, and list five items where the blank-image answer was still right. Say what the question or text gave away in each.

**C. Any team.** Write ten answerable questions and five false-premise questions about the lab's charts, such as "How tall is bar D?". Extend the three toy systems to answer them, score each system on correctness, support and abstention, and report one table. If you have access to a VLM, run the benign-instruction test from the lecture on 20 charts and report the share of one-word answers in each of the three conditions. Do not use harmful prompts.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
