type: title
title: Benchmarking and evaluation
subtitle: Day 10 · Week 2
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: Model A scores 71.2. Model B scores 70.4. Is A better?

Notes: Take answers for a minute. Push for the questions a reviewer would ask: on how many items, the same items, the same prompt, the same scorer, how many runs were tried before this one. Each question is a slide today.

---
title: Four tasks, four ways of keeping score
subtitle: By the end of today every team can compute its own metric by hand

| Task | Ranked on | Today |
|---|---|---|
| 6 MMCultureQA | BERTScore F1 | Greedy token matching |
| 7 CLaS | Language compliance and relevance | Two scores, one system |
| 9 Framing | Multi-label micro-F1; macro-F1 secondary | Pooled and per-frame counts |
| 12 ArtEmotion-28 | Macro-F1 over languages; log loss in bits; within-painting Pearson | Three subtasks, three metrics |

- Days 6 to 9 built systems for languages, images and speech. Today we ask what their scores mean.
- Today is also the course deadline for your frozen experiment plan

Notes: Metrics are from the task cards in SEMEVAL_TASK.md, read from the task sites on 16 November 2026.

---
type: section
title: Part 1: Benchmarks
subtitle: What they are and how they break

---
title: A benchmark is four things
subtitle: Example: ArtEmotion-28, subtask 1

| Part | In ArtEmotion-28 subtask 1 |
|---|---|
| Task | Painting and a native-speaker caption → one of eight emotions |
| Data | 173,745 annotations of 1,658 paintings |
| Metric | Macro-F1 per language, then the mean over languages |
| Protocol | Two runs, text-only and multimodal; a decoy-image check; no outside annotations of test paintings; Codabench |

- Two systems are comparable only when all four match
- Score on another split, or with another F1 average, and you have run a different benchmark

Notes: Ask each team to fill the same four rows for its own task in two minutes. Most teams cannot fill "protocol" from memory. That is the row that disqualifies submissions.

---
title: A score is a conditional statement
subtitle: It holds for one setting of five things

- One checkpoint
- One prompt template
- One dataset version and split
- One decoding rule: greedy or sampled, temperature, maximum length
- One metric implementation: tokeniser, normalisation, averaging

### What follows
- Write all five beside every number
- Choose them on development data, then freeze them
- Keep the test set out of prompts, retrieval indexes, fine-tuning and model selection

Notes: Give one example from day 3: the same checkpoint with greedy decoding and with temperature 1.0 gives different answers, so different scores.

---
title: Benchmarks break in four ways

| Failure | What happens | Evidence |
|---|---|---|
| Contamination | The test items were in the pretraining data | Models complete test instances from their first words (Golchin and Surdeanu, 2023) |
| Saturation | Scores reach the ceiling; systems still fail simple cases | Dynabench (Kiela et al., 2021) |
| Leaderboard overfitting | Many teams tune on one test set; the top score is partly luck | Next slide |
| Prompt sensitivity | Formatting alone moves the score | Up to 76 accuracy points on LLaMA-2-13B (Sclar et al., 2023) |

Notes: The 76 points is the largest spread Sclar et al. report, from few-shot prompt formatting such as separators and spacing. A contamination check lowers the doubt and rarely proves the data were absent.

---
title: The best of 50 equal systems looks 7 points better
subtitle: Lab: 50 systems, all with true accuracy 0.70, scored on 200 test items
layout: stack

![](figures/10_winner.png)

- The winner shows 0.771 on the leaderboard and 0.699 on fresh items. No system was better.

Notes: Simulation from tutorial 10, section 4, seed 42, 2,000 repeats. One score has a standard error of 0.032, so the winner sits a little over two standard errors up. Recht et al. 2019 rebuilt the ImageNet test set and saw drops of 11 to 14 per cent with the model ranking largely kept; they attribute it to harder images more than to tuning. Either way a test score is one draw.

---
title: African-language benchmarks you can cite

| Benchmark | Task | Coverage |
|---|---|---|
| AfriSenti (2023) | Tweet sentiment | 14 languages, 110,000+ tweets |
| MasakhaNER (2021) | Named entities | 10 languages |
| MasakhaNEWS (2023) | News topic | 16 languages |
| AfriQA (2023) | Cross-lingual retrieval QA | 10 languages |
| SIB-200 (2023) | Topic, on FLORES-200 sentences | 200+ languages |
| IrokoBench (2024) | Inference, maths, multiple choice | 17 languages, human-translated |
| AfriHate (2025) | Hate and abusive language | 15 languages |
| AfroBench (2023) | 22 datasets, 15 tasks | 64 languages |

Notes: Counts are from each paper's abstract; links are in the lecture notes. Native speakers annotated AfriSenti and AfriHate. IrokoBench and SIB-200 are translated from English sources, so the questions carry the source's topics. Ask of any benchmark: who wrote the items, who labelled them, in which variety of the language?

---
type: section
title: Part 2: Scores you can trust
subtitle: Calibration, judges, paired tests

---
title: The Brier score charges for confident errors

$$
\mathrm{Brier} = \frac{1}{N}\sum_{i}(p_i - y_i)^2
$$

| Prediction p | Label y | Squared error |
|---|---|---|
| 0.9 | 1 | 0.01 |
| 0.6 | 0 | 0.36 |
| 0.2 | 0 | 0.04 |

- Brier = 0.41 / 3 = 0.137. Accuracy is 2/3.
- Move the second prediction to 0.51: accuracy stays 2/3, Brier falls to 0.103

Notes: Do it on the board. (0.01 + 0.2601 + 0.04)/3 = 0.103. Accuracy sees only the side of 0.5. A proper score also rewards honest probabilities.

---
title: Same accuracy, different calibration
layout: split

- Calibrated: of the answers given 80% confidence, 80% are right
- ECE: bin by confidence, then average the gap between accuracy and confidence, weighted by bin size
- Lab: two models with the same decisions
  - honest: ECE 0.009
  - overconfident: ECE 0.130
- ECE moves with the number of bins and the sample size. Print the table too.

![](figures/10_reliability.png)

Notes: 5,000 simulated predictions, tutorial section 2. The overconfident model reports sigmoid(2.5 z) where the truth is sigmoid(z). Points under the dotted line are overconfident. When this model says 0.99 or more it is right 91% of the time.

---
title: Abstaining trades coverage for accuracy
layout: split

- Answer only when confidence passes a threshold
- Coverage = answered / total
- Selective accuracy = correct / answered
- Lab, overconfident model:
  - threshold 0.9: coverage 0.55, accuracy 0.836
  - threshold 0.99: coverage 0.22, accuracy 0.908
- Report both numbers. Choose the threshold on development data.

![](figures/10_coverage.png)

Notes: A system reaches perfect accuracy by answering one easy question. The likelihood of a generated sequence is not a calibrated probability that the answer is right.

---
title: A model judge has known biases

| Bias | Check |
|---|---|
| Position: prefers the answer shown first | Show every pair in both orders |
| Verbosity: prefers the longer answer | Compare scores against answer length |
| Self-preference: prefers its own outputs | Use a judge from another model family |
| Weak in the target language | Compare with native-speaker labels on a subset |

- Write the rubric before you see outputs. Hide which system wrote each answer.
- In the lab a simulated judge changes its verdict on 3 of 6 pairs when only the order changes

Notes: Zheng et al. 2023 name position, verbosity and self-enhancement biases. Wang et al. 2023 changed which system won by changing the order of presentation. Task 7 relevance is likely to be judged by a model: check 20 Swahili outputs by hand.

---
title: Two runs on the same items: test the disagreements

- b = items the first run gets right and the second gets wrong
- c = the reverse. Items where both agree carry no information.
- If the runs were equal, b would be a fair coin tossed b + c times

$$
p = P(X \geq b), \qquad X \sim \mathrm{Binomial}(b + c,\ 0.5)
$$

| b | c | Net gain | p |
|---|---|---|---|
| 15 | 5 | 10 items | 0.021 |
| 12 | 8 | 4 items | 0.25 |

Notes: This is the one-sided exact McNemar test. For b = 15, c = 5: sum of C(20, k) for k = 15..20 is 21,700, divided by 2^20. Seeds are a second source of variation: report the spread over seeds as well.

---
type: section
title: Part 3: The four capstone metrics
subtitle: One worked number for each

---
title: Task 6: BERTScore matches tokens greedily
layout: split

- Embed every token of reference x and candidate x̂ with a pretrained encoder
- Recall: each reference token takes its best cosine with a candidate token
- Precision: each candidate token takes its best cosine with a reference token

$$
F = \frac{2PR}{P+R} = \frac{2(0.70)(0.85)}{1.55} = 0.768
$$

- The third candidate token has no match and lowers precision

![](figures/10_greedy.png)

Notes: Zhang et al. 2019, arXiv 1904.09675. Board work: R = (0.9 + 0.8)/2, P = (0.9 + 0.8 + 0.4)/3. Pink marks the best match for each reference token; bold black marks the best for a candidate token.

---
title: On short answers BERTScore can mislead
subtitle: Toy: reference answer "Dodoma", one random vector per word

| Candidate | Precision | Recall | F1 |
|---|---|---|---|
| Dodoma | 1.00 | 1.00 | 1.00 |
| mji wa Dodoma | 0.34 | 1.00 | 0.51 |
| Nairobi | 0.68 | 0.68 | 0.68 |
| ndizi | 0.03 | 0.03 | 0.03 |

- Extra words lower precision. A wrong answer of the right type keeps a high cosine.
- The gaps are the toy's: the two city vectors were built to overlap
- Real scores sit in a narrow band near the top, so read differences, not levels
- Ask the organisers which encoder and settings the scorer uses

Notes: Tutorial section 7. The question is "Mji mkuu wa Tanzania ni upi?" (What is the capital of Tanzania?). "mji wa Dodoma" is "the city of Dodoma"; "ndizi" is "banana". The paper proposes baseline rescaling because raw scores fall in a narrow range. The task page names BERTScore F1 and no settings.

---
title: Task 9: micro-F1 follows the common frames
layout: split

| Frame | TP | FP | FN | F1 |
|---|---|---|---|---|
| A | 40 | 10 | 10 | 0.800 |
| B | 2 | 2 | 6 | 0.333 |
| C | 1 | 0 | 3 | 0.400 |

- Micro: pool the counts. 86 / (86 + 12 + 19) = 0.735
- Macro: mean of the three F1 values = 0.511
- Never predict B or C: micro falls to 0.714, macro to 0.267

![](figures/10_micro_macro.png)

Notes: F1 = 2TP / (2TP + FP + FN). Dropping B and C: TP 40, FP 10, FN 10 + 8 + 4 = 22, so micro = 80/112 = 0.714 and macro = 0.8/3 = 0.267. Task 9 ranks on micro and reports macro as secondary.

---
title: Task 12 subtask 1: every language counts the same

- Compute macro-F1 over the eight emotions for each language, then average the languages
- Example: Swahili 0.58, Hausa 0.52, isiZulu 0.40 gives 0.50, whatever the number of test rows

### The image-use check is a paired test
- Each test row has a decoy: the same caption with another painting
- b = rows right with the real painting and wrong with the decoy; c = the reverse
- The multimodal run must pass a one-sided McNemar test at p < 0.05: b = 15, c = 5 passes
- A text-only run must give real and decoy rows the same label at least 99% of the time

Notes: The three language scores are an invented example. The organisers' baseline is 0.535 multimodal against 0.550 text-only: the multimodal baseline does not beat text. Our reading of b and c is the standard form of the test; read the organisers' scorer for the exact definition.

---
title: Task 12 subtasks 2 and 3 score a predicted share

### Subtask 2: log loss in bits per annotator

$$
\ell = -\,[\,y\log_2 p + (1-y)\log_2(1-p)\,]
$$

- Gold share y = 0.8, prediction p = 0.7: 0.412 + 0.347 = 0.759 bits
- Predicting 0.5 always costs 1 bit. The organisers' baseline scores 0.80.

### Subtask 3: within-painting Pearson correlation
- For each painting, correlate predicted and gold shares across languages, then average
- Level and scale do not count. Predicting 0.5y + 0.05 scores r = 1.

Notes: The floor for y = 0.8 is the entropy, 0.722 bits. In the lab a system with the right level per painting and guessed language differences scores r = 0.21. A constant prediction per painting has no defined correlation: check the scorer.

---
title: Task 7: two scores, and both must be high
subtitle: Invented rows, to show the arithmetic

| System | Compliance | Relevance | Harmonic mean |
|---|---|---|---|
| Answers in the source language | 0.05 | 0.95 | 0.10 |
| One fixed Swahili sentence | 1.00 | 0.02 | 0.04 |
| Steered, α chosen on dev | 0.90 | 0.60 | 0.72 |

- Compliance: language identification says the answer is in the target language
- Relevance: the answer addresses the question
- The task page does not publish how the two combine. The course uses the harmonic mean.

Notes: The rows are invented to show the arithmetic: 2(0.9)(0.6)/1.5 = 0.72. Day 11 measures this trade-off in a toy. Language identification is weak on two- or three-word answers.

---
title: A refusal rate hides three other outcomes
subtitle: Invented counts per 100 adversarial prompts, to show how to read the table

| Outcome | English | Yoruba |
|---|---|---|
| Refused | 69 | 0 |
| Misunderstood | 1 | 80 |
| Harmful | 10 | 8 |
| Safe | 20 | 12 |

- Attack success rate = harmful / all prompts: 10% and 8% here
- By refusal rate Yoruba looks unsafe. The table says the model did not understand the prompts.
- Reported: refusal up to 69% in English, as low as 0% in Yoruba (arXiv 2610.39590)
- Over-refusal is the share of **benign** prompts refused. Measure it on a separate set (XSTest).

Notes: Only the two refusal rates come from the Yoruba and Nigerian Pidgin red-teaming study, whose authors say the models failed to understand the prompts. The other six cells are invented so that each column sums to 100. Native speakers assign the labels. HarmBench gives a fixed protocol for automated red teaming.

---
title: Report how well the labellers agree

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- Two native speakers label 50 responses as harmful or not. They agree on 40: observed agreement 0.8.
- The first calls 30 harmful, the second 25
- Chance agreement = 0.6 × 0.5 + 0.4 × 0.5 = 0.5
- κ = (0.8 − 0.5) / 0.5 = 0.6
- Validate a model judge the same way: κ between the judge and the native-speaker labels
- Keep prompt and response text out of the repository. Commit labels and IDs.

Notes: Cohen's kappa. The example is invented. Reuse prompt IDs across translations so the gap to English is paired. safety_eval.py in the extensions builds the outcome table with bootstrap intervals.

---
title: What goes in the paper besides the score

### Model card (Mitchell et al., 2018)
- Intended use, training data, who the system was evaluated on, limits

### Data statement (Bender and Friedman, 2018)
- Who wrote the text, who annotated it, which language varieties

### For every result
- The five conditions of the score, the number of items, an interval
- Per-language scores beside the average
- Failed runs and unparsed outputs, counted under a rule you wrote in advance

---
title: Today you freeze the plan
subtitle: One dated file, committed to your repository

- Research question; task, subtask and languages
- Development split, metric and scorer version
- The baseline you reproduced, with its score
- One change and two ablations
- Seeds, compute limit, and the rule for choosing between runs

### After the freeze
- Bugs get fixed and experiments run. A change of plan is logged with a date and a reason.
- The test set is for the official submission only
- A result picked after many development runs is reported as exploratory

Notes: The list follows the team addendum in SEMEVAL_TASK.md. Walk round during the lab and read each team's file.

---
type: big
title: Your Task 6 answer scores BERTScore F1 0.85. Is it right?

Notes: It may not be. A wrong answer of the right type scores high, and raw scores sit in a narrow band. Good responses: compare with the score of a deliberately wrong answer on the same question; read 20 outputs with a speaker of the language; report exact match beside BERTScore.

---
title: Lab: 40 minutes

- Open `course/tutorials/10_evaluation.ipynb`. NumPy only.
- Sections 1 to 8, together: Brier, reliability, a judge audit, the 50-system leaderboard, McNemar, micro and macro F1, greedy matching, within-painting correlation
  - Predict each output before you run the cell
- Section 9, in pairs: write two functions until the tests pass
  - `micro_macro_f1(Y_true, Y_pred)`
  - `log_loss_bits(p, y)`
- Then run your organisers' scorer on your baseline's development predictions and put the number in your plan

---
title: Readings

### Before the next session (20 minutes)
- Zhang et al., 2019. BERTScore. Sections 1 to 3. Or the metric section of your own task page.

### Benchmarks
- Muhammad et al., 2023. AfriSenti.
- Adelani et al., 2024. IrokoBench.
- Sclar et al., 2023. Sensitivity to prompt formatting.

### Reporting
- Guo et al., 2017. On Calibration of Modern Neural Networks.
- Zheng et al., 2023. Judging LLM-as-a-Judge.
- Mazeika et al., 2024. HarmBench.

Notes: Links and the full list are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- The metric your task is ranked on, and one output it would score too generously
- One number you measured in the lab
- One line from your frozen plan: the rule for choosing between runs

Quiz 10 opens the next session: six questions, ten minutes.
