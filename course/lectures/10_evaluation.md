# Lecture 10 — Benchmarking and evaluation

[Course home](../../README.md) · Week 2, day 5 · 120 minutes · [Slides](../slides/10_evaluation.pptx) · [Notebook](../tutorials/10_evaluation.ipynb) · [Exercises](../exercises/10_evaluation.md) · [Quiz](../quizzes/10_evaluation.md)

## Learning outcomes

- Describe a benchmark by its task, data, metric and protocol, and name four ways a benchmark stops measuring what it claims.
- Compute the metric your capstone task is ranked on, by hand on a small example and in code.
- Report a result with calibration, a paired comparison and an outcome table, and freeze an experiment plan before the test set is touched.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 9 (speech) |
| 10–35 | Slides 1–9: what a benchmark is, how benchmarks break, African-language benchmarks |
| 35–60 | Slides 10–22: calibration, judges, paired tests, the four capstone metrics; BERTScore and micro/macro F1 on the board |
| 60–65 | Break |
| 65–105 | [Notebook 10](../tutorials/10_evaluation.ipynb): sections 1–8 together, section 9 in pairs |
| 105–115 | Slides 23–27: safety evaluation, what goes in the paper, the plan freeze; discuss the BERTScore question |
| 115–120 | Exit ticket: your task's metric, one output it scores too generously, one line of your frozen plan |

## Teaching notes

### What a benchmark is
A benchmark has four parts: a task, a dataset, a metric and a protocol. ArtEmotion-28 subtask 1 shows all four. The task maps a painting and a native-speaker caption to one of eight emotions. The data are 173,745 annotations of 1,658 paintings. The metric is macro-F1, computed per language and then averaged. The protocol asks for a text-only run and a multimodal run, checks image use with decoy paintings, bans annotations of the evaluation paintings from any other source, and takes submissions on Codabench.

Two systems are comparable only when all four parts match. A team that scores on a different split, or with a different F1 average, has run a different benchmark.

### A score is a conditional statement
A score describes one checkpoint, one prompt template, one dataset version, one decoding rule and one metric implementation. Change any of the five and the comparison may change. Freeze a protocol chosen on development data before final testing. Keep the test set out of prompt examples, retrieval indexes, fine-tuning and model selection.

Choose the metric by task: macro-F1 for imbalanced classification; exact match with a declared normalisation for short answers; retrieval metrics for evidence selection; word and character error rates for speech; a human rubric for open-ended answers. An average hides errors. Report counts, an interval and the slices that matter, such as each language.

### How benchmarks break
**Contamination.** A public test set may have entered a model's pretraining data. Golchin and Surdeanu (2023) prompt a model with the dataset name, the split and the start of a test instance, and flag the instance when the model completes it nearly word for word. A contamination check lowers the doubt. It rarely proves the data were absent.

**Saturation.** Models reach the ceiling of a benchmark while still failing on simple challenge examples; Kiela et al. (2021) built Dynabench in response. A saturated benchmark no longer separates systems.

**Leaderboard overfitting.** Many teams tune against one test set, and the top score is partly luck. In the lab, 50 systems with the same true accuracy of 0.70 are scored on 200 items. The best of them shows 0.771 on the leaderboard and 0.699 on fresh items. Recht et al. (2019) rebuilt the ImageNet test set and saw accuracy fall by 11% to 14%, with the ranking of models largely kept; they attribute the drop to slightly harder images more than to tuning on the test set. Both effects say the same thing to a team: a test score is one draw.

**Prompt sensitivity.** Sclar et al. (2023) changed only the formatting of few-shot prompts, such as separators and spacing, and measured differences of up to 76 accuracy points on LLaMA-2-13B. Report the prompt, and report a range over several formats when the system is prompted.

### African-language benchmarks
| Benchmark | Task | Coverage |
|---|---|---|
| AfriSenti (Muhammad et al., 2023) | Tweet sentiment | 14 languages, over 110,000 tweets |
| MasakhaNER (Adelani et al., 2021) | Named entity recognition | 10 languages |
| MasakhaNEWS (Adelani et al., 2023) | News topic classification | 16 languages |
| AfriQA (Ogundepo et al., 2023) | Cross-lingual open-retrieval question answering | 10 languages, over 12,000 examples |
| SIB-200 (Adelani et al., 2023) | Topic classification on FLORES-200 sentences | Over 200 languages and dialects |
| IrokoBench (Adelani et al., 2024) | Inference, maths and multiple-choice questions, human-translated | 17 languages |
| AfriHate (Muhammad et al., 2025) | Hate speech and abusive language | 15 languages |
| AfroBench (Ojo et al., 2023) | A collection of 22 datasets over 15 tasks | 64 languages |

Native speakers annotated AfriSenti and AfriHate. IrokoBench and SIB-200 are translated from English sources, so their questions carry the source's topics and culture. Ask of any benchmark who wrote the items, who labelled them and in which variety of the language.

### Calibration and abstention
For a binary prediction with probability $p$ and label $y$, the Brier score is $N^{-1}\sum_i(p_i-y_i)^2$. In multiclass settings, sum across classes and state the convention. Calibration asks whether events given probability $p$ happen about $p$ of the time. Expected calibration error (ECE) bins the predictions by confidence and averages the gap between accuracy and confidence, weighted by bin size. ECE depends on the binning and the sample size, so print the reliability table beside it and report a proper score such as Brier.

Two models with the same decisions can differ in calibration. In the lab both have accuracy 0.732. The honest one has ECE 0.009. The overconfident one has ECE 0.130, and when it says 0.99 or more it is right 91% of the time.

Selective prediction answers only when confidence passes a threshold. Coverage is answered over total. Selective accuracy is correct over answered. Report both: a system reaches perfect accuracy by answering one easy question. Choose the threshold on development data. The likelihood of a generated sequence is not a calibrated probability that the answer is right.

### Judging generated outputs
Model judges show position bias, verbosity bias and a preference for their own outputs (Zheng et al., 2023). Wang et al. (2023) changed the order in which two answers were shown and changed which system the judge ranked first. Randomise the order, write an explicit rubric, hide which system wrote each answer, and compare the judge with a human-labelled subset. Measure agreement and read the disagreements. A model that writes both the answers and the rubric is marking its own work. Word-overlap metrics penalise correct paraphrases. Embedding metrics can reward fluent text that is wrong.

### Paired comparisons
Two systems scored on the same items give paired data. Test the items on which they disagree. Let $b$ count items where the first is right and the second wrong, and $c$ the reverse. Under the hypothesis that the systems are equal, $b$ is Binomial$(b+c, \tfrac12)$. The one-sided exact McNemar p-value is $P(X\ge b)$. With $b=15$ and $c=5$ it is $21{,}700/2^{20}=0.021$. With $b=12$ and $c=8$ it is 0.25.

Seeds and test items are separate sources of variation: report the spread over seeds, and resample items for the interval. Count failed runs and outputs that could not be parsed under a rule stated in advance.

### The four capstone metrics
**Task 6, BERTScore F1** (Zhang et al., 2019). Embed each token of the reference $x$ and the candidate $\hat x$ with a pretrained encoder and normalise the vectors. Match greedily:

$$
R=\frac{1}{|x|}\sum_{x_i\in x}\max_{\hat x_j\in\hat x}x_i^\top\hat x_j,\qquad
P=\frac{1}{|\hat x|}\sum_{\hat x_j\in\hat x}\max_{x_i\in x}x_i^\top\hat x_j,\qquad
F=\frac{2PR}{P+R}.
$$

It rewards a paraphrase that word overlap would miss. On short answers it has three weaknesses. Every extra token must find a match in a one- or two-token reference, so filler words lower precision. A wrong answer of the right type, such as another city, keeps a high cosine with the reference. And raw scores sit in a narrow band near the top of the range, which is why the paper offers baseline rescaling. The score also depends on the encoder, the layer and whether inverse-document-frequency weights are on. The task page names the metric only; ask the organisers for the settings and use their scorer.

**Task 9, multi-label micro-F1, with macro-F1 secondary.** Each of the 14 frames is a yes/no decision per paragraph. Micro-F1 pools true positives, false positives and false negatives over frames and computes one F1. Macro-F1 computes F1 per frame and averages. Micro follows the common frames. Macro gives a rare frame the same weight as a common one.

**Task 12, subtask 1: macro-F1 per language, then the mean over languages.** A language with few test rows counts as much as a large one. The image-use check is the McNemar test above, on real and decoy rows: $b$ counts rows right with the real painting and wrong with the decoy. A text-only run must give real and decoy rows the same label at least 99% of the time.

**Task 12, subtasks 2 and 3.** Subtask 2 scores a predicted share $p$ of positive annotators against the gold share $y$ with log loss in bits per annotator, $-[y\log_2p+(1-y)\log_2(1-p)]$. Predicting 0.5 costs exactly 1 bit whatever the gold share. The organisers' baseline scores 0.80. Subtask 3 computes, for each painting, the Pearson correlation between predicted and gold shares across languages, then averages over paintings. Correlation ignores level and scale, so the score rewards knowing how languages differ on the same painting.

**Task 7, compliance and relevance.** Compliance is the share of answers that language identification places in the target language. Relevance is the share that answer the question. The task page does not publish how the two combine; the course uses a harmonic mean.

### Responsible evidence
A model card (Mitchell et al., 2018) records intended use, data provenance, the populations evaluated, limits and operating constraints. A data statement (Bender and Friedman, 2018) describes who produced and who annotated the text, and which language varieties it covers. For fairness, ask who uses the system and which harms matter: equal scores on one aggregate metric do not establish fairness. In low-resource settings the label definitions themselves may need local interpretation.

### Safety evaluation
Attack success rate (ASR) is the share of adversarial prompts that draw a harmful answer. Refusal rate is the share refused. Over-refusal is the share of benign prompts refused (XSTest). A refusal rate alone can mislead. In the Yoruba and Nigerian Pidgin red-teaming study, refusal fell from up to 69% in English to as low as 0% in Yoruba, and the authors say the models failed to understand the prompts. In the Hausa study, the base model looked safe by refusal rate because it produced little usable Hausa.

Label each response as refused, misunderstood, harmful or safe. Native speakers assign the labels, and Cohen's kappa on a double-labelled subset reports their agreement: $\kappa=(p_o-p_e)/(1-p_e)$, where $p_o$ is observed agreement and $p_e$ is the agreement expected by chance from each annotator's label rates. Reuse prompt IDs across translations so the gap to English is paired, and resample prompts for the interval, as in session 6. Validate any LLM judge against the native-speaker labels before using it in a low-resource language. `extensions/safety_eval.py` builds this table from a CSV of labels. HarmBench shows a fixed protocol for automated red teaming. See the [safety thread](../SAFETY.md).

### Freezing the experiment plan
Day 10 is the course deadline for a frozen plan. Freezing means the team commits one dated file to its repository that fixes:

- the research question, the task, subtask and languages;
- the development split, the metric and the version of the scorer;
- the baseline already reproduced, with its score;
- the one change proposed and the two ablations;
- the seeds, the compute limit, and the rule for choosing between runs.

After the freeze, the team still fixes bugs and still runs the experiments. A change to the plan goes into the same file as a dated deviation with its reason. The test set is used for the official submission only. A result chosen after looking at many development runs is reported as exploratory.

## Worked example

**Brier and coverage.** For $p=[0.9,0.6,0.2]$ and $y=[1,0,0]$, Brier $=(0.01+0.36+0.04)/3=0.137$. At confidence threshold 0.8, with confidence $\max(p,1-p)$, two cases are answered and both are right: coverage 2/3, selective accuracy 1.0.

**BERTScore.** A reference has two tokens and a candidate three. The cosines are:

| | $\hat x_1$ | $\hat x_2$ | $\hat x_3$ |
|---|---|---|---|
| $x_1$ | **0.9** | 0.2 | 0.1 |
| $x_2$ | 0.3 | **0.8** | 0.4 |

Recall takes the best match in each row: $(0.9+0.8)/2=0.85$. Precision takes the best in each column: $(0.9+0.8+0.4)/3=0.70$. $F=2(0.70)(0.85)/1.55=0.768$. The third candidate token has no good match and pulls precision down.

**Micro and macro F1.** Three frames with (TP, FP, FN) of (40, 10, 10), (2, 2, 6) and (1, 0, 3) have F1 of 0.800, 0.333 and 0.400. Macro-F1 is their mean, 0.511. Pooled counts are (43, 12, 19), so micro-F1 is $86/(86+31)=0.735$.

**Macro-F1 over languages.** Suppose a system's macro-F1 is 0.58 on Swahili, 0.52 on Hausa and 0.40 on isiZulu. The ranked score is their mean, 0.50, however many test rows each language has.

**Log loss.** Gold share 0.8, prediction 0.7: $-(0.8\log_2 0.7+0.2\log_2 0.3)=0.412+0.347=0.759$ bits. Predicting the gold share itself gives the floor, 0.722 bits.

**Compliance and relevance.** Of 100 answers, 90 are in Swahili and 60 answer the question. Compliance is 0.90, relevance 0.60, and their harmonic mean is $2(0.9)(0.6)/1.5=0.72$.

**Cohen's kappa.** Two annotators label 50 responses as harmful or not and agree on 40, so $p_o=0.8$. The first calls 30 harmful and the second 25, so $p_e=0.6\times0.5+0.4\times0.5=0.5$. $\kappa=(0.8-0.5)/0.5=0.6$.

## Guided tutorial

**Question:** What does each score reward, and when does it flatter a system?

The notebook computes a Brier score, reliability bins and coverage on ten invented predictions, then on 5,000 simulated ones from an honest and an overconfident model. It audits a simulated judge for order effects, simulates a leaderboard of 50 equal systems, runs the exact McNemar test, and computes micro and macro F1, a toy greedy-matching score on random word vectors, and within-painting correlation. Students then write `micro_macro_f1` and `log_loss_bits` against tests. All data are invented or simulated and say nothing about a real model.

**Shared-task path:** each team runs its organisers' scorer on the development predictions of its reproduced baseline and writes the number into the frozen plan. The safety extension, `extensions/safety_eval.py`, builds the outcome table from a CSV of labels; see the [extension guide](../extensions/README.md).

## Practical questions

1. Can two equally accurate models differ in calibration?
2. Why must an abstention threshold be chosen on development data?
3. What would reveal position bias in a model judge?
4. A Task 6 answer scores BERTScore F1 0.85. What else do you need to know before calling it right?

## After class

Complete the [exercise sheet](../exercises/10_evaluation.md) in 45–60 minutes. Commit your team's frozen plan today. Read sections 1–3 of the BERTScore paper, or the metric section of your own task's page, before the next session (20 minutes). [Quiz 10](../quizzes/10_evaluation.md) opens day 11.

## Readings

**Required**
- [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675) (Zhang et al., 2019). Sections 1–3: greedy matching, and why raw scores need rescaling. Task 6 teams read all of it.

**Benchmarks**
- [AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages](https://arxiv.org/abs/2302.08956) (Muhammad et al., 2023). How a 14-language benchmark was collected and annotated by native speakers.
- [IrokoBench](https://arxiv.org/abs/2406.03368) (Adelani et al., 2024). Human-translated tests for LLMs in 17 African languages; read the gap to English.
- [AfroBench: How Good are Large Language Models on African Languages?](https://arxiv.org/abs/2311.07978) (Ojo et al., 2023). A map of 22 datasets over 64 languages; use it to find related work for your paper.
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (Liang et al., 2022). Sections 1–2: scenarios and metrics as separate choices.
- [Dynabench: Rethinking Benchmarking in NLP](https://arxiv.org/abs/2104.14337) (Kiela et al., 2021). The case that static benchmarks saturate.
- [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811) (Recht et al., 2019). A rebuilt test set, and what the drop does and does not show.
- [Time Travel in LLMs: Tracing Data Contamination in Large Language Models](https://arxiv.org/abs/2308.08493) (Golchin and Surdeanu, 2023). A contamination check you can run.
- [Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324) (Sclar et al., 2023). Formatting alone moves accuracy.

**Reporting**
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al., 2017). Reliability diagrams, ECE and temperature scaling.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023). Section 3 lists the judge biases.
- [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926) (Wang et al., 2023). Order of presentation changes the winner.
- [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) (Dror et al., 2018). Which paired test fits which metric.
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2018). The template for the model card in your package.
- [Data Statements for Natural Language Processing](https://aclanthology.org/Q18-1041/) (Bender and Friedman, 2018). The questions to answer about any dataset you build or use.

**Safety evaluation**
- [HarmBench](https://arxiv.org/abs/2402.04249) (Mazeika et al., 2024). A fixed protocol for automated red teaming.
- [XSTest](https://arxiv.org/abs/2308.01263) (Röttger et al., 2023). Benign prompts that measure over-refusal.
- [Red-Teaming Large Language Models in Low-Resource African Languages: Yoruba and Nigerian Pidgin](https://arxiv.org/abs/2610.39590). The source of the refusal figures above; a 2026 preprint, so read it critically.
