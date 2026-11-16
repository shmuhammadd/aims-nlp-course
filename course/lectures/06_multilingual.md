# Lecture 06 — Multilingual and low-resource modelling; building data for African languages

[Course home](../../README.md) · Week 2, day 1 · 120 minutes · [Slides](../slides/06_multilingual.pptx) · [Notebook](../tutorials/06_multilingual.ipynb) · [Exercises](../exercises/06_multilingual.md) · [Quiz](../quizzes/06_multilingual.md)

## Learning outcomes

- Compare micro, macro, and per-language performance, and put a paired interval on a difference.
- Explain transfer, continued pretraining, and language sampling trade-offs, and choose an encoder for an African language.
- Plan the building of an annotated dataset: collection, guidelines, native-speaker annotators, agreement, adjudication, licence and consent.
- Compute Cohen's kappa and say what it does and does not show.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 5 (alignment: preference optimisation, RLHF and safety) |
| 10–25 | Slides 1–8: the Task 12 link, conditional transfer, five pipelines, encoder choices, language sampling |
| 25–40 | Slides 9–13: the worked example on the board, F1 and its denominators, paired and clustered bootstrap |
| 40–60 | Slides 14–23: building a dataset, with AfriSenti as the example; Cohen's kappa on the board; adjudication, code-switching, licence and consent |
| 60–65 | Break |
| 65–105 | [Notebook 06](../tutorials/06_multilingual.ipynb): sections 1–4 together, section 5 in pairs |
| 105–115 | Slides 24–25: safety does not transfer; discuss the literal-against-localised question |
| 115–120 | Exit ticket: one measured result, one reason it may not transfer, one data decision to write down |

## Teaching notes

### Transfer is conditional
A shared model may reuse syntax, semantics, or subword patterns across languages, but transfer depends on training exposure, script, domain, and supervision. Related languages are not interchangeable. Code-switching and spelling variation can invalidate assumptions in a monolingual tokenizer or label schema. Translation-based augmentation can introduce translationese and change culturally situated sentiment.

A multilingual pipeline can use a frozen encoder with a task head, task-specific fine-tuning, language adapters, continued pretraining, or instruction tuning. Compare an inexpensive classifier before investing in generative fine-tuning.

### Encoder choices for African languages
XLM-R (Conneau et al., 2019) is a masked language model trained on one hundred languages from more than two terabytes of filtered CommonCrawl text. AfroXLM-R (Alabi et al., 2022) continues XLM-R's pretraining on text in 17 of the most-resourced African languages and three other languages widely spoken on the continent; the authors call this multilingual adaptive fine-tuning. Check each model's language list for your language before choosing. A language the encoder never saw is split into many small pieces by its tokenizer, as on day 1.

Task 12 has Hausa, Igbo and Yoruba captions, among 11 African languages. The organisers' text-only XLM-R baseline scores macro-F1 0.550 on subtask 1, above their multimodal baseline at 0.535. A team can start by swapping the encoder, continuing pretraining on captions, or changing the language sampling.

### Sampling and representation
If language $l$ has $n_l$ examples, a temperature-smoothed sampling distribution can be $q_l=n_l^\alpha/\sum_j n_j^\alpha$, where $\alpha=1$ follows the corpus and $\alpha=0$ samples languages uniformly. Reducing $\alpha$ increases exposure for smaller languages, but repeated examples may overfit. With sizes 10,000, 1,000 and 100 and $\alpha=0$, each example of the smallest language is seen 33 times in 10,000 draws. Count unique examples and effective repetitions, not only optimiser steps.

### Metrics with explicit denominators
For a class, precision $=TP/(TP+FP)$, recall $=TP/(TP+FN)$, and F1 is their harmonic mean. State how undefined terms are treated. Macro-F1 averages class F1 values and highlights minority-class failures. Overall accuracy weights examples; equal-language accuracy averages language accuracies. These answer different questions. A language with 20 examples has much wider uncertainty than one with 2,000. Task 12 computes macro-F1 per language and then averages over languages.

Use paired bootstrap resampling when comparing predictions on the same evaluation units. If speakers or documents induce dependence, resample clusters and not individual rows. In the lab, resampling 22 assumed authors in place of 110 items widens the interval on a difference from $[-0.045, 0.055]$ to $[-0.091, 0.073]$. A confidence interval quantifies a sampling procedure under assumptions; it does not correct an unrepresentative sample.

### Building a dataset
The work has six steps: collect text, write guidelines, pilot, annotate, adjudicate, release. AfriSenti (Muhammad et al., 2023) is the running example. It holds more than 110,000 tweets in 14 African languages from four language families: Amharic, Algerian Arabic, Hausa, Igbo, Kinyarwanda, Moroccan Arabic, Mozambican Portuguese, Nigerian Pidgin, Oromo, Swahili, Tigrinya, Twi, Xitsonga and Yoruba. Native speakers annotated the tweets, and the data served a SemEval shared task with over 200 participants. The paper describes the collection method, the annotation process and the problems met for each language; read those sections for the detail.

**Collecting.** Decide the source, because it sets the domain. Decide how you find text in the language: a keyword list finds the easy cases and misses the rest. Decide whether code-switched and mixed-script text stays; it is how many people write. A tweet such as "Na gode sosai, this is great" is Hausa and English at once. Record how spelling and dialect vary: Yoruba is often typed without tone marks, and Hausa is written in Latin and in Arabic script.

**Guidelines and annotators.** Write one definition per label with examples that fit and one that nearly fits. Say what to do with sarcasm, mixed feeling, proverbs and quoted speech. Use native speakers of the variety in the data. Pilot on about 100 items with two annotators, measure agreement, read every disagreement with them, and revise.

**Agreement.** Cohen's kappa compares observed agreement $p_o$ with the agreement $p_e$ that two annotators with the same label rates would reach by chance: $\kappa=(p_o-p_e)/(1-p_e)$, with $p_e=\sum_c p_{1c}\,p_{2c}$. Kappa is 1 for full agreement and 0 for chance. When one class is rare, raw agreement is high by chance and kappa is the more honest number: a table with 92% agreement can have $\kappa=0.29$. Report kappa with the label counts, and by slice. In the lab's simulation kappa is 0.74 on tweets in one language and 0.19 on code-switched tweets.

**Adjudication.** With three annotators, take the majority label and send items with no majority to an adjudicator, a senior native speaker who applies the guidelines. In the lab's simulation 3.3% of items have no majority, and the majority label matches the true label on 0.952 of items against 0.842 for one annotator. Release the individual labels with the majority label. AfriHate (Muhammad et al., 2025), with hate speech and abusive language in 15 African languages, releases its individual annotations.

**Licence and consent.** Read a dataset's annotation and access conditions before using it. Do not assume a web-hosted dataset is unrestricted or still available in the same form. A platform's terms may forbid sharing text, which is why many datasets share IDs and labels. State your licence: Task 12's annotations are CC BY 4.0 and its paintings are third-party works. Pay annotators, obtain consent, and let them skip distressing items.

A data statement should identify language varieties, collection context, annotators, consent and access conditions, exclusions, and known limitations (Bender and Friedman, 2018). Ask speakers to interpret errors, with credit and appropriate compensation in real projects. Do not make students disclose sensitive language or identity information to participate.

For this course, synthetic examples make all mandatory labs independent of downloads; a real-data extension uses student-supplied permitted TSV files.

### Safety does not transfer by default
Most safety training is done in English. Yong et al. (2023) report that translating unsafe English prompts into low-resource languages got past GPT-4's safeguards. The Illusion of Cross-Lingual Safety (2026) probes hidden-state refusal signals in Twi, Hausa, Amharic, and Swahili. For most language-model pairs, harmful prompts kept under 10% of the English refusal signal, although literal and localized versions of a prompt sat close together in meaning. Its dataset pairs literal translations with culturally localized prompts, so a translation test and a localization test can give different answers.

Cultural knowledge fails on its own track. CAB-Igbo finds that fluent answers can lack cultural substance, and that the gap is already present in English. Discussion: a model refuses a literal Hausa translation but answers the localized version of the same request. Which result goes in the safety report? See the [safety thread](../SAFETY.md).

## Worked example

A model gets 90/100 English examples and 3/10 Hausa examples correct. Overall accuracy is 93/110=84.55%, while equal-language accuracy is (0.90+0.30)/2=60%. Neither should be reported without the per-language counts.

Two annotators label 100 tweets. Both say positive for 40, both say negative for 30, A alone says positive for 20 and B alone for 10. Then $p_o=0.70$. A says positive 60 times and B 50 times, so $p_e=0.6\times0.5+0.4\times0.5=0.50$ and $\kappa=(0.70-0.50)/(1-0.50)=0.40$.

## Guided tutorial

**Question:** Show how one overall score hides a language, bootstrap a paired comparison over items and over authors, weigh languages for sampling, and measure annotator agreement beyond chance.

The notebook compares two invented systems with the same overall accuracy (0.845) and different Hausa accuracy (0.30 and 0.70), computes paired and clustered intervals, prints sampling weights for three corpus sizes, and computes kappa for two tables and for three simulated annotators. Students then write `cohen_kappa` and `macro_f1_by_language` against tests. Every table is invented or simulated.

**Real-data extension:** Run `extensions/afrisenti_baseline.py` on permitted train, dev and test TSV exports. It fits a character TF-IDF classifier without downloading or redistributing tweets. See the [extension guide](../extensions/README.md) for commands, hardware and validation status.

## Practical questions

1. Why might uniform language sampling hurt a very small language?
2. Can translated evaluation data establish performance on naturally occurring code-switching?
3. Why is macro-F1 useful but insufficient for a multilingual system?
4. Two annotators agree on 92% of posts in a hate-speech pilot. What else do you need before you start the full annotation?

## After class

Complete the [exercise sheet](../exercises/06_multilingual.md) in 45–60 minutes. Read the data collection and annotation sections of the AfriSenti paper before the next session (20 minutes). [Quiz 6](../quizzes/06_multilingual.md) opens day 7.

## Readings

**Required**
- [AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages](https://aclanthology.org/2023.emnlp-main.862/) (Muhammad et al., 2023). The sections on data collection and annotation: how a 14-language dataset was built.

**Models**
- [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) (Conneau et al., 2019). XLM-R; sections 1–3, and the language list in the appendix.
- [Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning](https://arxiv.org/abs/2204.06487) (Alabi et al., 2022). AfroXLM-R; read which languages were added and the evaluation tasks.
- [MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/abs/2005.00052) (Pfeiffer et al., 2020). Language and task adapters; read figure 1.

**Data**
- [AfriHate](https://arxiv.org/abs/2501.08284) (Muhammad et al., 2025). Hate speech and abusive language in 15 African languages; read the annotation challenges.
- [Data Statements for Natural Language Processing](https://aclanthology.org/Q18-1041/) (Bender and Friedman, 2018). The schema in section 4; use it for exercise 3.

**Safety**
- [Low-Resource Languages Jailbreak GPT-4](https://arxiv.org/abs/2310.02446) (Yong et al., 2023). Four pages; read the method and the per-language table.
- [Multilingual Jailbreak Challenges in Large Language Models](https://arxiv.org/abs/2310.06474) (Deng et al., 2023). An independent study of the same effect.
- [The Illusion of Cross-Lingual Safety in Low-Resource Languages](https://arxiv.org/abs/2608.11146) (2026 preprint). Literal against localised prompts; read critically, beside Yong et al.
- [Do Language Models Understand Igbo Culture? Beyond Multiple Choice](https://arxiv.org/abs/2610.35168) (2026 preprint). CAB-Igbo; cultural knowledge in open-ended answers.
