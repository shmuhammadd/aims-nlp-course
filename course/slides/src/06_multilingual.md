type: title
title: Multilingual and low-resource modelling; building data for African languages
subtitle: Day 6 · Week 2
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: Your model scores 85% on the test set. How well does it work in Hausa?

Notes: Nobody can answer from the number given. Collect what the class would need to know: how many Hausa items, the Hausa score, who labelled them. Slide 10 gives a case where 85% overall hides 30% in Hausa.

---
title: Where today sits
subtitle: Week 1 built and adapted one model. Week 2 asks: for which languages?

| Today | Question | Capstone link |
|---|---|---|
| Part 1: Models | Which encoder, and how to adapt it? | Task 12 text-only baseline: XLM-R, macro-F1 0.550 |
| Part 2: Measurement | Which number, with what uncertainty? | Task 12 ranks on macro-F1 per language, then averaged |
| Part 3: Data | Who wrote and labelled the data? | Task 12 captions come from native speakers |

- Task 12 has captions in Hausa, Igbo and Yoruba, and in eight more African languages
- Task 9 is multilingual with no African language: Bulgarian, English, Portuguese, Russian

Notes: Facts from the course task cards. The Task 12 multimodal baseline scores 0.535, below the text-only 0.550, so the text encoder is the place to start.

---
type: section
title: Part 1: Models
subtitle: One model, many languages

---
title: Transfer between languages is conditional

- A shared model can reuse syntax, meaning and subword pieces across languages
- How much it reuses depends on four things
  - how much of the language it saw in pretraining
  - the script
  - the domain of the text
  - how much labelled data you have
- Related languages are not interchangeable
- Code-switched text and spelling variation break a tokenizer or label scheme built for one language
- Translated training data reads like translation, and can shift sentiment that depends on culture

Notes: A code-switched example to write on the board: "Na gode sosai, this is great". Na gode sosai is Hausa for thank you very much. Ask which language tag the tweet gets, and whether a Hausa-only filter would have kept it.

---
title: Five ways to use a multilingual model

| Pipeline | What trains | Cost |
|---|---|---|
| Frozen encoder and a task head | The head | Minutes on a CPU once features are cached |
| Fine-tune the encoder | Every weight | One GPU run per task |
| Language or task adapters | Small added layers | Less memory; one adapter per language (MAD-X) |
| Continued pretraining | Every weight, on unlabelled text in the language | Needs raw text, no labels |
| Instruction tuning | A generative model, or a LoRA adapter on it | Day 4 |

- Run the cheap classifier first. It is the baseline every other row must beat.

Notes: Pfeiffer et al., 2020 is MAD-X. Continued pretraining uses the day-3 objective on new text.

---
title: Encoder choices for African languages

### XLM-R (Conneau et al., 2019)
- A masked language model trained on one hundred languages, from more than two terabytes of filtered CommonCrawl text
- The Task 12 organisers' text baseline

### AfroXLM-R (Alabi et al., 2022)
- XLM-R after more pretraining on text in 17 of the most-resourced African languages and three other languages widely spoken in Africa
- The authors call this multilingual adaptive fine-tuning. It is the continued-pretraining row of the last slide.

- Before you choose, check each model's language list for your language. A language the encoder never saw is tokenized into small pieces (day 1).

Notes: Both descriptions are from the papers' abstracts. Do not state from memory which African languages are in XLM-R's hundred; send teams to the appendix of the paper.

---
title: Smoothed sampling gives small languages more turns
layout: split

Sample language l with probability

$$
q_l = \frac{n_l^{\alpha}}{\sum_j n_j^{\alpha}}
$$

- α = 1 follows corpus size. α = 0 is uniform.
- Sizes 10,000, 1,000 and 100, α = 0.5: q = 0.71, 0.22, 0.07
- In 10,000 draws at α = 0, each example of the smallest language is seen 33 times
- More turns, no new sentences. Watch for memorising.

![](figures/06_sampling.png)

Notes: √10000 = 100, √1000 = 31.6, √100 = 10; the sum is 141.6. At α = 0: 3,333 draws over 100 examples. Count unique examples and repeats, not only optimiser steps.

---
type: section
title: Part 2: Measurement
subtitle: Which number, and how sure are you?

---
title: Worked example: one score hides a language
layout: split

System A gets 90 of 100 English items and 3 of 10 Hausa items.

$$
\mathrm{overall} = \frac{93}{110} = 0.845
$$

$$
\mathrm{mean\ over\ languages} = \frac{0.90 + 0.30}{2} = 0.60
$$

- System B gets 86 and 7: also 0.845 overall, and 0.78 over languages
- Report the per-language counts with either number

![](figures/06_micro_macro.png)

Notes: The table is invented for the lab. Overall accuracy weights items. The mean over languages weights languages. Neither is wrong; they answer different questions.

---
title: F1 needs its denominators stated

$$
P = \frac{TP}{TP + FP} \qquad R = \frac{TP}{TP + FN} \qquad F_1 = \frac{2\,TP}{2\,TP + FP + FN}
$$

- Macro-F1 is the mean of the per-class F1 values, so a small class counts as much as a large one
- Lab example, six items: accuracy 0.667, macro-F1 0.472. One class is never predicted and scores 0.
- Say what you do when a class has no gold and no predicted items in a language: score it 0, or leave it out
- Task 12 computes macro-F1 per language, then averages over languages

Notes: The six items: gold 0,0,0,1,1,2 and predictions 0,0,0,0,1,0. Class F1 values are 0.75, 0.667 and 0. Lab exercise B implements the leave-it-out rule and shows the two rules differ: 0.733 against 0.489 for one language.

---
title: Paired bootstrap: resample the items, keep the pairing

```
idx = rng.integers(0, n, size=(2000, n))    # item indices
diff = (correct_b[idx] - correct_a[idx]).mean(axis=1)
low, high = np.quantile(diff, [0.025, 0.975])
```

- Both systems answered the same items, so one set of indices serves both
- Lab, all 110 items: B − A = 0.0, interval [−0.045, 0.055]
- Lab, the 10 Hausa items: B − A = 0.4, interval [0.1, 0.7]
- Ten items cannot say more than that
- An interval describes sampling noise. It does not repair a test set that misses the people you care about.

Notes: A language with 20 examples has a far wider interval than one with 2,000. Ask what interval width the class would want before choosing B for Hausa.

---
title: Items by one author rise and fall together
subtitle: Lab: suppose 22 authors wrote five items each
layout: stack

![](figures/06_bootstrap.png)

- Resample authors, not items, when speakers or documents link the items. The interval widens.

Notes: Seed 42, 2,000 resamples. The author grouping is an assumption made for the demonstration. In Task 12, many annotators caption the same painting, so the painting is the natural cluster.

---
type: section
title: Part 3: Building data
subtitle: Someone has to write the labels

---
title: A dataset is built in six steps
layout: stack

![](figures/06_data_pipeline.png)

- Each box is a decision that belongs in the paper. The loop back from the pilot is where most of the time goes.

Notes: The numbers in the boxes are a common design, not a rule. The instructor led AfriSenti and AfriHate; tell the class what took longest.

---
title: Running example: AfriSenti

- A sentiment benchmark of more than 110,000 tweets in 14 African languages, from four language families
- The languages: Amharic, Algerian Arabic, Hausa, Igbo, Kinyarwanda, Moroccan Arabic, Mozambican Portuguese, Nigerian Pidgin, Oromo, Swahili, Tigrinya, Twi, Xitsonga and Yoruba
- Native speakers annotated the tweets
- It was the data for a SemEval shared task with over 200 participants
- The paper describes the collection method, the annotation process and the problems met for each language

Notes: Muhammad et al., EMNLP 2023. Every fact on this slide is from the paper's abstract. Read sections on collection and annotation for the detail; the instructor can add first-hand accounts.

---
title: Collecting text: decide before you download

| Decision | Why it matters |
|---|---|
| Source: social media, news, radio transcripts | Sets the domain your model will work in |
| How you find text in the language | A keyword list finds the easy cases and misses the rest |
| Code-switched and mixed-script text: keep or drop | It is how many people write |
| Dialect and spelling | Yoruba is often typed without tone marks; Hausa is written in Latin and in Arabic script |
| Personal data and consent | Names and handles are in the text |

- Write each decision down on the day you make it. It becomes the data statement.

Notes: Tie to day 1: every row changes what the tokenizer sees. A filter that keeps only clean, single-language text builds a test set easier than real use.

---
title: Guidelines and annotators

- Write one definition per label, with two examples that fit and one that nearly fits
- Say what to do with sarcasm, mixed feeling, proverbs and quoted speech
- Use native speakers of the variety in the data. A speaker of one dialect may misread another.
- Pilot on 100 items with two annotators, measure agreement, and read every disagreement with them
- Revise the guidelines, then pilot again
- Pay annotators, get consent, and let them skip items that distress them

Notes: The last line is the rule from the safety thread and matters most for hate-speech data such as AfriHate, where each instance is annotated by native speakers familiar with the local culture.

---
title: Cohen's kappa: agreement beyond chance
subtitle: Two annotators label 100 tweets as positive or negative

| | B: positive | B: negative |
|---|---|---|
| **A: positive** | 40 | 20 |
| **A: negative** | 10 | 30 |

$$
p_o = \frac{40 + 30}{100} = 0.70 \qquad p_e = 0.6 \times 0.5 + 0.4 \times 0.5 = 0.50
$$

$$
\kappa = \frac{p_o - p_e}{1 - p_e} = \frac{0.70 - 0.50}{0.50} = 0.40
$$

Notes: p_e is the agreement two annotators with these label rates would reach by guessing. A says positive 60 times, B 50 times. Derive on the board: κ is the share of the possible improvement over chance that the annotators achieved. κ = 1 is full agreement, 0 is chance. Landis and Koch (1977) call 0.41 to 0.60 moderate; treat such bands as convention.

---
title: High agreement can still be low kappa
layout: split

Hate speech is rare. Two annotators label 100 posts.

- Both "hateful": 2. Both "not": 90. Split: 8.
- p_o = 0.92
- p_e = 0.06² + 0.94² = 0.887
- κ = 0.033 / 0.113 = 0.29
- Agreeing on "not hateful" is cheap. Kappa counts what is left.
- Report kappa with the label counts beside it

![](figures/06_kappa_prevalence.png)

Notes: The figure is a formula: two annotators who are each right 90% of the time, with independent errors. Their raw agreement is 0.82 at any class balance, and kappa falls from 0.64 at a 50% positive share to 0.25 at 5%.

---
title: Three annotators, a majority, an adjudicator

| Lab simulation, 600 tweets, 3 labels | Share |
|---|---|
| All three annotators agree | 0.607 |
| Two of three agree | 0.360 |
| No majority: goes to an adjudicator | 0.033 |

- One annotator matches the true label on 0.842 of tweets. The majority label does on 0.952.
- Release the individual labels as well as the majority. Disagreement is information.
- An adjudicator is a senior native speaker who reads the guidelines and decides

Notes: Simulated annotators with invented error rates, seed 42. AfriHate releases its individual annotations, as its abstract states.

---
title: Kappa by slice shows where guidelines fail

| Lab simulation | Tweets | Kappa |
|---|---|---|
| All tweets | 600 | 0.58 |
| One language | 424 | 0.74 |
| Code-switched | 176 | 0.19 |

- The simulation gives annotators a 10% error rate on tweets in one language and 35% on code-switched tweets
- One overall kappa hides the weak slice, as one overall accuracy hid Hausa
- What to do: add code-switched examples to the guidelines, and recruit annotators who use both languages

Notes: The error rates are invented to make the point. In a real project you find the weak slice by tagging items during the pilot.

---
title: Licence, consent and the data statement

- A platform's terms may forbid sharing text. Many datasets share IDs and labels.
- Choose a licence and state it. Task 12: annotations are CC BY 4.0; the paintings are third-party works.
- A dataset on the web is not free of conditions, and may not stay in the same form

### A data statement says (Bender and Friedman, 2018)
- Which language varieties, from where and when
- Who the annotators are, and how they were recruited and paid
- What was left out, and the known limits

Notes: Ask speakers to interpret the errors, with credit and pay. Do not make students disclose their own language or identity to take part.

---
title: Safety does not transfer by default

- Most safety training is done in English
- Yong et al. (2023): unsafe English prompts translated into low-resource languages got past GPT-4's safeguards
- The Illusion of Cross-Lingual Safety (2026) probes refusal signals in hidden states for Twi, Hausa, Amharic and Swahili
  - for most language-model pairs, harmful prompts kept under 10% of the English refusal signal
  - literal and localised versions of a prompt sat close together in meaning
- CAB-Igbo (2026): fluent answers can lack cultural substance, and the gap is already there in English
- Report a harmful rate per language, with a paired interval against English

Notes: Yong et al. report 79% on AdvBench in their abstract. The two 2026 papers are preprints from the AfriSafe group; pair each with an independent paper. The paired interval is today's bootstrap over prompt IDs.

---
type: big
title: A model refuses a literal Hausa translation of a harmful request and answers the localised version. Which result goes in the safety report?

Notes: Both, in separate rows. The literal translation tests whether English safety training carried across. The localised prompt tests what a Hausa speaker would write. A report with only the first overstates safety.

---
title: Lab: 40 minutes

- Open `course/tutorials/06_multilingual.ipynb`. NumPy only.
- Sections 1 to 4, together
  - Two systems with one overall score
  - Paired bootstrap over items, then over authors
  - Sampling weights for three corpus sizes
  - Kappa for two tables, then three simulated annotators
- Section 5, in pairs: write two functions until the tests pass
  - `cohen_kappa` for two label arrays
  - `macro_f1_by_language` with a stated rule for absent classes
- Task 12 teams: run your function on your development split tonight

---
title: Readings

### Before the next session (20 minutes)
- Muhammad et al., 2023. AfriSenti. The sections on data collection and annotation.

### Models
- Conneau et al., 2019. XLM-R. Sections 1 to 3.
- Alabi et al., 2022. Adapting pre-trained language models to African languages.
- Pfeiffer et al., 2020. MAD-X.

### Data and safety
- Muhammad et al., 2025. AfriHate.
- Bender and Friedman, 2018. Data Statements for NLP.
- Yong et al., 2023. Low-Resource Languages Jailbreak GPT-4.
- The Illusion of Cross-Lingual Safety in Low-Resource Languages, 2026.

Notes: Links are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One result you measured today, with its number
- One reason it might not hold for real annotators or a real model
- One decision about your team's data that you will write down tonight

Quiz 6 opens the next session: six questions, ten minutes.
