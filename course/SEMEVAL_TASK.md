# SemEval-2027 task cards

[Course home](../README.md) · [Capstone brief](PROJECT.md) · [Paper guide](SEMEVAL_PAPER.md)

Each team enters **one** of the four tasks below, one subtask or track, and at most three languages. The facts on this page were read from the task websites on 16 November 2026. The organisers' pages govern: check them before you rely on a date, a rule or a number here.

All four tasks share the SemEval-2027 calendar: evaluation runs **10–31 January 2027**, system papers are due in February 2027 (tentative), and the workshop is at ACL 2027 in Kyoto, 17–22 August 2027. See the [SemEval-2027 site](https://semeval.github.io/SemEval2027/) and its [task list](https://semeval.github.io/SemEval2027/tasks).

## Which task suits which team

| Task | You build | African languages | Course days you will lean on | Starting point |
|---|---|---|---|---|
| 12 ArtEmotion-28 | Emotion and sentiment for paintings in 28 languages | Hausa, Igbo, Yoruba, Swahili, Kinyarwanda, Emakhuwa, isiZulu, isiXhosa, Setswana, isiNdebele, Darija | 6, 7, 8, 10 | Starter notebook and three baselines |
| 6 MMCultureQA | Image plus spoken or written question → short answer | Swahili, Amharic; Oromo and Somali announced | 8, 9, 12 | Dev-test leaderboard on EvalAI |
| 7 CLaS | Steer an LLM to answer in a target language by editing activations | Swahili (target, track 2) | 11, 13 | Prompting baseline and scorer |
| 9 Multimodal framing | Frame labels for news paragraphs and images | None (Bulgarian, English, Portuguese, Russian) | 6, 7, 8 | Trial data |

Task 12 is the default for a team with no strong preference: it has the most African languages, a 130 MB dataset and working baselines.

## Task 12 — ArtEmotion-28

- **Site and data:** <https://github.com/artemotion-28/artemotion-28>. 173,745 annotations (train and validation) and 1,658 paintings. Annotations are CC BY 4.0; the images are third-party WikiArt works.
- **Subtask 1, emotion:** painting plus a native-speaker caption → one of eight emotions (amusement, awe, contentment, excitement, anger, disgust, fear, sadness). Two runs are required: text-only and multimodal. Ranked on macro-F1, computed per language and then averaged, using the multimodal run.
- **Subtask 2, divergence:** painting plus the source language's sentiment distribution → the share of positive annotators in a target language. Ranked on log loss in bits per annotator.
- **Subtask 3, image only:** painting plus a target language → the share of positive annotators. Ranked on within-painting Pearson correlation across languages.
- **Image-use check (subtask 1):** each test row has a decoy with the same caption and another painting. A multimodal run must do better with the right image (one-sided McNemar test, p < 0.05). A text-only run must give real and decoy rows the same label at least 99% of the time.
- **Organisers' baselines:** subtask 1, XLM-R plus ViT with concatenation, macro-F1 0.535 multimodal against 0.550 text-only; subtask 2, log loss 0.80; subtask 3, correlation 0.45 with frozen CLIP ViT-B/32 features.
- **Rules:** annotations of evaluation paintings from any other source, including the original ArtELingo release, may not be used. English, Arabic and Chinese are not scored at test time. Guidance on LLMs and extra data is to be published with the platform details.
- **Submission:** Codabench, one competition per subtask: [1](https://www.codabench.org/competitions/18466/), [2](https://www.codabench.org/competitions/18467/), [3](https://www.codabench.org/competitions/18468/). Subtask 3 closes on **16 January 2027, 23:59 UTC**; subtask 1 and 2 test files are released on 17 January.
- **Conflict of interest:** the course instructor is a co-organiser of this task. Students do not receive test labels or any information that other teams lack, and each paper states the connection.

## Task 6 — MMCultureQA

- **Site:** <https://mmcultureqa-semeval27.github.io/>. Built on the OASIS dataset.
- **Subtask 1, spoken visual QA:** image plus a spoken question → short text answer. **Subtask 2, textual visual QA:** image plus a written question → short text answer. One track per language.
- **Languages:** 17 tracks open, including Kiswahili (Tanzania) and Amharic (Ethiopia); five more announced, including Afaan Oromo and Somali.
- **Data:** up to 10,000 training and 1,000 development items per track. Spoken questions are synthetic in train and dev; the test set uses human recordings.
- **Metric:** BERTScore F1.
- **Submission:** dev-test leaderboards on EvalAI. The test-phase platform and the rules on external data and LLM APIs were not stated on the page; ask in the task's Slack channel.

## Task 7 — CLaS

- **Site:** <https://clas-semeval2027.github.io/clas.semeval2027/task/>. Code, data and scorer: <https://github.com/d-gurgurov/CLaS-SemEval-2027>.
- **Goal:** given a question in a source language, make the model answer in a target language by intervening on its internal computations. A target-language instruction in the prompt is not ranked.
- **Tracks:** 1, ten high-resource languages to each other (90 pairs); 2, the same ten sources to cs, el, fa, vi, uk, mt, no, sw, th, tr (100 pairs); 3, all twenty languages into ja, ru or de; bonus, refusals in the requested language.
- **Models ranked:** `meta-llama/Llama-3.1-8B-Instruct` (licence-gated; request access early) and `CohereLabs/tiny-aya-global`.
- **Scoring:** language compliance by language identification, and relevance of the answer to the question. The page does not publish how the two are combined.
- **Submission:** code and requirements through a form on the task site; the organisers run it on their GPUs. Two test submissions and one final submission per track.

## Task 9 — Multimodal framing in the news

- **Site:** <https://propaganda.math.unipd.it/semeval2027task9/>.
- **Subtask 1:** label each paragraph of a news article with zero or more of 14 generic frames (Card et al., 2015). **Subtask 2:** label the article's image with zero or more frames, without (2a) or with (2b) gold text frames.
- **Languages:** Bulgarian, English, Portuguese (European), Russian.
- **Metric:** multi-label micro-F1; macro-F1 is reported as secondary.
- **Submission:** Codabench, after the organisers approve the team. Training data is released in stages; confirm the release and evaluation dates on the task page, where the years shown were inconsistent when this card was written.

## Class fields

| Field | Value |
|---|---|
| Participation route | Live shared task |
| Team size | Pairs, or individual by agreement |
| Course deadlines | Day 5 organisers' baseline reproduced; day 10 experiment plan frozen; day 14 peer review; day 15 paper, package and defence |
| Official submission owner | One named person per team checks and submits the agreed files |
| After the course | Evaluation 10–31 January 2027; paper in February 2027 |

## Team addendum

Add to your project folder: members and roles; the task, subtask and languages; one research question; the baseline you reproduced and its score; the change you propose; two planned ablations; your compute limit; and the commands that train, predict, score and export. Record a submission receipt only after a real submission.
