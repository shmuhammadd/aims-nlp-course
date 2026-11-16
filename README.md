# Advanced Language and Multimodal Models

**AIMS · 2026 edition · 3 weeks · 5 days a week · 2 hours a day · 30 contact hours**

A course on how language and multimodal models are trained, adapted, aligned, evaluated and steered, taught with African languages and limited compute in view. Students build each mechanism in a small NumPy lab, then use it in a SemEval-2027 shared task and write a system-description paper.

By the end, students can implement causal attention and next-token training; explain prompting, LoRA, preference optimisation and safety alignment; build a multilingual dataset and measure annotator agreement; work with image-text, vision-language and speech models; benchmark a system with the right metric and a paired test; find and steer a direction in a model's activations; and defend a result with controlled ablations.

## Start here

1. Read the [syllabus](course/SYLLABUS.md).
2. Do the [prerequisite diagnostic](course/PREREQUISITES.md) and the [setup](course/SETUP.md).
3. Choose one of the four [SemEval-2027 tasks](course/SEMEVAL_TASK.md) on day 1 and read the [capstone brief](course/PROJECT.md).
4. Each day: slides and notes, the notebook, the exercise sheet, and next morning the quiz.

Each two-hour session has a 10-minute quiz on the previous day, about 50 minutes of teaching, a 40-minute notebook lab and a discussion. Exercise sheets take 45–60 minutes after class. The capstone takes 12–16 hours across the three weeks.

## Week 1 — Foundations and post-training

By Friday each team has reproduced the organisers' baseline for its task, starting with text.

| Day | Session | Slides | Notes | Notebook | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 1 | Language models, tokens and baselines; the four shared tasks | [Deck](course/slides/01_tokenization.pptx) | [Notes](course/lectures/01_tokenization.md) | [Lab](course/tutorials/01_tokenization.ipynb) | [Sheet](course/exercises/01_tokenization.md) | [Quiz](course/quizzes/01_tokenization.md) |
| 2 | The causal Transformer | [Deck](course/slides/02_transformers.pptx) | [Notes](course/lectures/02_transformers.md) | [Lab](course/tutorials/02_transformers.ipynb) | [Sheet](course/exercises/02_transformers.md) | [Quiz](course/quizzes/02_transformers.md) |
| 3 | Pretraining: data, objectives, scaling; decoding and sampling | [Deck](course/slides/03_pretraining.pptx) | [Notes](course/lectures/03_pretraining.md) | [Lab](course/tutorials/03_pretraining.ipynb) | [Sheet](course/exercises/03_pretraining.md) | [Quiz](course/quizzes/03_pretraining.md) |
| 4 | Post-training I: prompting, instruction tuning and LoRA | [Deck](course/slides/04_adaptation.pptx) | [Notes](course/lectures/04_adaptation.md) | [Lab](course/tutorials/04_adaptation.ipynb) | [Sheet](course/exercises/04_adaptation.md) | [Quiz](course/quizzes/04_adaptation.md) |
| 5 | Post-training II: alignment (preference optimisation, RLHF, safety) | [Deck](course/slides/05_alignment.pptx) | [Notes](course/lectures/05_alignment.md) | [Lab](course/tutorials/05_alignment.ipynb) | [Sheet](course/exercises/05_alignment.md) | [Quiz](course/quizzes/05_alignment.md) |

## Week 2 — Languages, modalities and benchmarking

By Friday each team has an improved or multimodal system and a frozen experiment plan.

| Day | Session | Slides | Notes | Notebook | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 6 | Multilingual and low-resource modelling; building data for African languages | [Deck](course/slides/06_multilingual.pptx) | [Notes](course/lectures/06_multilingual.md) | [Lab](course/tutorials/06_multilingual.ipynb) | [Sheet](course/exercises/06_multilingual.md) | [Quiz](course/quizzes/06_multilingual.md) |
| 7 | Image-text representations: CLIP and SigLIP | [Deck](course/slides/07_contrastive.pptx) | [Notes](course/lectures/07_contrastive.md) | [Lab](course/tutorials/07_contrastive.ipynb) | [Sheet](course/exercises/07_contrastive.md) | [Quiz](course/quizzes/07_contrastive.md) |
| 8 | Vision-language models | [Deck](course/slides/08_vlm.pptx) | [Notes](course/lectures/08_vlm.md) | [Lab](course/tutorials/08_vlm.ipynb) | [Sheet](course/exercises/08_vlm.md) | [Quiz](course/quizzes/08_vlm.md) |
| 9 | Speech and multilingual ASR | [Deck](course/slides/09_speech.pptx) | [Notes](course/lectures/09_speech.md) | [Lab](course/tutorials/09_speech.ipynb) | [Sheet](course/exercises/09_speech.md) | [Quiz](course/quizzes/09_speech.md) |
| 10 | Benchmarking and evaluation | [Deck](course/slides/10_evaluation.pptx) | [Notes](course/lectures/10_evaluation.md) | [Lab](course/tutorials/10_evaluation.ipynb) | [Sheet](course/exercises/10_evaluation.md) | [Quiz](course/quizzes/10_evaluation.md) |

## Week 3 — Inside the model, systems and the paper

By Friday each team submits its paper, prediction package and defence.

| Day | Session | Slides | Notes | Notebook | Exercises | Quiz |
|---|---|---|---|---|---|---|
| 11 | Interpretability and cross-lingual steering | [Deck](course/slides/11_steering.pptx) | [Notes](course/lectures/11_steering.md) | [Lab](course/tutorials/11_steering.ipynb) | [Sheet](course/exercises/11_steering.md) | [Quiz](course/quizzes/11_steering.md) |
| 12 | Retrieval, tools and agents | [Deck](course/slides/12_rag_agents.pptx) | [Notes](course/lectures/12_rag_agents.md) | [Lab](course/tutorials/12_rag_agents.ipynb) | [Sheet](course/exercises/12_rag_agents.md) | [Quiz](course/quizzes/12_rag_agents.md) |
| 13 | Efficient inference and quantisation | [Deck](course/slides/13_efficiency.pptx) | [Notes](course/lectures/13_efficiency.md) | [Lab](course/tutorials/13_efficiency.ipynb) | [Sheet](course/exercises/13_efficiency.md) | [Quiz](course/quizzes/13_efficiency.md) |
| 14 | Paper clinic and peer review | [Deck](course/slides/14_paper_clinic.pptx) | [Notes](course/lectures/14_paper_clinic.md) | [Lab](course/tutorials/14_paper_clinic.ipynb) | [Sheet](course/exercises/14_paper_clinic.md) | [Quiz](course/quizzes/14_paper_clinic.md) |
| 15 | Defence and concept check | [Deck](course/slides/15_capstone.pptx) | [Notes](course/lectures/15_capstone.md) | [Lab](course/tutorials/15_capstone.ipynb) | [Sheet](course/exercises/15_capstone.md) | [Concept check](course/CONCEPT_CHECK.md) |

**Optional module:** [video, omni models and generative media](course/lectures/x1_video_omni.md), with its [notebook](course/tutorials/x1_video_omni.ipynb) and [sheet](course/exercises/x1_video_omni.md). No shared task uses video, so it is reading for interested students.

## Two words that change meaning

**Alignment** on day 5 means making a model's behaviour match what people want and what is safe. In SemEval Task 7, taught on day 11, it means the same content landing in the same place in a model's activations across languages.

## Capstone: one SemEval-2027 task per team

| Task | What the team builds |
|---|---|
| [6 MMCultureQA](https://mmcultureqa-semeval27.github.io/) | Answer culturally grounded questions about an image, asked in text or speech |
| [7 CLaS](https://clas-semeval2027.github.io/clas.semeval2027/task/) | Steer an LLM to answer in a target language by editing its activations |
| [9 Multimodal framing](https://propaganda.math.unipd.it/semeval2027task9/) | Label the frames in news paragraphs and news images |
| [12 ArtEmotion-28](https://github.com/artemotion-28/artemotion-28) | Predict emotion and sentiment for paintings across 28 languages, 11 of them African |

Evaluation runs 10–31 January 2027 and system papers are due in February 2027. Details, rules and the course milestones are in the [task cards](course/SEMEVAL_TASK.md), the [capstone brief](course/PROJECT.md) and the [paper guide](course/SEMEVAL_PAPER.md). The weekly [assignments](course/assignments/) tie each week's sheets to a capstone milestone.

## Assessment

- Three weekly portfolios, 15% each: five exercise sheets and that week's capstone milestone. See [week 1](course/assignments/week1.md), [week 2](course/assignments/week2.md), [week 3](course/assignments/week3.md).
- SemEval capstone, 45%: system, experiments, system-description paper, reproducible package and defence.
- Individual [concept check](course/CONCEPT_CHECK.md), 10%.
- Daily quizzes are formative by default. An instructor who wants them to count can use the option in the [syllabus](course/SYLLABUS.md#assessment).

## Materials and how to run them

- **Slides** are PowerPoint decks on the Imperial College London template. Their sources and the build command are in [course/slides/](course/slides/README.md).
- **Notebooks** run on a CPU with NumPy, offline, in seconds. Each ends with two functions for students to write against tests. Their data are small and synthetic so that the right answer is known; they measure no real model.
- **Shared-task work** uses the organisers' data and starter code, and the optional [pretrained-model scripts](course/extensions/README.md).
- **Checks:** `python scripts/check_course.py` and `python scripts/run_notebooks.py`. What has and has not been run is in [validation](course/VALIDATION.md).
- **Instructors:** start with the [instructor guide](course/instructor/README.md). Answer keys are in this repository; keep it private or change the numbers before using them for marks.

## Previous edition

The [previous outline](archive/README-previous-edition.md), [slides](slides/) and [practicals](practicals/) remain as background on n-grams, classification, word vectors and attention. Students enter having completed the [Google DeepMind AI Research Foundations path](https://www.skills.google/paths/3135).
