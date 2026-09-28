# Advanced Language and Multimodal Models

**AIMS · 2026 edition · 3 intensive weeks · 5 days per week · 2 hours per day · 30 contact hours**

A practical, mathematically grounded course on how modern language and multimodal models are trained, adapted, evaluated, and used in evidence-based systems. The course retains a focus on African languages, multilingual evaluation, and research under limited compute. Students build small mechanisms from first principles, then investigate pretrained models through optional extensions.

By the end, students will be able to implement causal attention and next-token training; explain SFT, LoRA, preference learning, and reasoning-time computation; build and evaluate retrieval and tool pipelines; analyze image-text, document, speech, and video systems; and defend a reproducible research result with controlled ablations.

## Start here

1. Read the [syllabus and assessment plan](course/SYLLABUS.md).
2. Complete the [prerequisite diagnostic](course/PREREQUISITES.md) and [environment setup](course/SETUP.md).
3. Follow each day's notes, notebook, practical questions, and exercise sheet below.
4. Start the [SemEval capstone](course/PROJECT.md) in week 1: develop a shared-task system and write a system-description paper. See the [task specification](course/SEMEVAL_TASK.md) and [paper-writing guide](course/SEMEVAL_PAPER.md).

**Scheduling assumption:** the two-hour daily slot includes explanation, board work, a short break, a 40-minute guided tutorial, and discussion. Exercises take 45–60 minutes after class; SemEval system development and paper writing are separately budgeted at 12–16 hours across three weeks. Official evaluation and paper deadlines may require follow-up after the course. Optional extensions are enrichment, not additional compulsory daily work. If two hours must be entirely lecture, run the same tutorial in a separate practical slot and add that time to the timetable.

## The 15-session course

Each notes file contains learning outcomes, a timed teaching plan, explanations and equations, a worked example, three practical discussion questions, and readings. Each exercise sheet has a derivation, an implementation task, and an investigation (10 points total).

| Week/day | Lecture | Teaching notes | Hands-on tutorial | Exercises |
|---|---|---|---|---|
| 1/1 | 01. Language models, tokens, and experimental baselines | [Notes](course/lectures/01_tokenization.md) | [Notebook](course/tutorials/01_tokenization.ipynb) | [Sheet](course/exercises/01_tokenization.md) |
| 1/2 | 02. Build and inspect a causal Transformer | [Notes](course/lectures/02_transformers.md) | [Notebook](course/tutorials/02_transformers.ipynb) | [Sheet](course/exercises/02_transformers.md) |
| 1/3 | 03. Pretraining: data, objectives, scaling, and small-model training | [Notes](course/lectures/03_pretraining.md) | [Notebook](course/tutorials/03_pretraining.ipynb) | [Sheet](course/exercises/03_pretraining.md) |
| 1/4 | 04. Instruction tuning and parameter-efficient adaptation | [Notes](course/lectures/04_adaptation.md) | [Notebook](course/tutorials/04_adaptation.ipynb) | [Sheet](course/exercises/04_adaptation.md) |
| 1/5 | 05. Preference optimization and reinforcement learning for reasoning | [Notes](course/lectures/05_preferences.md) | [Notebook](course/tutorials/05_preferences.ipynb) | [Sheet](course/exercises/05_preferences.md) |
| 2/1 | 06. Multilingual and low-resource language modelling | [Notes](course/lectures/06_multilingual.md) | [Notebook](course/tutorials/06_multilingual.ipynb) | [Sheet](course/exercises/06_multilingual.md) |
| 2/2 | 07. Retrieval-augmented generation and evidence-based answers | [Notes](course/lectures/07_rag.md) | [Notebook](course/tutorials/07_rag.ipynb) | [Sheet](course/exercises/07_rag.md) |
| 2/3 | 08. Reasoning, test-time compute, and tool-using systems | [Notes](course/lectures/08_reasoning.md) | [Notebook](course/tutorials/08_reasoning.ipynb) | [Sheet](course/exercises/08_reasoning.md) |
| 2/4 | 09. Efficient inference, compression, and deployment tradeoffs | [Notes](course/lectures/09_efficiency.md) | [Notebook](course/tutorials/09_efficiency.ipynb) | [Sheet](course/exercises/09_efficiency.md) |
| 2/5 | 10. Reliable evaluation, calibration, and responsible reporting | [Notes](course/lectures/10_evaluation.md) | [Notebook](course/tutorials/10_evaluation.ipynb) | [Sheet](course/exercises/10_evaluation.md) |
| 3/1 | 11. Multimodal representations: CLIP, SigLIP, and cross-modal retrieval | [Notes](course/lectures/11_contrastive.md) | [Notebook](course/tutorials/11_contrastive.ipynb) | [Sheet](course/exercises/11_contrastive.md) |
| 3/2 | 12. Vision-language models, document understanding, and grounding | [Notes](course/lectures/12_vlm.md) | [Notebook](course/tutorials/12_vlm.ipynb) | [Sheet](course/exercises/12_vlm.md) |
| 3/3 | 13. Speech, audio representations, and multilingual ASR | [Notes](course/lectures/13_speech.md) | [Notebook](course/tutorials/13_speech.ipynb) | [Sheet](course/exercises/13_speech.md) |
| 3/4 | 14. Video, unified multimodal systems, and generative frontiers | [Notes](course/lectures/14_video_omni.md) | [Notebook](course/tutorials/14_video_omni.ipynb) | [Sheet](course/exercises/14_video_omni.md) |
| 3/5 | 15. SemEval systems, scientific writing, and project defence | [Notes](course/lectures/15_capstone.md) | [Notebook](course/tutorials/15_capstone.ipynb) | [Sheet](course/exercises/15_capstone.md) |

## Practical design

All 15 core notebooks run on a CPU with NumPy and contain their own authored or synthetic data. They require no API key, paid service, or model download. Each includes checks, a controlled intervention, interpretation prompts, and links to follow-on work. Synthetic examples are explicitly labelled and must not be presented as benchmark evidence.

The [real-model extensions](course/extensions/README.md) cover Qwen3 text inference, a tiny Transformer trained from scratch, response-only LoRA SFT, AfriSenti classification from permitted local exports, CLIP retrieval, SmolVLM visual QA, Whisper ASR, and optional video inference. These need additional packages and model downloads; hardware estimates are planning guidance, not measured guarantees. See [validation status](course/VALIDATION.md) for what was actually executed.

## Assessment and teaching support

- Weekly portfolios: 45% (three portfolios, five daily sheets each).
- SemEval capstone: 45% (shared-task system, official-format prediction package, experiments, system-description paper, reproducibility, and defence).
- Individual concepts check: 10%.

The [project brief](course/PROJECT.md) includes three-week milestones, a paper-focused marking rubric, and a submission checklist. The SemEval year, task, and required subtasks/languages will be set by the instructor; official participation dates are separate from course deadlines. [Instructor guidance](course/instructor/README.md) includes answer guides for all 15 sessions and a concept-check marking key. Answers are visible in a public repository: use fresh instances for summative assessment.

Research readings combine foundations with 2025–2026 case studies. See the [source and model guide](course/READINGS.md) for the research cutoff, primary sources, and distinctions between established methods and recent preprints. State of the art here means current methods and critical evaluation, not a claim that one checkpoint leads every benchmark.

## Previous edition

The [previous course outline](archive/README-previous-edition.md), existing [slides](slides/), and [practicals](practicals/) are retained. They are historical foundation resources; their dependencies and external links have not been revalidated for this edition. The new teaching material is under `course/`. No existing slide deck or practical has been replaced.

The earlier [Google DeepMind AI Research Foundations learning path](https://www.skills.google/paths/3135) is optional enrichment in this three-week design; its former badge requirement is recorded in the archived outline. Assigned core readings use primary papers and public documentation; students do not need the bundled books directory to complete the course.
