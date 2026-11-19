# Syllabus — Advanced Language and Multimodal Models

[Course home](../README.md)

## Audience and prerequisites

AIMS students with linear algebra, probability, calculus, Python and introductory machine learning. Students should know matrix multiplication, gradients, train/test separation and basic classification metrics. The [diagnostic](PREREQUISITES.md) shows what to revise; it is not a gate. The labs need NumPy only. PyTorch helps with the shared task.

## Prior learning: DeepMind AI Research Foundations

Students have completed the Google DeepMind AI Research Foundations path, which teaches building and fine-tuning a small language model. Week 1 still teaches tokens, the Transformer and pretraining in full, because the rest of the course rests on them. The day-1 diagnostic sets the depth: where the class already knows a derivation, shorten the board work and give the time to the lab. The badge is not course evidence; the course is assessed on its own work.

## The three weeks

| Week | Theme | Sessions | Capstone milestone on Friday |
|---|---|---|---|
| 1 | Foundations and post-training | 1 tokens and baselines · 2 Transformer · 3 pretraining and decoding · 4 prompting, instruction tuning, LoRA · 5 alignment | Organisers' baseline reproduced |
| 2 | Languages, modalities and benchmarking | 6 multilingual and data building · 7 CLIP and SigLIP · 8 vision-language models · 9 speech · 10 benchmarking and evaluation | One improvement; experiment plan frozen |
| 3 | Inside the model, systems and the paper | 11 interpretability and steering · 12 retrieval, tools, agents · 13 efficiency · 14 paper clinic · 15 defence | Paper, package and defence |

Images arrive on day 7. Teams therefore begin with text in week 1: Task 12 requires a text-only run as well as a multimodal one, and Tasks 7 and 9 start from text. Task 6 teams run the organisers' starter code as given from day 1 and open it up in week 2.

Video, omni models and generative media are an [optional module](lectures/x1_video_omni.md). No shared task uses them.

## Outcomes and evidence

| Outcome | Sessions | Evidence |
|---|---|---|
| Explain model objectives, tensor shapes and decoding | 1–3 | Tokenisation audit, causal-invariance test, training curve, sampling table |
| Adapt and align a model, and say what each training signal teaches | 4–5 | Prompt and adapter comparison, DPO calculation, refusal check |
| Build and audit multilingual data | 6 | Agreement statistic, data statement, per-language scores |
| Work with image, vision-language and speech models | 7–9 | Contrastive retrieval, image interventions, WER and CER |
| Benchmark a system honestly | 10 | Task metric by hand, paired test, calibration table |
| Read and write a model's activations | 11 | Steering vector, compliance and relevance sweep, ablation |
| Build and bound a system | 12–13 | Retrieval ablation, validated tool calls, memory budget |
| Do reproducible research and write it up | 1–15 | Shared-task predictions, system paper, peer review, defence |
| Evaluate safety across languages and modalities | 4, 5, 6, 8, 10, 11 | The [safety thread](SAFETY.md) outputs |

## A session

Every session is 120 minutes: 10 quiz on the previous day, 25 concepts, 25 derivation and worked example, 5 break, 40 notebook lab, 10 discussion, 5 exit ticket. That is 30 contact hours, 10 of them in labs. Day 14 is a writing workshop and day 15 holds the defences; their plans are in the session notes.

Outside class, plan 45–60 minutes a day for the exercise sheet, 15–20 minutes for the required reading, and 12–16 hours over the three weeks for the capstone. Total workload is about 60–70 hours. Official evaluation (10–31 January 2027) and the paper deadline (February 2027) fall after the course.

## Assessment

| Component | Weight | What is marked |
|---|---:|---|
| Portfolio 1 (week 1) | 15% | Five exercise sheets and the baseline milestone: [assignment](assignments/week1.md) |
| Portfolio 2 (week 2) | 15% | Five sheets and the frozen plan: [assignment](assignments/week2.md) |
| Portfolio 3 (week 3) | 15% | Five sheets and the peer review: [assignment](assignments/week3.md) |
| Shared-task capstone | 45% | [Rubric](PROJECT.md): paper 25, evaluation 25, system 20, reproducibility 15, task protocol 10, defence 5 |
| Concept check | 10% | [Individual, closed book](CONCEPT_CHECK.md), 25 minutes |

Each sheet is marked out of 10: derivation 2, implementation 4, investigation 4. Portfolios are due at the end of days 5, 10 and 15.

Overall grade = 0.15 × (P1 + P2 + P3) × 2 + 0.45 × project + 0.10 × concepts × 5, with each portfolio out of 50, the project out of 100 and the concept check out of 20.

**Quizzes.** Fourteen 10-minute quizzes open days 2 to 15. They are formative by default: mark them in class and use the results to choose what to revisit. An instructor who wants more closed-book individual work can count the best ten quizzes for 9% and reduce each portfolio from 15% to 12%. Announce any such change before day 1.

Course marks do not depend on leaderboard rank or on a paper being accepted. A careful negative result can earn full marks.

## Working rules

Students may discuss, use documentation and use coding assistants, and must disclose what they used. Each student must be able to explain every line they submit. No invented runs, citations, intervals or measurements. Sheets may be explored in pairs and are written up alone. Capstone teams record who did what.

Use permitted data and consented media. Keep private recordings and personal information out of public repositories. Where a project makes claims about a language, a speaker of that language should check the outputs, or the paper should say that nobody did.

## Access and contingency

The labs run offline once NumPy is installed. The capstone needs the task data and, for some tasks, model downloads and a GPU: pair students for scarce GPU time and download large files before class. Printed notes and saved notebook outputs support a student without a working laptop. Provide captions or transcripts for any audio or video shown in class.
