# Syllabus — Advanced Language and Multimodal Models

[Course home](../README.md)

## Audience and prerequisites

AIMS students with linear algebra, probability, calculus, Python, and introductory machine learning. No previous LLM API experience is required. Students should know matrix multiplication, gradients, train/test separation, and basic classification metrics. Use the [diagnostic](PREREQUISITES.md) to identify preparation needs; it is not a gatekeeping exam. PyTorch familiarity helps the optional model extensions but is not required for the NumPy core.

## Outcomes and evidence

| Outcome | Where developed | Evidence |
|---|---|---|
| Explain model objectives and tensor operations | 1–3 | Tokenization audit, causal-invariance check, training curve |
| Adapt models and critique training signals | 4–6 | Adapter experiment, preference analysis, language-slice report |
| Build evidence and tool pipelines | 7–8 | Retrieval ablation and bounded tool trace |
| Measure efficiency and reliability | 9–10 | Memory calculation, quantization audit, calibration report |
| Analyze multimodal representations and perception | 11–14 | Contrastive retrieval, image interventions, WER, temporal sampling |
| Conduct reproducible research | 1–15 | SemEval prediction package, system-description paper, ablations, oral defence |

## Weekly arc

**Week 1 — Model mechanisms and adaptation.** Move rapidly from tokenization and experimental design to causal Transformers, pretraining, instruction tuning, low-rank adaptation, and preference/reward optimization. The foundational recap is compressed; n-grams and classical classification remain support material in the previous edition.

**Week 2 — Systems and scientific evaluation.** Study multilingual transfer, retrieval, reasoning and tools, memory and compression, and evaluation. Students use the selected SemEval task protocol and freeze their main experiment plan by day 10, before selecting final results.

**Week 3 — Multimodal learning and integration.** Align images and text, evaluate visual language models and document understanding, analyze speech and video, and transfer evaluation lessons to the SemEval system-description paper and defence. The selected shared task may be text-only or multimodal. Generative media objectives are introduced for comparison; training a large media generator is outside the course scope.

## Contact time and independent work

Every session is exactly 120 minutes: 10 retrieval quiz, 25 concepts, 25 derivation/examples, 5 break, 40 tutorial, 10 practical discussion, 5 exit ticket. This totals 30 contact hours, including 10 hours of guided tutorials. Day 15 uses the same time blocks for a SemEval paper clinic and defence preparation; presentation substitutions are described in the project brief.

Plan 45–60 minutes per day for exercises (11.25–15 hours total), 10–15 minutes for a targeted reading (2.5–3.75 hours), and 12–16 hours for SemEval development and paper writing. Total in-course workload is approximately 56–65 hours. Official shared-task evaluation and paper revisions after the course require separately announced time. Instructors should reduce optional reading before extending the working day. Exercises can use pairs for exploration, with individual explanations submitted. The SemEval capstone can be done in pairs or individually, with contribution records and defence questions.

## Assessment

Each daily sheet is scored out of 10: derivation 2, implementation 4, investigation 4. Five sheets form a weekly portfolio out of 50; each portfolio contributes 15 course percentage points. Daily practical questions and exit tickets are formative. Weekly portfolios are due after days 5, 10, and 15; instructors announce the local submission time before day 1.

The capstone is participation in an instructor-selected SemEval shared task and a complete system-description paper. It is scored out of 100 and contributes 45%. The [rubric](PROJECT.md) allocates 25 points to the paper, 25 to evaluation/analysis, 20 to system development, 15 to reproducibility/submission readiness, 10 to task/data protocol, and 5 to the defence. Use the [task specification](SEMEVAL_TASK.md) and [paper guide](SEMEVAL_PAPER.md). Course grading does not depend on leaderboard rank or workshop acceptance. If official evaluation is later, day 15 requires a complete development-results paper and a submission-ready workflow; official participation follows the organiser schedule.

The [individual concepts check](CONCEPT_CHECK.md), out of 20, contributes 10%. Run it for 25 minutes during the final independent-work period or an agreed assessment slot outside the 30 contact hours. If all assessment must fit the contact hours, replace 25 minutes of day-15 project clinic with the check and move that clinic to office hours.

Overall grade = 0.15×P1/50×100 + 0.15×P2/50×100 + 0.15×P3/50×100 + 0.45×project + 0.10×concepts/20×100.

## Participation and reproducibility

Students may discuss approaches and use documentation, coding assistants, or other tools if disclosed. Each must explain their code, verify outputs, cite external ideas, and distinguish generated text from measured evidence. Never fabricate runs, citations, confidence intervals, or compute measurements. Public answer guides are for learning; instructors should change instances for high-stakes assessment.

Use permitted datasets and consented media. Do not submit private recordings or identifiable student information in public repositories. Synthetic data are appropriate for the daily mechanism labs. The capstone uses the selected SemEval task data and rules; a synthetic-only lab does not replace shared-task work. Projects using natural language claims should include data and interpretation from people able to assess that language, or acknowledge the lack of such validation.

## Accessibility and contingency

All core labs work offline after installing NumPy. The SemEval capstone additionally requires access to the selected task data and, during official participation, its submission platform. Choose a feasible baseline and arrange permitted local data access before class. Pair students for scarce GPU access and run optional large downloads before class. Printed notes plus notebook outputs support students without a working laptop. Core labs avoid mandatory accounts and paid APIs. Use captions/transcripts for any instructor-selected audio/video demonstration and provide an equivalent text-based analysis option.
