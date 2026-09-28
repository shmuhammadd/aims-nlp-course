# Instructor guide

[Course home](../../README.md) · [Syllabus](../SYLLABUS.md) · [Project](../PROJECT.md)

## Before the course

Run the [checks](../SETUP.md) on the actual teaching platform. Install core dependencies before class; predownload optional checkpoints only after checking model cards and available disk/RAM. No paid service is required. Keep a CPU-only environment and notebook outputs as a backup. The exact status of local checks is in [VALIDATION.md](../VALIDATION.md).

Give students the prerequisite diagnostic a week early. In day 1, use their answers to decide whether the matrix/gradient recap needs more board time. The new notes are lecture-ready source material rather than PowerPoint decks; use their examples and equations directly or adapt them to your presentation format. Old PDF slides are supplementary historical material.

## SemEval capstone preparation

Complete the [task specification](../SEMEVAL_TASK.md) before the cohort starts development: select the year/task, required tracks/languages, allowed resources, metric/scorer, data access, and exact deadlines. The user has chosen SemEval, but has not yet specified those identifiers. Do not silently reuse the archived task selection.

Introduce the [paper guide](../SEMEVAL_PAPER.md) in week 1. Check the baseline and outline on day 5, methods draft on day 10, peer review on day 14, and complete course paper on day 15. Reserve the same assessment weight for a strong negative result. If the official evaluation window falls later, grade the complete development-results paper and validated submission workflow, then schedule official evaluation and paper revision separately.

The daily synthetic labs remain teaching demonstrations; the capstone must follow the selected task’s data and rules. Do not require an extra modality if the task is text-only.

## Daily facilitation

Ask students to predict an invariant before running code. Walk through the first code block, then let pairs inspect the intervention. In discussion, ask “what evidence would make this conclusion false?” Use the final five-minute ticket to record one measured result, one limitation, and one next experiment. Address the most common confusion in the next day's opening quiz.

Do not let model installation consume the guided tutorial. Core labs deliberately make mechanisms visible on small data; model extensions connect them to practical systems. Keep the distinction explicit. In weeks 2–3, revisit a single capstone example so students see how retrieval, perception, reasoning, and evaluation interact.

## A measured overfitting example

The [tiny-decoder trace](../validation/tiny_decoder_run.json) is an actual CPU run. Ask students to choose a checkpoint using development loss before showing the final steps. Around step 60 the development curve stops improving while training loss continues down. This is useful if the class cannot install PyTorch during week 1.

## Differentiation

For students needing support, give the worked derivation and ask them to explain the assertions before modifying code. For advanced students, use the optional pretrained pathway, gradient checks, multi-positive retrieval, clustered bootstrap, or compositional holdouts. Do not add marks for compute expenditure.

The practical questions are formative; answer guides explain the intended concepts. Exercise investigations can have several defensible answers. Grade design and evidence, not agreement with an expected performance ordering. Each daily guide uses the same 2/4/4 anchors.

## Answer guides

- [01. Language models, tokens, and experimental baselines](01_tokenization.md)
- [02. Build and inspect a causal Transformer](02_transformers.md)
- [03. Pretraining: data, objectives, scaling, and small-model training](03_pretraining.md)
- [04. Instruction tuning and parameter-efficient adaptation](04_adaptation.md)
- [05. Preference optimization and reinforcement learning for reasoning](05_preferences.md)
- [06. Multilingual and low-resource language modelling](06_multilingual.md)
- [07. Retrieval-augmented generation and evidence-based answers](07_rag.md)
- [08. Reasoning, test-time compute, and tool-using systems](08_reasoning.md)
- [09. Efficient inference, compression, and deployment tradeoffs](09_efficiency.md)
- [10. Reliable evaluation, calibration, and responsible reporting](10_evaluation.md)
- [11. Multimodal representations: CLIP, SigLIP, and cross-modal retrieval](11_contrastive.md)
- [12. Vision-language models, document understanding, and grounding](12_vlm.md)
- [13. Speech, audio representations, and multilingual ASR](13_speech.md)
- [14. Video, unified multimodal systems, and generative frontiers](14_video_omni.md)
- [15. SemEval systems, scientific writing, and project defence](15_capstone.md)

[Concept-check key](concept_check_key.md)

Answers are visible in public clones. Release teaching copies after deadlines where appropriate and change instances for summative checks.
