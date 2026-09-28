# Research and model guide

[Course home](../README.md)

**Research check: 28 September 2026.** The course combines established mechanisms with selected 2025–2026 research. This is a dated teaching snapshot, not an exhaustive leaderboard survey. Recent preprints are critical-reading material; reported gains require scrutiny of data, baselines, evaluation budgets, and independent reproduction. Small classroom checkpoints are chosen for accessibility rather than state-of-the-art scores.

## Priority reading map

Read the abstract, the objective or architecture section, and one result/ablation per assigned paper. Limit core preparation to 10–15 minutes daily; deep reading is optional.

| Sessions | Foundation | Contemporary case study or practical reference | Question to bring to class |
|---|---|---|---|
| 1–3 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [Chinchilla](https://arxiv.org/abs/2203.15556) | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | Which conclusions depend on data and compute allocation? |
| 4 | [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314) | [PEFT documentation](https://huggingface.co/docs/peft/quicktour) | Which memory terms are actually reduced? |
| 5, 8 | [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath](https://arxiv.org/abs/2402.03300) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | What does the reward or preference signal fail to measure? |
| 6 | [XLM-R](https://arxiv.org/abs/1911.02116) | [AfriSenti](https://aclanthology.org/2023.emnlp-main.862/) | Which languages, varieties, domains, and annotators are represented? |
| 7 | [RAG](https://arxiv.org/abs/2005.11401) | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Is failure caused by retrieval, context use, or generation? |
| 9 | [FlashAttention](https://arxiv.org/abs/2205.14135), [PagedAttention](https://arxiv.org/abs/2309.06180) | [Speculative decoding](https://arxiv.org/abs/2211.17192) | What workload and hardware produce the reported speedup? |
| 10 | [HELM](https://arxiv.org/abs/2211.09110), [calibration](https://arxiv.org/abs/1706.04599) | [Model cards](https://arxiv.org/abs/1810.03993) | What is missing from the aggregate metric? |
| 11 | [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343) | [SigLIP 2](https://arxiv.org/abs/2502.14786) | Does alignment establish counting or spatial understanding? |
| 12 | [LLaVA](https://arxiv.org/abs/2304.08485) | [Gemma 3](https://arxiv.org/abs/2503.19786), [SmolVLM card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | What visual information is lost during preprocessing? |
| 13 | [wav2vec 2.0](https://arxiv.org/abs/2006.11477), [Whisper](https://arxiv.org/abs/2212.04356) | [Whisper implementation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/whisper) | How do normalization and speaker splits change the conclusion? |
| 14 | [DDPM](https://arxiv.org/abs/2006.11239), [flow matching](https://arxiv.org/abs/2210.02747) | [Qwen3.5-Omni (2026)](https://arxiv.org/abs/2604.15804) | Are modality coverage, temporal grounding, and streaming evaluated separately? |
| 15 | Controlled ablations, the selected task protocol, and the [SemEval system-paper guide](https://semeval.github.io/system-paper-template.html) | [ReactBench (May 2026)](https://arxiv.org/abs/2605.29579), [KnowHal (August 2026)](https://arxiv.org/abs/2608.03782) | Which failures can the benchmark diagnose, and which causal claims exceed its evidence? |

## Current-research discussion briefs

**Qwen3 (2025):** inspect dense versus mixture-of-experts design and the relationship between reasoning modes, inference budget, and multilingual evaluation. The 0.6B checkpoint is a feasible teaching example; results for much larger family members do not transfer automatically to it.

**SigLIP 2 (2025):** identify the combined training objectives and multilingual data choices. Use ablations to distinguish effects of the objective, data mixture, and model scale. A stronger encoder does not alone settle downstream grounding.

**Gemma 3 (2025):** examine how local/global attention and visual processing interact with long-context inference. Check modality support by checkpoint; family-level descriptions are not a substitute for a model card.

**Qwen3.5-Omni (April 2026):** read as a contemporary technical report on multimodal systems. Build a claim-evidence table with section/table references rather than copying benchmark headlines. The course neither requires running this model nor claims to reproduce its results.

**ReactBench and KnowHal (2026 preprints):** compare targeted visual interventions with paired true/false-premise questions. Critique evaluation construction, reliance on model-generated material, and how well labels distinguish perception, knowledge, and reasoning errors. Treat causal interpretations cautiously when based on verbal explanations.

## Updating for a later cohort

Before teaching, confirm the date and version of the recent readings, access to model weights, library compatibility, and local hardware performance. Replace a case study if it is superseded while retaining the objective, ablation, and evaluation lesson. Record changes in the validation log. Do not silently update checkpoints mid-assessment: freeze revisions and prompts so student comparisons remain interpretable.

The original [Speech and Language Processing online draft](https://web.stanford.edu/~jurafsky/slp3/) remains a useful reference for foundational gaps. The old books/slides directory is not a prerequisite for accessing these primary sources.
