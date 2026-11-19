# Reading guide

[Course home](../README.md)

Each session has one required reading of 15–20 minutes, done before the next morning's quiz. The session notes list further readings under "for your task" and "going deeper", each with a line on what to read and why. Every arXiv link in the course was checked against its title on 16 November 2026.

## Required reading, one per session

| Day | Session | Read | Sections |
| --- | --- | --- | --- |
| 1 | Tokens and baselines | [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) (Petrov et al., 2023) | 1–3 |
| 2 | Causal Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | 3.1–3.3 |
| 3 | Pretraining and decoding | [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019) | 1–3 |
| 4 | Prompting, tuning, LoRA | [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) | 1 and 4.1 |
| 5 | Alignment | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) | 1, 3, 4 |
| 6 | Multilingual and data building | [AfriSenti](https://aclanthology.org/2023.emnlp-main.862/) (Muhammad et al., 2023) | Data collection and annotation |
| 7 | CLIP and SigLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021) | 1–2, figure 1 |
| 8 | Vision-language models | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (Liu et al., 2023) | 3–4 |
| 9 | Speech and ASR | [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., 2022) | 1–2, limitations |
| 10 | Benchmarking and evaluation | [BERTScore](https://arxiv.org/abs/1904.09675) (Zhang et al., 2019) | 1–3 |
| 11 | Steering | [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (Arditi et al., 2024) | 1–3 |
| 12 | Retrieval, tools, agents | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) | 1–3 |
| 13 | Efficiency | [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) | 1–3 |
| 14 | Paper clinic | [Guidelines for SemEval System Papers](https://semeval.github.io/system-paper-template.html) | All, with your draft open |
| 15 | Defence | Your own paper and the review you received | |

The section numbers were taken from the papers' structure as remembered, not re-read page by page. Check each one when you set the reading.

## Reading for your task

| Task | Start with | Then |
| --- | --- | --- |
| 12 ArtEmotion-28 | The [task repository](https://github.com/artemotion-28/artemotion-28) and its baselines | [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP 2](https://arxiv.org/abs/2502.14786), [XLM-R](https://arxiv.org/abs/1911.02116), [AfroXLM-R](https://arxiv.org/abs/2204.06487) |
| 6 MMCultureQA | The [task site](https://mmcultureqa-semeval27.github.io/) | [BERTScore](https://arxiv.org/abs/1904.09675), [An Introduction to Vision-Language Modeling](https://arxiv.org/abs/2405.17247), [Whisper](https://arxiv.org/abs/2212.04356), [BLEnD](https://arxiv.org/abs/2406.09948) |
| 7 CLaS | [CLaS-Bench](https://aclanthology.org/2026.findings-acl.1086/) and the [task repository](https://github.com/d-gurgurov/CLaS-SemEval-2027) | [Language-Specific Neurons](https://arxiv.org/abs/2402.16438), [Do Llamas Work in English?](https://arxiv.org/abs/2402.10588), [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) |
| 9 Framing | The [task site](https://propaganda.math.unipd.it/semeval2027task9/) and its trial data | [XLM-R](https://arxiv.org/abs/1911.02116), [CLIP](https://arxiv.org/abs/2103.00020), [LoRA](https://arxiv.org/abs/2106.09685) |
| AfriGuard | The [task page](https://africanlp.github.io/2027/shared-task) and [paper](https://arxiv.org/abs/2610.26474) | [LoRA](https://arxiv.org/abs/2106.09685), then the day 5 and day 11 readings |

## African-language resources named in the course

- [AfriSenti](https://aclanthology.org/2023.emnlp-main.862/): sentiment in 14 languages (day 6).
- [AfriHate](https://arxiv.org/abs/2501.08284): hate speech and abusive language (day 6).
- [IrokoBench](https://arxiv.org/abs/2406.03368) and [AfroBench](https://arxiv.org/abs/2311.07978): LLM benchmarks (day 10).
- [AfriQA](https://arxiv.org/abs/2305.06897): cross-lingual question answering (day 12).
- [FLEURS](https://arxiv.org/abs/2205.12446): speech in 102 languages (day 9).
- The [safety thread](SAFETY.md) lists the 2026 preprints on safety in Hausa, Yoruba, Nigerian Pidgin and Igbo.

## Reading a recent preprint

Several readings are 2025–2026 preprints. For each one, fill a claim-evidence table: the claim, the table or figure behind it, the data size, the baseline, and what the paper cannot rule out. Mark numbers as author-reported. A preprint that nobody has reproduced is a source of hypotheses.

## Books

No single book covers the course. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (Jurafsky and Martin) fills gaps in the foundations. [An Introduction to Vision-Language Modeling](https://arxiv.org/abs/2405.17247) (Bordes et al., 2024) is a free primer for week 2. Two paid books, *Vision Language Models* (O'Reilly, 2026) and *Large Vision-Language Models* (Springer), are optional; see [books/README.md](../books/README.md).

## For a later cohort

Before teaching again, re-check the task cards, the recent preprints and the model versions. Replace a case study that has been superseded and keep the lesson it carried. Freeze model revisions and prompts for the length of the course.
