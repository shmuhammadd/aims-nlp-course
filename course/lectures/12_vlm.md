# Lecture 12 — Vision-language models, document understanding, and grounding

[Course home](../../README.md) · Week 3, day 2 · 120 minutes

## Learning outcomes

- Explain how visual tokens condition an autoregressive decoder.
- Build a controlled visual question-answering evaluation.
- Use image interventions to detect reliance on language priors.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/12_vlm.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### Connecting vision to a language decoder
A common VLM combines a vision encoder, a projector or resampler, and a language decoder. Visual features must be mapped to the decoder hidden dimension. They may be inserted as tokens in the decoder sequence or used through cross-attention. A projector-only alignment stage can precede multimodal instruction tuning; training schedules differ across model families.

The conditional objective is \(-\sum_t\log p_\theta(y_t\mid y_{<t},x_{text},z_{image})\). As in text SFT, mask prompt and padding labels. Image preprocessing and token placement belong to the checkpoint's processor; a generic text tokenizer cannot safely substitute for it. Resolution, cropping, aspect ratio, and patch count affect both cost and information loss.

### Documents require detail
A chart question may require detecting axes, reading labels, identifying a mark, and performing arithmetic. OCR errors can propagate into numerically fluent but wrong answers. Tables encode relations in layout; flattening them without row/column structure can change meaning. For document QA, compare OCR-plus-text, image-only, and image-plus-text paths. Preserve page IDs, bounding boxes when available, and source evidence.

### Hallucination and interventions
A VLM can answer from a common language prior even when visual evidence disagrees. Compare an original image, blank image, relevant crop, shuffled image, and counterfactual edit under the same question. If the answer stays unchanged when evidence changes, inspect whether the question is answerable without vision or the model ignores visual inputs. Intervention effects are diagnostic, not a complete causal explanation.

Use separate metrics for answer correctness, evidential support, and abstention. In open-ended evaluation, match units and allow equivalent forms under a frozen rule. False-premise questions test whether a model invents evidence to satisfy the prompt. Count errors, attribute errors, spatial-relation errors, and reading errors need different examples.

### Recent systems and practical scope
Gemma 3 offers a case study in modern multimodal model design; do not assume every size in a family has identical modalities. The mandatory lab creates bar-chart pixels and reads them with an explicit geometric baseline, so students can construct controlled evidence and ablations. The SmolVLM extension performs actual learned VLM inference on the same chart and lets students compare a specialized baseline with a general model. A small VLM need not beat a purpose-built chart reader.

## Worked example

A 448×448 image with nonoverlapping 14×14 patches produces 32×32=1024 patch positions before any pooling or special tokens. Doubling each dimension gives 4096 positions, multiplying raw patch count by four. Actual processors may crop or resample, so inspect their outputs.

## Guided tutorial

**Question:** Generate chart pixels, extract bar heights, and audit blank/counterfactual image interventions.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Run `extensions/hf_vision.py --mode vlm` for learned visual question answering; inspect the image, prompt, processor configuration, and generated answer together.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why is a blank-image baseline useful?
2. Can a correct numerical answer be visually ungrounded?
3. Why might a specialized chart parser outperform a small VLM on this lab?

## After class

Complete the [exercise sheet](../exercises/12_vlm.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Multimodal chat templates](https://huggingface.co/docs/transformers/main/chat_template_multimodal)
