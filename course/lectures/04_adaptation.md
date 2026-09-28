# Lecture 04 — Instruction tuning and parameter-efficient adaptation

[Course home](../../README.md) · Week 1, day 4 · 120 minutes

## Learning outcomes

- Construct response-only supervised fine-tuning labels.
- Derive and train a low-rank weight update.
- Compare adaptation methods under a fixed data and compute budget.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/04_adaptation.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### From text continuation to instruction following
Supervised fine-tuning (SFT) maximizes likelihood of desired responses conditional on prompts. A chat template serializes role boundaries and special tokens. A template mismatch between training and inference can change behavior even when the words are identical. Split by task/source before producing prompt variants so paraphrases do not leak.

For response-only SFT, prompt and padding positions have ignored labels (commonly -100), while assistant response positions contribute to the causal objective. The model still attends to prompt tokens. The model implementation usually shifts labels internally; do not shift them a second time. For multi-turn data, decide whether every assistant span or only the final response is supervised. Inspect actual token IDs and labels before training.

### LoRA algebra
For a frozen matrix \(W\in\mathbb R^{d_{in}\times d_{out}}\), write \(W'=W+sAB\), with \(A\in\mathbb R^{d_{in}\times r}\), \(B\in\mathbb R^{r\times d_{out}}\), and typically \(s=\alpha/r\). This course uses row-vector inputs, so \(Y=X(W+sAB)\). Many libraries store the transposed weight; check conventions rather than copying shapes blindly.

Only \(r(d_{in}+d_{out})\) adapter parameters train. With random A and zero B, the initial function matches the base model. On the first update, A has zero gradient while B generally does not; making both factors zero would block learning. LoRA reduces optimizer state for trainable weights but does not eliminate base weights or activation memory.

### Quantization and adaptation
QLoRA combines a frozen quantized base with trainable adapters and other memory-saving techniques. It is not simply training all parameters in four-bit arithmetic. Quantization introduces approximation error, and kernels/device support affect feasibility. Our NumPy lab learns a low-rank residual in a linear task, making updates and limitations visible. The optional model extension performs actual response-only causal-LM LoRA training.

### Experimental discipline
Compare the unadapted checkpoint, prompt-only baseline, and adapter under the same held-out protocol. Use a small learning-rate search on development data. Track trainable parameters, peak memory where measurable, time, output format validity, and task quality. Better training fit on eight examples establishes pipeline operation, not generalization. Inspect failures by language and prompt style.

## Worked example

For a 4096×4096 projection with rank 8, full fine-tuning updates 16,777,216 weights; the adapter updates 65,536 (0.390625%). Biases and other adapted layers would add parameters. A reduction in trainable weights does not imply the same percentage reduction in total memory.

## Guided tutorial

**Question:** Fit a low-rank adapter to a frozen linear model and inspect response-only label masking.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Run `extensions/hf_lora.py` for a small actual SFT experiment; inspect the printed trainable parameters and token masks before interpreting loss.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why initialize only one LoRA factor to zero?
2. Should prompt tokens be removed from the input for response-only SFT?
3. What must be saved alongside an adapter to reproduce inference?

## After class

Complete the [exercise sheet](../exercises/04_adaptation.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [PEFT quick tour](https://huggingface.co/docs/peft/quicktour)
