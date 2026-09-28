# Lecture 02 — Build and inspect a causal Transformer

[Course home](../../README.md) · Week 1, day 2 · 120 minutes

## Learning outcomes

- Trace tensor shapes through a decoder block.
- Implement stable causal attention and test future-token invariance.
- Explain positional encoding, residual connections, and modern decoder variants.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/02_transformers.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### The computational graph
For token IDs \(x\), embeddings produce \(X\in\mathbb R^{T\times d}\). A pre-norm decoder block computes \(H=X+\mathrm{Attention}(\mathrm{Norm}(X))\), then \(Y=H+\mathrm{MLP}(\mathrm{Norm}(H))\). The final normalized hidden state is projected to vocabulary logits. Residual streams carry information forward; normalization controls activation scale. LayerNorm centers and scales; RMSNorm scales without subtracting the mean.

### Attention is content-dependent mixing
Let \(Q=XW_Q\), \(K=XW_K\), and \(V=XW_V\). Then
\[A=\operatorname{softmax}((QK^\top)/\sqrt{d_k}+M),\qquad O=AV.\]
The softmax is row-wise. A causal mask sets entries with key position greater than query position to negative infinity. Padding masks are distinct: they suppress padding, not future positions. Subtract each row maximum before exponentiation. The \(\sqrt{d_k}\) factor controls dot-product variance under a simple independent-component model.

Multiple heads learn different mixing functions. Concatenate head outputs and project back to the residual dimension. A common modern feed-forward block uses a gated activation such as SwiGLU. These choices alter quality and cost; the notebook uses a deliberately small one-head forward pass so every number can be inspected.

### Position and context
Attention without positional information is permutation equivariant. Absolute embeddings add position vectors. Rotary position embeddings rotate query/key pairs as a function of position, making dot products sensitive to relative displacement. RoPE is not a guarantee of extrapolation beyond training length. Grouped-query attention shares key/value heads across several query heads; it reduces KV-cache size, not query-head count.

### Autoregression and training
Training uses teacher forcing: inputs \(x_{0:T-1}\) predict targets \(x_{1:T}\). A causal mask permits parallel training across positions. Generation repeatedly appends one sampled token and conditions the next prediction on it. The key test is causal invariance: modifying a future token must leave earlier logits unchanged. Attention visualizations can reveal patterns but are not sufficient explanations of causality in a model.

## Worked example

For T=4 and d_k=2, QKᵀ is 4×4. Query row 1 (zero-indexed) may use columns 0 and 1 only. Its weights sum to one over those two positions. A batch adds a leading B dimension; heads add H.

## Guided tutorial

**Question:** Implement a decoder forward pass in NumPy; inspect masked probabilities, vocabulary logits, and a future-token perturbation.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Compare the block with a Qwen3 configuration using `extensions/hf_text.py`; identify head counts and normalization choices in the model configuration.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Why does masking after softmax without renormalization change the scale?
2. Can a decoder train all positions in parallel without seeing the future?
3. What property would an all-zero future mask test fail to detect?

## After class

Complete the [exercise sheet](../exercises/02_transformers.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / rotary position embeddings](https://arxiv.org/abs/2104.09864)
- [Qwen3 Technical Report (2025)](https://arxiv.org/abs/2505.09388)
