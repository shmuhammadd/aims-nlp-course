# Lecture 07 — Retrieval-augmented generation and evidence-based answers

[Course home](../../README.md) · Week 2, day 2 · 120 minutes

## Learning outcomes

- Separate retrieval quality from answer quality.
- Build a retrieval baseline with source identifiers and abstention.
- Design oracle, missing-evidence, and distractor ablations.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Retrieval quiz: recall the previous session; day 1 uses the prerequisite diagnostic |
| 10–35 | Concepts and concrete examples from the first half of these notes |
| 35–60 | Board derivation, worked example, and failure analysis |
| 60–65 | Break |
| 65–105 | Guided [notebook tutorial](../tutorials/07_rag.ipynb) |
| 105–115 | Practical questions in pairs, then discuss evidence |
| 115–120 | Exit ticket: one result, one limitation, one next experiment |

## Teaching notes

### Decompose the system
A retrieval-augmented generation (RAG) system indexes evidence, retrieves candidates, optionally reranks them, builds a context, and generates an answer. The language model is only one component. A fluent response with a citation can still be unsupported; a relevant document may still be ignored. Evaluate each stage.

Sparse retrieval uses term overlap, often weighted by inverse document frequency. BM25 also handles term-frequency saturation and document length. Dense retrieval compares learned representations and can recover paraphrases but may fail on exact identifiers. A hybrid system combines signals; a reranker scores query-document pairs jointly. The lab implements a transparent TF-IDF cosine baseline, not BM25 or neural retrieval.

### Chunking and index design
Document chunks must retain source IDs and useful context. Small chunks may omit antecedents; large chunks increase noise and context cost. Overlapping chunks can inflate apparent retrieval coverage and split related evidence across results. Separate indexed reference documents from evaluation questions; never index gold answers as if they were naturally available evidence. Record corpus version, chunker, encoder, and index settings.

For one relevant document per query, hit@k asks whether it appears in the top k. With several relevant documents, recall@k divides retrieved relevant documents by the total relevant set. Mean reciprocal rank averages 1/rank of the first relevant hit. Define which convention is used rather than calling all these measures recall.

### Grounding and abstention
Generate with explicit evidence boundaries and stable source IDs. Evaluate whether each factual claim is supported by the cited source, whether the source is relevant, and whether the answer addresses the question. Abstain when evidence is absent or contradictory according to a calibrated policy. A retrieval similarity score is not a probability of answer correctness.

### Controlled experiments
Compare no-retrieval, retrieved context, oracle context, shuffled context, and missing-evidence conditions. If oracle context helps but retrieval does not, improve retrieval. If neither helps, inspect the answer model and task. Long-context models do not eliminate retrieval economics or distraction effects. Treat retrieved instructions as untrusted data; a document should not be allowed to change tool permissions or grading rules.

## Worked example

For relevant-document ranks [1,3,missing,2], MRR=(1+1/3+0+1/2)/4=0.4583 and hit@2=2/4. An answer accuracy of 75% with oracle evidence and 25% with retrieved evidence points toward retrieval/context selection as a likely bottleneck, requiring further ablation.

## Guided tutorial

**Question:** Index an authored campus document collection, compute retrieval metrics, and return cited evidence or abstain.

Open the notebook and predict each result before running it. Spend 5 minutes on setup and data inspection, 15 on the mechanism, 10 on the controlled intervention, and 10 on interpretation. The core uses NumPy and authored/synthetic data; the notebook identifies its limits. Save outputs, configuration, and a short interpretation. Do not report an expected trend as a measured result.

**Real-model or research extension:** Pass retrieved evidence into `extensions/hf_text.py` as a prompt and score generated claims manually. Compare with oracle and shuffled evidence; the core lab uses extraction, not generated answers.

See the [extension guide](../extensions/README.md) for commands, hardware assumptions, and validation status.

## Practical questions

1. Can a correct answer have an incorrect citation?
2. Why must document IDs survive chunking?
3. What does an oracle-context experiment isolate?

## After class

Complete the [exercise sheet](../exercises/07_rag.md) in 45–60 minutes. Read the first reference's abstract and method overview before the next session; remaining references are optional depth. For long papers, focus on the objective, one figure/table, and one limitation.

## Readings

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
