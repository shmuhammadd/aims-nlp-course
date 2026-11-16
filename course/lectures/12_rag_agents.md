# Lecture 12 — Retrieval, tools and agents

[Course home](../../README.md) · Week 3, day 2 · 120 minutes · [Slides](../slides/12_rag_agents.pptx) · [Notebook](../tutorials/12_rag_agents.ipynb) · [Exercises](../exercises/12_rag_agents.md) · [Quiz](../quizzes/12_rag_agents.md)

## Learning outcomes

- Score retrieval and answers separately, and use oracle, shuffled and missing-evidence conditions to find which stage fails.
- Compare sampling, voting and verification under one budget, and say why pass@k is not the accuracy of a vote.
- Build a tool dispatcher that checks every call against a schema, runs inside a step budget, and treats retrieved text and tool output as data.

## Session plan

| Minutes | Activity |
|---|---|
| 0–10 | Quiz 11 (interpretability and steering) |
| 10–35 | Slides 1–12: the Task 6 problem, the retrieval pipeline, sparse and dense retrieval, metrics, grounding, the four evidence conditions |
| 35–60 | Slides 13–22: sampling and voting, verification, the bounded loop, prompt injection; pass@k and the vote on the board |
| 60–65 | Break |
| 65–105 | [Notebook 12](../tutorials/12_rag_agents.ipynb): sections 1–8 together, section 9 in pairs |
| 105–115 | Slides 23–24: the Task 6 retrieval experiment; discuss the oracle question |
| 115–120 | Exit ticket: one measured result, one way the toy flatters it, one experiment for your task |

## Teaching notes

### The problem
A Task 6 question shows an image and asks about local culture, in Kiswahili or Amharic among other languages. A small vision-language model may never have seen the fact it needs. Days 2 to 5 changed weights and day 11 changed activations. Today the model stays fixed and the system around it changes: it can be handed evidence, given more samples, or given tools.

### A retrieval system is a pipeline
A retrieval-augmented generation (RAG) system indexes evidence, retrieves candidates, may rerank them, builds a context and generates an answer (Lewis et al., 2020). The language model is one stage of five. A fluent answer with a citation can still be unsupported, and a relevant document can still be ignored. Score each stage.

### Sparse, dense and hybrid retrieval
Sparse retrieval matches words, weighted by inverse document frequency. BM25 also saturates term frequency and corrects for document length. Dense retrieval compares learned vectors from an encoder (Karpukhin et al., 2020). It can recover a paraphrase, and a multilingual encoder can match a Swahili question to an English passage. It can fail on exact identifiers such as a name or a code. A hybrid system combines the two scores. A reranker scores each query-document pair jointly and is slower. The lab implements TF-IDF with cosine similarity, a sparse baseline. It does not implement BM25 or a neural retriever.

In the lab, queries that reuse the documents' words reach MRR 1.0. Six paraphrases reach MRR 0.5. "Where can I project slides?" misses a document about a projector, and "Maktaba inafunguliwa saa ngapi?" shares no word with an English document about the library.

### Chunking and index design
Each chunk keeps its source ID and enough context to be read alone. A small chunk can lose the noun a pronoun refers to. A large chunk adds noise and uses more of the context window. Overlapping chunks can inflate retrieval scores and return the same evidence twice. Keep reference documents apart from evaluation questions, and never index gold answers. Record the corpus version, the chunker, the encoder and the index settings.

### Retrieval metrics
With one relevant document per query, hit@k asks whether it is in the top $k$. With several relevant documents, recall@k is the retrieved relevant documents over all relevant ones. Mean reciprocal rank (MRR) averages $1/\text{rank}$ of the first relevant hit, with 0 when nothing relevant is retrieved. Say which of the three you report.

### Grounding and abstention
Generate with marked evidence and stable source IDs. Check three things for each answer: does it address the question, is the cited source relevant, and does the source support each factual claim. Abstain when the evidence is absent or contradictory, under a threshold chosen on development questions. A retrieval similarity is not a probability that the answer is right. In the lab, "Does the library offer medical treatment?" retrieves the library's opening hours with a score of 0.52 and the reader cites it.

### Controlled experiments
Compare five conditions on the same questions: no retrieval, retrieved context, oracle context (the gold document handed over), shuffled context (a wrong document), and missing evidence (the gold document removed from the index). If oracle context helps and retrieved context does not, fix retrieval. If neither helps, the reader or the task is the limit. Under shuffled and missing evidence the right behaviour is to abstain, so every answer there is unsupported.

In the lab, retrieval gives 6 correct sources out of 10 and oracle context gives 7. The reader abstains on 3 questions even with the gold document. With the gold document missing, it still cites a source for 2.

A long context window does not remove the need to retrieve. Liu et al. (2023) moved the relevant passage through a long context and found accuracy highest when it sat at the start or the end and lowest in the middle.

### More inference can change the result
A model can spend extra computation by writing a longer answer, sampling several candidates, revising, searching or calling tools. Each has its own cost and its own failure. Compare them under one declared budget of tokens, time or money. Greedy accuracy, majority-vote accuracy and pass@k answer different questions. pass@k counts a question as solved if any of $k$ samples is right (Chen et al., 2021), which assumes an oracle that recognises the right one. A deployed system has no such oracle.

Self-consistency samples several reasoning paths and takes a majority vote over the final answers (Wang et al., 2022). The vote helps when wrong answers disagree with each other. In the lab each sample is right with probability 0.4. With wrong answers spread over four values, the vote rises to 0.92 at 33 samples. With wrong answers agreeing on one value, it falls to 0.12. Correlated mistakes and parsing errors are the usual causes of the second case. Best-of-n selects with a scoring function, which can itself be biased.

A written chain of thought can help a model solve a problem. It is not a guaranteed account of the computation that produced the answer. Score final answers and checkable intermediate objects, such as equations and tool outputs. A polished explanation can justify a wrong answer, so do not reward length.

### Verification
Use a calculator for arithmetic. Use tests for code, knowing they check some properties and miss others. A verifier has false positives and false negatives; measure it on labelled cases before trusting it.

### A bounded agent loop
The state holds the user's task, the evidence so far, the proposed action, the tool result and the remaining budget. A controller proposes an action. A dispatcher checks the tool name against an allow-list and the arguments against a schema. The tool runs and returns an observation. The controller decides whether to continue. ReAct (Yao et al., 2022) is this loop with a language model as controller, writing a thought before each action. Cap the number of steps and return a named failure when the budget runs out. Log every action and outcome.

A tool schema holds more than a name: typed arguments, allowed ranges, required fields, and rejection of unexpected fields. The lab uses a scripted controller so the boundaries are easy to see; it shows nothing about a model's ability to plan. Never execute Python from a generated string. A calculator with named operations is enough for the lesson.

### Prompt injection and tool boundaries
A retrieved page, an image or a tool response can contain text written as an instruction. Greshake et al. (2023) call this indirect prompt injection: the attacker never talks to the model and plants the text where the system will retrieve it. Treat everything that comes back from a retriever or a tool as data. Keep system policy and external content apart, and never let a document change tool permissions or grading rules.

Structured output makes a call easy to parse. It does not make the call true or permitted. In the lab a poisoned document carries a well-formed call to an allowed tool, and the dispatcher accepts it. The guard that works is provenance: the agent runs only calls the controller proposed. Give tools the least privilege that does the job, and ask a person before any action with side effects. Test malicious and malformed proposals, duplicate calls, an exhausted budget and inappropriate actions beside ordinary success.

### Capstone links
Retrieval of cultural facts is a candidate improvement for Task 6. The task page did not state its rules on external data, so ask the organisers before building an index from outside sources. Self-consistency is a cheap test-time method for any task: sample several answers, normalise them and vote. It costs $k$ forward passes and no training.

## Worked example

**Retrieval.** For relevant-document ranks [1, 3, missing, 2], MRR $=(1+1/3+0+1/2)/4=0.458$ and hit@2 $=2/4$.

**Locating the failure.** Answer accuracy of 75% with oracle evidence and 25% with retrieved evidence points at retrieval or context selection. Run the shuffled condition next: if accuracy stays at 25% with a wrong document, the reader was answering from memory all along.

**pass@k.** With independent samples each right with probability $p=0.4$, the chance that at least one of $k=4$ is right is $1-0.6^4=0.870$.

**Majority vote.** Take the same $p=0.4$, three samples, and wrong answers that all agree. The vote is right when at least two samples are right: $p^3+3p^2(1-p)=0.064+0.288=0.352$. That is below one sample's 0.4, and pass@3 is $1-0.6^3=0.784$.

## Guided tutorial

**Question:** When a retrieval system gives a wrong answer, which stage failed? And what must stand between a model's proposed tool call and its execution?

The notebook indexes four invented campus rules with TF-IDF, scores verbatim and paraphrased queries, extracts a cited sentence or abstains, and answers ten questions under four evidence conditions. It then runs a schema-checked calculator dispatcher, a two-step controller with a budget, a poisoned document, and a simulation of voting against pass@k. Students write `mrr` and `validate_call` against tests. The reader copies a sentence and the controller is a script, so the lab says nothing about a real model.

**Shared-task path:** Task 6 teams draft the retrieval experiment for their plan: the passages to index, the encoder, and the no-retrieval, retrieved and oracle rows on development questions. For a real-model extension, pass retrieved evidence into `extensions/hf_text.py` as a prompt and mark the generated claims by hand; feed any proposed call through the validator, never into `eval` or a shell. See the [extension guide](../extensions/README.md).

## Practical questions

1. Can a correct answer have an incorrect citation?
2. Why must document IDs survive chunking?
3. What does an oracle-context experiment isolate?
4. Why is pass@k not the accuracy of majority voting?
5. What belongs in a tool schema besides its name?
6. Does a longer reasoning trace show that a system is more reliable?

## After class

Complete the [exercise sheet](../exercises/12_rag_agents.md) in 45–60 minutes. Read sections 1–3 of the RAG paper before the next session (20 minutes). [Quiz 12](../quizzes/12_rag_agents.md) opens day 13.

## Readings

**Required**
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020). Sections 1–3: a retriever and a generator trained as one system.

**Retrieval**
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020). The two-encoder retriever and its comparison with BM25.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023). Where evidence sits in the context changes accuracy.
- [AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages](https://arxiv.org/abs/2305.06897) (Ogundepo et al., 2023). Questions in 10 African languages with answers in other-language text; the nearest benchmark to a Task 6 retrieval system.

**Test-time compute and tools**
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) (Wang et al., 2022). Sample, then vote; read the method and one results table.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022). The thought, action, observation loop.
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Chen et al., 2021). Section 2 defines pass@k.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (2025). A model trained to spend tokens on reasoning before it answers.

**Safety**
- [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) (Greshake et al., 2023). Sections 1–3: how planted text reaches a model through retrieval.
