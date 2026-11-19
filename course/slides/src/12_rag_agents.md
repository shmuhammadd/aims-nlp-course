type: title
title: Retrieval, tools and agents
subtitle: Day 12 · Week 3
footer: Shamsuddeen Hassan Muhammad | AIMS 2026

---
type: big
title: The question needs a fact the model never learned. What can you do without training it?

Notes: Take answers for a minute. Expect "fine-tune" first; rule it out. The three answers we want are today's three parts: hand it the evidence, let it try several times and check, or give it a tool.

---
title: Where today sits
subtitle: The model stays fixed. The system around it changes.

| Lever | What changes | Cost | Day |
|---|---|---|---|
| LoRA, SFT, DPO | Weights | A training run | 4, 5 |
| Steering | Activations at one layer | Forward passes | 11 |
| **Retrieval** | **The context the model reads** | **An index and a search** | **12** |
| **Sampling and voting** | **How many answers it gives** | **k forward passes** | **12** |
| **Tools** | **What it may call** | **A dispatcher and a budget** | **12** |

- Task 6 asks about local culture from an image, in Kiswahili and Amharic among others. A small vision-language model may lack the fact.
- Voting over several samples is a cheap method for any of the five tasks

Notes: Task 6 facts are from the task card. Its rules on external data were not stated on the page; a team that wants to retrieve from outside sources must ask the organisers first.

---
type: section
title: Part 1: Retrieval
subtitle: Hand the model the evidence, then check it used it

---
title: A retrieval system is a pipeline. Score each stage.
layout: stack

![](figures/12_pipeline.png)

- A fluent answer with a citation can still be unsupported. A relevant chunk can still be ignored.

Notes: This is retrieval-augmented generation, Lewis et al. 2020. Draw the five boxes on the board and leave them up: every later slide in part 1 points at one box. A reranker, when used, sits between the retriever and the top-k.

---
title: Sparse retrieval matches words. Dense retrieval matches vectors.

| | Sparse (TF-IDF, BM25) | Dense (two encoders) |
|---|---|---|
| Matches on | Shared words, weighted by rarity | Nearness of learned vectors |
| Good at | Names, codes, exact terms | Paraphrase; another language, with a multilingual encoder |
| Fails on | Paraphrase and other languages | Rare identifiers; text unlike its training data |
| Needs | No training | A trained encoder and a vector index |

- Hybrid: combine both scores. Reranker: score each query-document pair jointly, slower.
- Lab: "Maktaba inafunguliwa saa ngapi?" shares no word with "The library opens at eight on weekdays"

Notes: The Swahili question means "What time does the library open?". The lab's TF-IDF index gives it a score of zero for every document. Dense passage retrieval: Karpukhin et al. 2020. The lab does not implement BM25 or a neural retriever.

---
title: Chunks carry their source ID

- A small chunk loses context: "It closes at eighteen" with no noun
- A large chunk adds noise and fills the context window
- Overlapping chunks can return the same evidence twice and inflate the score
- Every chunk keeps the ID of its document, so that each claim can be traced
- Keep the reference documents apart from the evaluation questions. Never index gold answers.
- Record the corpus version, the chunker, the encoder and the index settings

Notes: Ask: why must IDs survive chunking? To trace claims, group chunks from one document, audit leakage and recover the original context.

---
title: Three retrieval metrics, one worked example

- **hit@k**: is the relevant document in the top k? One relevant document per query.
- **recall@k**: retrieved relevant documents over all relevant ones
- **MRR**: mean of 1/rank of the first relevant hit; 0 if it is not retrieved

Ranks of the relevant document for four queries: 1, 3, missing, 2

$$
\mathrm{MRR} = \frac{1}{4}\left(1 + \frac{1}{3} + 0 + \frac{1}{2}\right) = 0.458 \qquad \mathrm{hit@2} = \frac{2}{4}
$$

- Say which of the three you report. They are not all "recall".

Notes: Do it on the board. hit@1 here is 1/4. Students write mrr with missing ranks in the lab.

---
title: Paraphrase halves the score of a word-matching index
subtitle: Lab: TF-IDF over four invented campus rules

| Queries | Ranks of the gold document | MRR | hit@1 |
|---|---|---|---|
| 4 that reuse the documents' words | 1, 1, 1, 1 | 1.00 | 1.00 |
| 6 paraphrases, one in Swahili | 2, 2, 1, 1, missing, missing | 0.50 | 0.33 |

- "Where can I project slides?" misses the document about a projector: no stemming
- A test set written by the author of the documents flatters the index
- Write evaluation questions without looking at the passages, or have another person write them

Notes: Tutorial section 2. Four documents make a tiny index; the pattern is what transfers. Ask which fix they would try first: stemming, character n-grams, or a dense encoder.

---
title: An answer names its source, or the system abstains

- Return the answer with the ID of the chunk that supports it
- Check three things: does it answer the question, is the source relevant, does the source support each claim
- Abstain when the evidence is missing or contradictory. Choose the threshold on development questions.

### Lab
- "Does the library offer medical treatment?" retrieves the opening hours with score 0.52, and the reader cites them
- The citation is real. It supports nothing.
- A similarity score is not the probability that the answer is right

Notes: Day 10 link: similarity is not calibrated confidence. A correct answer can also carry a wrong citation, which is practical question 1.

---
title: Four evidence conditions locate the failure
layout: split

- **Retrieved**: the whole index
- **Oracle**: only the gold document
- **Shuffled**: only a wrong document
- **Missing**: the gold document removed
- Oracle helps and retrieved does not: fix retrieval
- Neither helps: the reader is the limit
- Any answer under shuffled or missing evidence is unsupported

![](figures/12_ablation.png)

Notes: Tutorial section 4, ten questions. Retrieval gives 6 correct and oracle 7, so better retrieval alone gains one question; on 3 the reader abstains even with the gold sentence. With the gold document missing the reader still cites a source twice. Add a no-retrieval row when the reader is a language model.

---
title: A long context window does not replace retrieval

- Putting every document in the prompt costs tokens on every question
- Distractors lower accuracy even when the right passage is present
- Position matters. Liu et al. (2023) moved the relevant passage through a long context:
  - accuracy was highest with the passage at the start or the end
  - and lowest with it in the middle
- Put the best evidence first, keep k small, and measure k as a setting

Notes: Lost in the Middle, arXiv 2307.03172. We give the direction of the finding and no numbers; the size depends on the model.

---
type: section
title: Part 2: Test-time compute and tools
subtitle: More than one forward pass per question

---
title: Four ways to spend more compute on one question

| Method | Extra cost | How it fails |
|---|---|---|
| Write out the reasoning | Longer output | The explanation can justify a wrong answer |
| Sample k answers and vote | k forward passes | Wrong answers that agree win the vote |
| Verify: calculator, tests, a checker | One check per candidate | The verifier has false positives and negatives |
| Call tools in a loop | Steps, latency, risk | Bad calls, loops, injected instructions |

- Compare methods under one stated budget of tokens, time or money
- Score the final answer and checkable intermediate results, never the length of the reasoning

Notes: A written chain of thought can help the model solve the problem. It is not a guaranteed account of the computation behind the answer. Self-consistency: Wang et al. 2022.

---
title: pass@k is not the accuracy of a vote

Each sample is right with probability p = 0.4, independently.

$$
\mathrm{pass@}k = 1 - (1-p)^k \qquad \mathrm{pass@4} = 1 - 0.6^4 = 0.870
$$

Three samples, and the wrong answers all agree:

$$
P(\mathrm{vote\ right}) = p^3 + 3p^2(1-p) = 0.064 + 0.288 = 0.352
$$

- pass@k needs an oracle that recognises the right sample. Code has tests. A short answer has none.
- The vote on three samples (0.352) is worse than one sample (0.400)

Notes: Board work. pass@3 = 1 − 0.216 = 0.784. pass@k is defined in Chen et al. 2021 for code, where unit tests play the oracle. Independence fails in practice: samples from one prompt and one model share their biases.

---
title: Voting helps only when the errors disagree
subtitle: Lab: each sample is right with probability 0.4; 4,000 questions per point
layout: stack

![](figures/12_vote.png)

- At 33 samples the vote reaches 0.92 when errors spread over four values and 0.12 when they agree

Notes: Tutorial section 8, seed 42. Ties count as failures, which is why the blue curve dips at k = 3. Before voting on your task, look at what the wrong samples say: if they repeat one answer, more samples will not help.

---
title: An agent is a loop with a budget
layout: split

- The controller proposes a call
- The dispatcher checks it against an allow-list and a schema
- The tool runs and returns an observation
- The controller reads it and proposes the next call, or answers
- Cap the steps. On an empty budget return a named failure, never a partial result.
- ReAct (Yao et al., 2022): a language model as controller, writing a thought before each action

![](figures/12_loop.png)

Notes: In the lab the controller is a script that computes (7 + 5) × 3 in two calls. With a budget of one step it returns status budget_exhausted and no answer; it does not return 12.

---
title: The dispatcher checks every call before it runs

```
def dispatch(call):
    if set(call) != {"tool", "args"}:
        raise ValueError("invalid call schema")
    if call["tool"] not in {"add", "multiply"}:
        raise ValueError("unknown tool")
    a, b = call["args"]["a"], call["args"]["b"]
    for x in (a, b):
        if type(x) not in (int, float) or not math.isfinite(x):
            raise ValueError("bad argument")
    return a + b if call["tool"] == "add" else a * b
```

- A schema holds typed arguments, ranges, required fields, and a rule against extra fields
- `type(x)`, because `isinstance(True, int)` is true in Python
- No generated string reaches `eval` or a shell

Notes: This is a shortened copy of the lab's dispatcher, which also checks the argument names and a range of one million. The lab version runs in the notebook; a model-driven controller has not been run in this course.

---
type: section
title: Part 3: Injection and tool boundaries
subtitle: What comes back from a tool is data

---
title: Retrieved text is data, not instructions

- A web page, a PDF, an image or a tool result can contain text written as an instruction
- Indirect prompt injection (Greshake et al., 2023): the attacker never talks to the model. They plant text where the system will retrieve it.
- The model reads one stream of tokens. Nothing in it marks which part the developer wrote.

### Rules for your system
- Mark retrieved text as quoted evidence in the prompt
- A document never changes tool permissions, the task, or the grading rule
- Test with a poisoned document in the index and count how often behaviour changes

Notes: arXiv 2302.12173. The paper demonstrates the attack on applications that combine a model with retrieval. Marking evidence in the prompt lowers the risk and does not remove it, which is why the next slide puts the guard outside the model.

---
title: A schema checks the form of a call, not who asked

- Lab: a poisoned document carries a well-formed call to `multiply`
- A naive agent runs every call it finds in its context. The dispatcher accepts it: allowed tool, finite arguments.
- The guard that worked is provenance: run only calls the controller proposed

### Defence in layers
- Allow-list and schema: stop malformed and unknown calls
- Provenance: stop calls that came from documents or tool output
- Least privilege: a read-only tool cannot delete anything
- A person confirms any action with side effects: send, pay, delete

Notes: Tutorial section 7. The injected call returned 10^12 in the naive agent. Ask what the cost would have been had the tool sent an email.

---
title: Test an agent on more than success

| Case | Expected behaviour |
|---|---|
| Malformed call: missing or extra field | Rejected, logged |
| Unknown tool, or a boolean posing as a number | Rejected, logged |
| Instruction inside a retrieved document | Ignored; the task continues |
| The same call repeated | Detected; the budget still counts it |
| Budget exhausted | Named failure, no partial answer |
| Question with no evidence | Abstain |

- Report the rate for each row beside task accuracy
- Log every call and observation: the trace is what you debug

---
title: Task 6: retrieval as an experiment

| Row | What it shows |
|---|---|
| No retrieval | What the model already knows |
| Retrieved passages, k chosen on dev | Your system |
| Oracle passage, on a hand-labelled subset | The ceiling if retrieval were perfect |
| Shuffled passage | Whether the model reads the evidence at all |

- Same questions, same decoding, BERTScore F1 from the organisers' scorer in every row
- Ask the organisers whether external data is allowed before you build the index
- Any task: sample several answers, normalise them, vote. Report the cost in forward passes.

Notes: The task page did not state rules on external data or LLM APIs when the course card was written. AfriQA is the nearest published benchmark for cross-lingual retrieval in African languages.

---
type: big
title: Oracle evidence lifts accuracy from 25% to 75%. Retrieved evidence leaves it at 25%. What do you fix?

Notes: Retrieval or context selection: the reader can use the right passage when it gets it. Next experiment: measure hit@k of the retriever directly, then run the shuffled condition to check that 25% is what the reader knows without evidence.

---
title: Lab: 40 minutes

- Open `course/tutorials/12_rag_agents.ipynb`. NumPy only.
- Sections 1 to 4, together: index, ranks, cited extraction, four evidence conditions
- Sections 5 to 8, together: dispatcher, budget, a poisoned document, voting
  - Predict each output before you run the cell
- Section 9, in pairs: write two functions until the tests pass
  - `mrr(ranks)` with missing ranks
  - `validate_call(call, schema)` for any tool list
- Task 6 teams: list the passages you could index and the rule you need confirmed

---
title: Readings

### Before the next session (20 minutes)
- Lewis et al., 2020. Retrieval-Augmented Generation. Sections 1 to 3.

### Retrieval
- Karpukhin et al., 2020. Dense Passage Retrieval.
- Liu et al., 2023. Lost in the Middle.

### Compute, tools and safety
- Wang et al., 2022. Self-Consistency.
- Yao et al., 2022. ReAct.
- Greshake et al., 2023. Indirect Prompt Injection.

Notes: Links and the full list are in the lecture notes.

---
title: Exit ticket

Write three lines and hand them in.

- One result you measured today, with its number
- One way the toy flatters that result
- One experiment for your own task: an evidence condition, or a vote

Quiz 12 opens the next session: six questions, ten minutes.
