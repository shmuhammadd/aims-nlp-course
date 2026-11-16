# Lecture 12 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

This session merges the former retrieval session and the former reasoning and tools session. It carries three parts in 50 minutes of slides, so keep to the worked numbers and move the rest to the lab.

## Teaching the session

- Draw the five-box pipeline on the board at the start and leave it up. Point at a box for every slide in part 1.
- Do MRR for ranks 1, 3, missing, 2 on the board. Ask what a missing rank contributes before writing the 0.
- Do pass@4 and the three-sample vote on the board side by side, with $p=0.4$. The class expects voting to help. Let the 0.352 surprise them, then show the two curves.
- Ask the class to predict the "missing" bar of the evidence figure before showing it. Most predict ten abstentions; the lab gives eight and two wrong citations.
- Common confusion: a citation is taken as proof of support. The medical-treatment query breaks that.
- Common confusion: a schema check is taken as a security boundary. Section 7 of the lab shows a valid injected call passing it.
- The dispatcher slide is a shortened copy of the lab code. No model-driven agent has been run in this course.
- Task 6 teams must confirm the rules on external data with the organisers before building an index. Say this twice.

## Practical questions

**1.** Yes. Correctness and support are separate: the answer can be right from the model's memory while the cited passage says nothing about it.

**2.** To trace each claim to its document, group chunks from one source, audit leakage between reference documents and test questions, and recover the surrounding context.

**3.** It hands the reader the right evidence, so any remaining errors belong to the reader. The gap between oracle and retrieved accuracy belongs to retrieval.

**4.** pass@k counts a question as solved if any sample is right. A vote needs the most frequent parsed answer to be right. With agreeing wrong answers the two move in opposite directions as k grows.

**5.** Typed arguments, allowed ranges, required fields, and rejection of unexpected fields.

**6.** No. Score correctness, grounding, cost and failures under one budget. A long trace can justify a wrong answer.

## Notebook

Solutions for the two stub cells are in [solutions/12_rag_agents.py](solutions/12_rag_agents.py). `python scripts/run_notebooks.py --only 12_rag_agents` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): top two for the first query are D2 at 0.62 and D1 at 0.11. Verbatim queries: ranks 1, 1, 1, 1, MRR 1.0. Paraphrases: ranks 2, 2, 1, 1, missing, missing; MRR 0.5, hit@1 0.33, hit@2 0.67. The medical-treatment query returns D1 at 0.52. Evidence conditions as (correct, abstained, wrong source): retrieved (6, 2, 2), oracle (7, 3, 0), shuffled (0, 10, 0), missing (0, 8, 2). Controller: 36 with budget 2, `budget_exhausted` with budget 1. The naive agent returns 10¹² from the injected call. Vote at k = 1, 3, 5, 9, 17, 33: errors spread 0.402, 0.354, 0.443, 0.604, 0.770, 0.915; errors agree 0.399, 0.358, 0.309, 0.253, 0.199, 0.115; pass@k 0.402, 0.792, 0.917, 0.992, 1.000, 1.000.

## Exercise sheet

**Exercise 1.** a. MRR $=(0.5+1+0.25+0+0.333)/5=0.417$; hit@1 $=1/5=0.2$; hit@3 $=3/5=0.6$. b. $1-0.8^k$: 0.2, 0.590, 0.832. Samples share one prompt and one model, so their errors are correlated. One point per part.

**Exercise 2.** a. With `#word#` padding and the same TF-IDF weighting, the reference run gives paraphrase ranks 4, 1, 1, 1, 1, missing: MRR 0.708 and hit@1 0.667, up from 0.5 and 0.333. "project" now matches "projector". The Swahili question still shares no trigram with the gold document, and the first paraphrase drops from rank 2 to rank 4; ask students to print which trigrams matched. Accept other numbers from a different padding choice if the code supports them. b. Check exact keys, list type, length bounds, `type(x) in (int, float)` and `math.isfinite` for every element. A solution that uses `isinstance` accepts `True` and loses a mark.

**Exercise 3.** Full marks need the three run scores, the vote and the oracle in one table, the count from part b, and a conclusion that follows from it: if wrong runs mostly agree, more runs will not help and the team should change the prompt or the model; if they disagree, more runs are worth their cost. The oracle score is an upper bound and must be labelled as one.

Marking anchors. Derivation (2): one point per part, with working. Implementation (4): working code 2, tests 1, reproducible run 1. Investigation (4): controlled design 1, measured results 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 12 key

1. B. 2. B. 3. B. 4. C.

5. a. MRR $=(1+0+0.5+0.25+1)/5=0.55$. b. hit@1 $=2/5=0.4$; hit@2 $=3/5=0.6$. c. Any one of: two copies of the relevant chunk can fill the top k, so recall@k counted over chunks rises while no new evidence is returned; duplicates push other relevant documents below k; a score counted per chunk now counts the same document twice.

6. a. pass@3 $=1-0.5^3=0.875$. b. The vote is right when at least two of three are right: $0.5^3+3(0.5)^2(0.5)=0.125+0.375=0.5$. c. pass@k assumes an oracle that recognises the right sample. A deployed system has the candidates and no gold answer, so it must select by vote or by a verifier, and both can pick wrong.
