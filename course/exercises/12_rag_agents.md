# Exercise sheet 12 — Retrieval, tools and agents

[Course home](../../README.md) · [Lecture notes](../lectures/12_rag_agents.md) · [Notebook](../tutorials/12_rag_agents.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. The relevant document has ranks [2, 1, 4, missing, 3] for five queries. Compute MRR, hit@1 and hit@3.

b. Each sample is right with probability $p=0.2$, independently. Compute the probability that at least one of $k$ samples is right for $k=1$, 4 and 8. Give one reason the samples from a real model are not independent.

## 2. Implement (4 points)

a. In notebook 12, replace the word features of the index with character trigrams: pad each word as `#word#` and take every run of three characters. Report the ranks, MRR and hit@1 on the six paraphrase queries before and after. Say which query is still missed and why. (2)

b. Add a `mean` tool to the dispatcher. It takes one argument, `values`, a list of 1 to 100 finite numbers. Write tests that the dispatcher rejects an unknown key, an empty list, a list holding `NaN`, a list holding `True`, and a list of 101 numbers. (2)

## 3. Investigate (4 points)

Use your team's own task and its development data.

Take three sets of development predictions from your baseline: three seeds, three prompts or three checkpoints. For classification tasks vote on the label; for Task 9 vote on each frame; for Tasks 6 and 7 normalise the answers (lower case, strip punctuation) before voting.

a. Report the score of each run, the score of the majority vote, and the oracle score, where an item counts as right if any run is right.

b. On the items the vote gets wrong, count how often at least two runs give the same wrong answer.

c. Say whether more runs would help, using your count from part b and the lab's two voting curves.

If you have only one run, say so, set `p` in the lab's simulation to your baseline's development accuracy, and answer part c from the simulation.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output or a scorer log.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
