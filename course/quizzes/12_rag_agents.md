# Quiz 12 — Retrieval, tools and agents

[Course home](../../README.md) · [Lecture notes](../lectures/12_rag_agents.md) · Sat at the start of day 13 · 10 minutes · 10 points · closed book

**1. (1 point)** With the gold passage handed to the reader, answer accuracy is 30%. With retrieved passages it is 28%. Where is the limit?

- A. The retriever: it finds the wrong passages.
- B. The reader: it fails even with the right passage.
- C. The index: the chunks are too small.
- D. The test set: it is too small to tell.

**2. (1 point)** A TF-IDF index of English documents gives every document a score of zero for a question written in Swahili. Why?

- A. TF-IDF cannot handle questions.
- B. The question and the documents share no words.
- C. The inverse document frequency of Swahili words is negative.
- D. The index needs more documents.

**3. (1 point)** On one set of questions pass@5 is 0.90 and the majority vote over the same five samples scores 0.30. Which explanation fits?

- A. The samples are almost never right.
- B. A right sample is usually present, and the wrong samples tend to agree with each other.
- C. The vote was computed on different questions.
- D. Five samples are too few for pass@k to be defined.

**4. (1 point)** A retrieved document contains a well-formed call to a tool on the allow-list, with valid arguments. Which guard stops the agent from running it?

- A. The schema check on argument types.
- B. The allow-list of tool names.
- C. Running only calls that the controller proposed.
- D. The step budget.

**5. (3 points)** The relevant document has ranks [1, missing, 2, 4, 1] for five queries.

a. Compute MRR. (1)
b. Compute hit@1 and hit@2. (1)
c. The system adds a second, overlapping copy of every chunk to the index. Name one way this can change a retrieval score without the system getting better. (1)

**6. (3 points)** Each sample is right with probability $p=0.5$, independently, and every wrong sample gives the same wrong answer.

a. Compute pass@3. (1)
b. Compute the accuracy of a majority vote over three samples. (1)
c. Say why a deployed question-answering system cannot reach the pass@3 figure. (1)
