# Quiz 1 — Language models, tokens and baselines

[Course home](../../README.md) · [Lecture notes](../lectures/01_tokenization.md) · Sat at the start of day 2 · 10 minutes · 10 points · closed book

**1. (1 point)** Model A uses a subword tokeniser and reports perplexity 20 on a test set. Model B uses characters and reports perplexity 4 on the same test set. Which statement is right?

- A. B is the better model.
- B. A is the better model.
- C. The two numbers cannot be compared; convert both to bits per byte.
- D. The two numbers can be compared after dividing by the vocabulary size.

**2. (1 point)** A BPE training corpus holds the word "abab" once and the word "abc" twice. Each word starts as single letters. Which pair is merged first?

- A. (b, a)
- B. (a, b)
- C. (b, c)
- D. (c, a)

**3. (1 point)** A dataset has 100 captions for each of 1,658 paintings. You need a training and a development set. Which split is sound?

- A. Shuffle all captions and cut at 90%.
- B. Put 90 captions of every painting in training and 10 in development.
- C. Put 90% of the paintings, with all their captions, in training.
- D. Sort captions by length and cut at 90%.

**4. (1 point)** The Amharic greeting ሰላም has 3 letters. Each takes 3 bytes in UTF-8. How many positions does a byte-level model use for it?

- A. 1
- B. 3
- C. 6
- D. 9

**5. (3 points)** A model gives the four tokens of a text the probabilities 1/2, 1/8, 1/4 and 1/8. The text is 8 bytes long.

a. Compute the NLL in bits. (1)
b. Write the perplexity as a power of 2. (1)
c. Compute the bits per byte. (1)

**6. (3 points)** A team builds its vocabulary on the whole dataset, then splits it at random by row, then picks the learning rate with the best test accuracy. Name two leaks in this procedure (2). Write the steps in an order that removes them (1).
