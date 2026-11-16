# Lecture 01 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz question 5 and exercise 1 before reusing them for marks.

## Teaching the session

- Minutes 0–10: there is no quiz. Students mark their own [diagnostic](../PREREQUISITES.md). Go over question 2 (cross-entropy, used in the worked example), question 4 (choosing on the test set) and question 6 (the majority baseline). Students with three or more unfamiliar items pair with a stronger colleague for week 1.
- Keep the capstone opening to 15 minutes. Its only purpose is that every pair leaves with a task written down. Collect the choices during the lab.
- State the Task 12 conflict of interest aloud once.
- Do the BPE example on the board before showing the table. Write the four verbs, let the class count pairs for two minutes, and ask which pair wins. Students who counted (n,a) = 4 by hand find the notebook easy.
- Students did tokenisation in the DeepMind path. Spend the time on the last third: units, splits, baselines.
- Do the three-token worked example on the board, then ask what happens when the token count doubles.
- Common confusion: a higher token count is read as "the model is worse at this language". It is a cost. Accuracy is a separate measurement.
- Common confusion: perplexities from different tokenisers are compared. Use the 6.31 against 3,994 result to stop it.

## Practical questions

**1.** On this sample and with this tokeniser, Hausa text costs twice as many tokens: twice the price, more latency, half the text in a fixed context window. It does not show lower accuracy on Hausa or that Hausa is a harder language. A fair comparison uses parallel text in the same domain and more than one sample.

**2.** The vocabulary is fitted to data. Built on all rows, it has seen the words of the held-out rows, so held-out text has fewer unknown tokens than new text would. Split first, then build the vocabulary on training rows.

**3.** The same split, whichever it is. Sentences from one article share topic, names and style, so one in training and one in test lets the model score by recognising the article.

## Notebook

Solutions for the two stub cells are in [solutions/01_tokenization.py](solutions/01_tokenization.py). `python scripts/run_notebooks.py --only 01_tokenization` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42; the notebook has no random step):

- Section 1: ƙasa 4 code points, 5 bytes; Ẹ káàárọ̀ 9 code points, 17 bytes, 9 after NFC; ሰላም 3 code points, 9 bytes; the two-code-point "é" becomes one under NFC.
- Section 2: pair counts (n,a) 4, (l,a) 3, then four pairs at 2.
- Section 3: development perplexity uniform 17, unigram 14.485, bigram 6.314 at α = 0.5; NLL 33.170 nats over 18 predictions.
- Section 4: α = 0.01, 0.1, 0.5, 1.0 give 2.901, 3.764, 6.314, 8.261 on the first development document and 6.101, 5.441, 7.242, 8.903 on the one with two unseen transitions.
- Section 5: 6.314 fitted on training only, 4.751 with the development document leaked; per word 3,993.9.
- Section 6: merges (n,a), (l,a), (m,a). After 20 merges on the English training documents: English 6 tokens for 14 letters, Hausa 13 for 15, Swahili 16 for 17. Bits per byte 2.025 for the bigram at α = 0.1 and 4.328 for uniform.

## Exercise sheet

**Exercise 1.** a. Bits: $1+2+2+3+3=11$. Nats: $11\ln 2=7.625$. Perplexity: $2^{11/5}=4.59$. Bits per byte: $11/10=1.1$. b. Perplexity $2^{11/10}=2.14$; bits per byte 1.1. Neither model is better: the total NLL and the text are the same. One point for part a, one for part b with the conclusion.

**Exercise 2.** Full marks need the table at 0, 5, 10 and 20 merges for three held-out sentences, the round-trip assertion, and the authorship statement. Tokens per letter starts at 1.0 for every sentence and falls fastest on the language of the training sentences. Accept any numbers the student's output supports. With four training sentences the merges may run out before 20; the notebook's `learn_bpe` stops when every word is one symbol.

**Exercise 3.** Marks are for the three measurements and what the student does with them. A majority-label predictor on an eight-class task with majority share $p$ has F1 of $2p/(1+p)$ on that class and 0 on the others, so macro-F1 is $2p/(8(1+p))$; with $p=0.25$ that is 0.05. A student who reports the majority share as the macro-F1 has used accuracy. For Task 12 the unit is the painting; for Task 6 the image; for Task 9 the article. Any overlap count is acceptable if the method is shown. Teams whose data is not yet released use trial data and say so.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): three measurements 2, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output.

## Quiz 1 key

1. C. 2. B: (a,b) occurs 2 times in "abab" and once in each of the two copies of "abc", 4 in all; (b,c) occurs 2 times and (b,a) once. 3. C. 4. D.

5. a. $1+3+2+3=9$ bits. b. $2^{9/4}=2^{2.25}\approx4.76$. c. $9/8=1.125$ bits per byte.

6. Leaks, any two (1 each): the vocabulary saw the held-out rows; a random row split can put rows from one source, or duplicates, on both sides; the learning rate was chosen on the test set, so the test score is a selection score. Order (1): remove duplicates, split by source, build the vocabulary on training rows, choose the learning rate on development data, score the test set once.
