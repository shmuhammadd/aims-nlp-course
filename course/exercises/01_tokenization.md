# Exercise sheet 1 — Language models, tokens and baselines

[Course home](../../README.md) · [Lecture notes](../lectures/01_tokenization.md) · [Notebook](../tutorials/01_tokenization.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

A model gives the five tokens of a held-out text the probabilities 0.5, 0.25, 0.25, 0.125 and 0.125. The text is 10 bytes long.

a. Compute the NLL in bits and in nats, the perplexity and the bits per byte.
b. A second tokeniser cuts the same text into 10 tokens, and the total NLL is unchanged. Compute the new perplexity and bits per byte. Which model is better?

## 2. Implement (4 points)

In notebook 1, use your `bpe_merge_step` and the notebook's `learn_bpe` and `tokenize`.

- Write six short sentences in a language you read well, other than English. Keep four for training and two held out. NFC-normalise them first.
- Learn 0, 5, 10 and 20 merges on your four training sentences. Report tokens per letter on your two held-out sentences and on the notebook's held-out English sentence, in one table.
- Add an assertion that joining the tokens of a word gives back the word, for every held-out word.

State who wrote the sentences and in which language.

## 3. Investigate (4 points)

Use your team's SemEval task data.

- Open the training file, or for Task 7 the development sentences for three of your languages. Report the number of rows per language, and per label where the task has labels.
- Compute one baseline that uses no model. For a classification task, predict the majority label and score it with the task's metric. For Task 6, answer every question with the most frequent training answer. For Task 7, report mean characters and UTF-8 bytes per sentence for each language.
- Name the unit that must stay in one split (painting, image, article). Count how many of those units appear in both the training and the development files.

Report the three results in one table and say what each means for your experiment plan. If the data is not yet released for your task, do this on its trial or sample data and say so.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
