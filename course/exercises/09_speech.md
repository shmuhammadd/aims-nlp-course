# Exercise sheet 9 — Speech and multilingual ASR

[Course home](../../README.md) · [Lecture notes](../lectures/09_speech.md) · [Notebook](../tutorials/09_speech.ipynb)

Budget: 45–60 minutes. This sheet is worth 10 points. Submit your working, your changed notebook, the numbers you measured and 150–250 words on what they show.

## 1. Derive (2 points)

a. A CTC model has two symbols, the blank `_` and `a`, and three frames. It gives `a` probability 0.6, 0.5 and 0.2 in frames 1, 2 and 3. List the eight paths with what each collapses to, and compute the probability of the output "a".
b. Align the reference "the clinic opens at nine today" with the hypothesis "clinic open at nine to day". Give $S$, $D$, $I$ and the WER.

## 2. Implement (4 points)

In notebook 9:

a. Write `wer_counts(ref, hyp)` that returns $(S, D, I)$ from the alignment. Assert that it returns (1, 1, 1) on the lecture example.
b. Write `corpus_wer(refs, hyps)`: total edits over total reference words. For references `["yes", "the lab opens today"]` and hypotheses `["yes yes yes", "the lab opens today"]`, report the corpus WER and the mean of the two utterance WERs, and say which you would put in a paper and why.
c. Write `normalise(text)` with rules you choose and list them. Report raw and normalised WER and CER for these pairs: ("Habari ya asubuhi", "habari ya asubuhi"), ("ninakupenda", "nakupenda"), ("ƙofa", "kofa"), ("Sannu.", "sannu"). Name one rule you decided against and the reason.

## 3. Investigate (4 points)

Choose one. State which numbers you measured.

**A. Task 6 teams.** Take 20 development items from your track. For each audio file, record the sample rate, the number of channels and the duration, and listen to five. Then, with consenting teammates, record the same five questions on a phone. Report the same three properties for your recordings and list every difference you can hear or measure between the synthetic and the human clips. If you have an ASR model, report WER and CER on both sets of five. Finish with a table of the systems you will compare on human audio and the number of recordings you will collect.

**B. Any team.** In the notebook's tone toy, build 12 voices evenly spaced from 0.8 to 1.3 with 20 clips each, and replace the nearest-mean recogniser with a nearest-neighbour one. Compare a random split of the 240 clips with a split that holds out four whole voices. Report both accuracies and explain the gap. Then design a 12-clip consented recording pilot for a language you speak: speakers, sessions, the split, the consent wording in one sentence and the normalisation rules.

## Submission checklist

- Seed, NumPy version and every changed setting are recorded.
- Each number in your write-up appears in a notebook output.
- You say which results are measured and which are your explanation.
- Collaboration and any AI assistance are disclosed. You can explain every line you submit.
