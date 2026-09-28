# Lecture 01 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. A tokenizer uses twice as many tokens for Hausa as English on your sample. What can and cannot be concluded?**

It indicates a cost disparity on this sample and tokenizer; it does not prove lower semantic accuracy or intrinsic language complexity. Match domains and inspect more data.

**2. Why is fitting the vocabulary before splitting a form of leakage?**

It exposes held-out lexical information to preprocessing; fit data-dependent transformations on training data.

**3. How should two sentences from the same source document be split?**

Keep the source document in one split to reduce dependence across splits.

## Exercise guidance

**Exercise 1.** NLL=3.4657, PPL=3.1748, bits/byte=5/12=0.4167; the byte denominator is given, not inferred from tokens.

**Exercise 2.** Use unicodedata.normalize("NFC", text); e + combining acute changes from two code points to one. Preserve originals and do not infer language accuracy.

**Exercise 3.** Grade source transparency, meaningful matching, correct counts, and avoidance of population-wide claims; no required winning tokenizer.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
