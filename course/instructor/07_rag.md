# Lecture 07 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Can a correct answer have an incorrect citation?**

Yes; correctness and evidential support are separate.

**2. Why must document IDs survive chunking?**

To trace claims, group related chunks, audit leakage, and recover original context.

**3. What does an oracle-context experiment isolate?**

It tests answer behavior when evidence selection is idealized, helping distinguish retrieval from generation failures.

## Exercise guidance

**Exercise 1.** MRR=(0.5+1+0.25+0+1/3)/5=0.4167; hit@1=0.2, hit@3=0.6.

**Exercise 2.** Fit index statistics only on the reference corpus; compare on the same queries and record all changed settings.

**Exercise 3.** Include false-premise and missing-detail questions; separate answerability labels from retrieval scores and report abstention coverage.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
