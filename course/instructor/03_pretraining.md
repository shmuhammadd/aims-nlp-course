# Lecture 03 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why can training perplexity improve while useful performance gets worse?**

Overfitting, distribution mismatch, and proxy-objective mismatch can each cause this.

**2. How would duplicated test documents distort a scaling comparison?**

It can reward memorization and exaggerate generalization; audit overlap before comparisons.

**3. Why is a randomly initialized bigram model not a small replica of a frontier LLM?**

It lacks long-range context, depth, and scale; it isolates objective and optimization only.

## Exercise guidance

**Exercise 1.** Use d log sum exp/dz=softmax(z); sum(p-onehot)=0.

**Exercise 2.** Copy weights on improvement; do not alias arrays or use the test set for selection.

**Exercise 3.** Proportions must sum to one; justify exposure without assuming corpus size equals value, and evaluate target plus general distributions.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
