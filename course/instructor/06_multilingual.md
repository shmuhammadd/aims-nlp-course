# Lecture 06 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why might uniform language sampling hurt a very small language?**

It may repeatedly expose a small set and overfit; inspect unique coverage and development curves.

**2. Can translated evaluation data establish performance on naturally occurring code-switching?**

No; translated and natural code-switched distributions differ.

**3. Why is macro-F1 useful but insufficient for a multilingual system?**

Class balance and language balance are separate dimensions; report both plus support and uncertainty.

## Exercise guidance

**Exercise 1.** Overall=56/75=0.7467; equal-language=(0.9+0.4+0.6)/3=0.6333.

**Exercise 2.** Fix the class inventory; use the same resampled indices for both systems. Report support and a warning for very small groups.

**Exercise 3.** Assess documentation of variety, domain, label meanings, access, split units, and limits; no claim that one language represents a continent.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
