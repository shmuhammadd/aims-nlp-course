# Lecture 02 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why does masking after softmax without renormalization change the scale?**

Dropping probabilities loses mass; use negative infinity before softmax or renormalize correctly.

**2. Can a decoder train all positions in parallel without seeing the future?**

Yes: teacher forcing plus the causal mask gives each position only its prefix.

**3. What property would an all-zero future mask test fail to detect?**

A zero-valued additive mask does not block attention; a future-perturbation test detects this.

## Exercise guidance

**Exercise 1.** First masked row is [1,0]; second weights are approximately [0.3302,0.6698], giving [0.3302,1.3395]. Unmasked first output is [0.6698,0.6605].

**Exercise 2.** Each head receives its own projection; concatenation is over feature axis. Perturb each suffix and compare all earlier outputs.

**Exercise 3.** Unmasked position-free output permutes with inputs; causal order introduces an asymmetric structure even without position embeddings.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
