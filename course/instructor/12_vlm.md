# Lecture 12 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why is a blank-image baseline useful?**

It estimates how much the task can be solved from language priors or dataset shortcuts.

**2. Can a correct numerical answer be visually ungrounded?**

Yes; it can be guessed, memorized, or inferred from the question instead of the image.

**3. Why might a specialized chart parser outperform a small VLM on this lab?**

It directly encodes the known rendering convention; this advantage may disappear under style or layout changes.

## Exercise guidance

**Exercise 1.** 576 and 1152; cropping, padding, special tokens, pooling, and dynamic tiling can change actual token counts.

**Exercise 2.** Grounded highest-bar answers should change under swaps and abstain on blank images; the parser baseline depends on fixed layout.

**Exercise 3.** Include questions whose answers change under image interventions; do not treat all refusals as successes.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
