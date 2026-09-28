# Lecture 10 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Can two equally accurate models differ in calibration?**

Yes; their probability estimates can differ while argmax labels are identical.

**2. Why must an abstention threshold be chosen on development data?**

Selecting on test outcomes makes the reported result optimistically biased.

**3. What would reveal position bias in a model judge?**

Swap answer order on the same comparisons and check whether preferences change systematically.

## Exercise guidance

**Exercise 1.** Brier=(.04+.49+.36+.01)/4=.225; accuracy=.5; coverage=.5, selective accuracy=1.

**Exercise 2.** Weight by bin counts, do not fill empty bins with fabricated accuracy, and explain bin sensitivity.

**Exercise 3.** Record disagreement and revise ambiguous anchors using development examples; report pilot size and avoid strong reliability claims.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
