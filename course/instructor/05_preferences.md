# Lecture 05 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Can DPO improve preference loss while worsening factuality?**

Yes; preferences may reward style, verbosity, or wrong beliefs. Measure factuality separately.

**2. Why is a group of identical rewards uninformative for standardized group advantages?**

All centered rewards are zero, so there is no within-group distinction.

**3. What happens if the chosen/rejected labels are reversed?**

The update favors the previously rejected response; audit pair construction and label semantics.

## Exercise guidance

**Exercise 1.** Derivative is β(sigmoid(z)-1); at equality it is -β/2.

**Exercise 2.** Conflicting labels limit separability; report uncertainty and do not present a loss improvement alone as alignment.

**Exercise 3.** Include parsing ambiguity, units, equivalent expressions, and hard-coded answer exploitation; audit the reward mechanism separately from the model.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
