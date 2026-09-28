# Lecture 15 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. What experiment distinguishes modality use from a memorized answer?**

Counterfactual swaps with fixed question wording and controlled removal/mismatch of each modality.

**2. How should the paper report results if the official SemEval evaluation has not opened?**

Present measured development results with metric, split, and track labels. State that official evaluation is pending; preserve the submission-ready workflow and later deadlines. Do not invent test scores, receipts, or ranks.

**3. What makes a negative result useful?**

A sound baseline, controlled comparison, adequate documentation, and a clear implication for a hypothesis.

## Exercise guidance

**Exercise 1.** Average d_i=correct_full_i-correct_ablated_i; resample item or cluster IDs jointly and take interval quantiles under stated assumptions.

**Exercise 2.** Preserve paired IDs, report coverage and selective accuracy, and distinguish evidence presence from evidence validity.

**Exercise 3.** Award one point each for a traceable measured development result, a concrete error linked to the argument, an alternative explanation/limitation, and correct metric/split/track and evaluation-status labels. The full SemEval paper is assessed separately; a synthetic chart example is not a shared-task result.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
