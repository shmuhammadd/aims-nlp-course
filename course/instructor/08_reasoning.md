# Lecture 08 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why is pass@k not the accuracy of majority voting?**

Pass@k counts any success; majority voting needs the most frequent parsed answer to be correct.

**2. What belongs in a tool schema besides its name?**

Typed arguments, allowed ranges, required fields, and explicit rejection of unexpected fields.

**3. Does a longer reasoning trace establish that a system is more reliable?**

No; evaluate correctness, grounding, cost, and failure modes under a controlled budget.

## Exercise guidance

**Exercise 1.** 0.2, 0.5904, 0.8322; shared prompt/model biases correlate samples.

**Exercise 2.** Validate exact schema and bounds; reject booleans masquerading as numbers, nonfinite values, and oversized requests.

**Exercise 3.** Use identical questions, a frozen extraction rule, and a fixed budget; report failed parses instead of silently discarding them.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
