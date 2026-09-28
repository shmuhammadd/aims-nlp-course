# Lecture 09 — instructor discussion and marking guide

Teaching support; distribute after the exercise deadline if using this repository privately. These files are public in a public clone, so use modified instances for high-stakes assessment.

## Practical questions

**1. Why does four-bit weight storage not make all inference memory four-bit?**

Caches and activations may retain higher precision, and metadata/workspaces remain.

**2. Why can a throughput improvement worsen user latency?**

Larger batches or queues improve device use but make individual requests wait.

**3. What additional evidence is needed after measuring low weight MSE?**

Downstream quality, stability, subgroup effects, real memory, and runtime on the target device.

## Exercise guidance

**Exercise 1.** 2×2×8192×32×4×128×2=1,073,741,824 bytes=1 GiB.

**Exercise 2.** Compute scales over the correct axis, report both weight and output MSE, and include scale storage in estimates.

**Exercise 3.** Do not compare unlike generation lengths; report TTFT, decode throughput, total latency, and hardware/software details.

## Marking anchors

Derivation (2): correct result (1), reasoning/conventions (1). Implementation (4): functioning method (2), meaningful verification (1), reproducibility (1). Investigation (4): controlled design (1), evidence/results or a fully specified protocol when requested (1), interpretation (1), limitations/provenance (1). Accept justified alternative methods. Do not award marks for invented measurements.
