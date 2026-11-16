# Lecture 13 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with 16.1 GB against 15 GB and list the class's options on the board. Return to the list at the end and put a cost beside each.
- Do the Llama KV-cache product on the board in powers of two: $2\cdot2^{12}\cdot2^{5}\cdot2^{3}\cdot2^{7}\cdot2=2^{29}$ bytes. Students who see the exponents add up remember the formula.
- Quantise $(0.52, -1.4, 0.71)$ by hand before showing the table. Then ask what happens if the scale is set by a weight of 30 elsewhere in the tensor.
- Say once that GB means $10^9$ bytes on the weight slides and GiB means $2^{30}$ bytes on the cache slides.
- Common confusion: a 4-bit model is assumed to use a quarter of all memory. Only the weights shrink.
- Common confusion: speculative decoding is taken for an approximation. The lab's histogram shows the output matches the target.
- The course has measured no speedup and no real-model memory. The 5.5 GB figure for a loaded 4-bit Llama-3.1-8B is a working estimate with overhead; ask the first team that loads it to report the measured peak.
- The 4-bit loading slide is a sketch that has not been run here.
- The fit table uses a 3 GB allowance on the T4 and 4 GB for the model on a laptop. Both are assumptions. Have each team redo the table for its own machine.

## Practical questions

**1.** The KV cache and activations usually stay at 16 bits, and scales, metadata, unquantised layers and the runtime's workspace are added to the weights.

**2.** A larger batch or a queue uses the device better and makes each request wait for the others.

**3.** Task quality on the outputs, quality for each language or subgroup, stability over seeds, and real memory and run time on the target device.

**4.** On a model that loads at both precisions, compute the vector at 4 and at 16 bits and report the cosine. Then apply the 4-bit vector to the 16-bit model and score compliance and relevance on the same development questions as the 16-bit vector.

## Notebook

Solutions for the stub cells are in [solutions/13_efficiency.py](solutions/13_efficiency.py). `python scripts/run_notebooks.py --only 13_efficiency` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42): Llama-3.1-8B weights 32.12, 16.06, 8.03 and 4.01 GB; it fits the T4 budget at 8 and 4 bits and the laptop budget at none. KV cache: 192 MiB for the 24-layer example; Llama-3.1-8B 0.125, 0.5, 2 and 16 GiB at 1,024, 4,096, 16,384 and 131,072 tokens. Weight error at 4 bits: 1.198 per tensor, 0.232 per channel; at 8 bits: 0.0205 and 0.00061. Output error: 78.5, 15.4, 1.33, 0.036. Share rounded to zero outside channel 0 at 4 bits per tensor: 1.0. Storage: 1,152 ideal, 2,304 in NumPy, 8,192 float32. Direction cosines at 8, 6, 4, 3, 2 bits: 1.0000, 0.9994, 0.9879, 0.9354, 0.5280. Speculative sampling: acceptance 0.752, overlap 0.75, 3.36 tokens per target pass. Your-turn A prints 4.0 GiB; B prints errors 1.11719 and 0.0037734.

## Exercise sheet

**Exercise 1.** a. $2\times2\times8192\times32\times4\times128\times2=1{,}073{,}741{,}824$ bytes $=1$ GiB. b. $q_{\max}=7$, $s=2.1/7=0.3$. Dividing gives $(3.33, -1.17, 7)$, integers $(3, -1, 7)$, reconstruction $(0.9, -0.3, 2.1)$. Errors are 0.10, 0.05 and 0; the largest, 0.10, is under $s/2=0.15$. One point per part.

**Exercise 2.** For the lab's matrix, 2,048 integers pack into 1,024 bytes; with 32 float32 scales (128 bytes) the total is 1,152, equal to the ideal figure. A reference packing: `u = (q + 8).astype(np.uint8).ravel(); packed = (u[0::2] << 4) | u[1::2]`. Full marks need the round-trip assertion, the three byte counts and a stated rule for an odd count, such as padding with one zero and trimming on unpack.

**Exercise 3.** Check the arithmetic against the model card and configuration the team cites. A good answer separates the estimate from any measured peak, and does not report a timing without device, batch, lengths and warm-up. Teams that cannot load the model lose no marks if they say so.

Marking anchors. Derivation (2): one point per part, with working. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): controlled design 1, measured results or a fully specified protocol 1, explanation 1, limitation 1. Give no marks for numbers that do not appear in an output or a shown calculation.

## Quiz 13 key

1. B ($4\times10^9\times4/8=2\times10^9$ bytes). 2. C. 3. C. 4. B.

5. a. $2\times4\times2048\times16\times4\times64\times2=2^{1+2+11+4+2+6+1}=2^{27}$ bytes $=128$ MiB. One point for the set-up with the leading 2 and the 2 bytes, one for the result. b. Unchanged, 128 MiB: the product $B\,T$ is the same.

6. a. $s=2.1/7=0.3$. Dividing gives $(2, -7, 3.33)$, so the integers are $(2, -7, 3)$. One point for the scale, one for the integers. b. Reconstruction is $(0.6, -2.1, 0.9)$; the largest error is 0.1, which respects the bound $s/2=0.15$.
