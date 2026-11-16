# Lecture 02 — instructor guide

Keep this file private if the cohort can see the repository. Change the numbers in quiz questions 5 and 6 and exercise 1 before reusing them for marks.

## Teaching the session

- Open with the question slide. The answer is "only the prediction at the last position". Return to it when you show the invariance test.
- Draw the residual stream as one horizontal line with the attention and MLP boxes hanging under it. Leave the drawing up: day 11 edits that line.
- Do the two-token example on the board before showing the table. Let the class compute $e^{1.414}\approx4.11$ and the weights 0.804 and 0.196.
- For the shape table, ask for each shape before revealing it. Use B = 2, T = 6, d = 8, H = 4, the numbers the lab prints.
- Do the GPT-2 count on the board. Students are surprised that the head count does not appear and that the MLP is two thirds of the block.
- The Llama configuration values (4,096; 32 layers; 32 heads; 8 key-value heads; 14,336; vocabulary 128,256) are from the model's `config.json`. The count reproduces 8.03 billion.
- Common confusion: the padding mask and the causal mask are treated as one thing. They hide different keys and have different shapes.
- Common confusion: students reshape (B, H, T, d/H) straight to (B, T, d) without swapping axes. The code runs and the result is wrong. Insist on a shape comment per line.
- Common confusion: a falling loss is taken as proof that the model is right. The all-zero-mask question near the end is there to break that.

## Practical questions

**1.** The rows no longer sum to one. Row $i$ keeps only the mass it had put on positions up to $i$, so early rows shrink most and the output scale depends on position. In the two-token example row 0 would sum to 0.804. Set the scores to $-\infty$ before the softmax, or renormalise each row.

**2.** Teacher forcing feeds the true tokens as input and the causal mask gives position $t$ only positions up to $t$. Each of the T predictions is made from its own prefix, so one pass yields T examples and none sees its target.

**3.** An additive mask of zeros blocks nothing. Every position reads the token it must predict, so the loss falls far below what the data allow, and the curve looks like success. The causal-invariance test fails at once: changing a future token moves earlier logits. Generation also gives it away, because at inference the next token is not there to copy.

## Notebook

Solutions for the two stub cells are in [solutions/02_transformers.py](solutions/02_transformers.py). `python scripts/run_notebooks.py --only 02_transformers` runs the notebook with those solutions in place of the stubs.

Reference outputs (seed 42):

- Section 1: scores [[2.828, 1.414], [1.414, 1.414]]; weights without mask [[0.804, 0.196], [0.5, 0.5]], output [[0.804, 0.391], [0.5, 1.0]]; with the causal mask [[1, 0], [0.5, 0.5]] and [[1, 0], [0.5, 1.0]].
- Section 2: logits (6, 12); attention row 1 is [0.53, 0.47, 0, 0, 0, 0].
- Section 3: causal prefix difference 0.0; unmasked 0.183.
- Section 4: loss 2.766 against a uniform reference of 2.485.
- Section 5: score −2.8705 for the three pairs two positions apart, −2.6334 for the pair one apart.
- Section 6: weight on padding keys 0.0; shapes (2, 6, 8) and (2, 4, 6, 6); GPT-2 small 124,439,808; Llama-3.1-8B block 218,112,000 and model 8,030,261,248.

## Exercise sheet

**Exercise 1.** a. The scaled scores are $I_2/\sqrt2$, so each row has 0.707 on its own key and 0 on the other. $e^{0.707}=2.028$, giving weights 0.670 and 0.330. Without a mask: weights [[0.670, 0.330], [0.330, 0.670]] and outputs [[0.670, 0.660], [0.330, 1.340]]. With the causal mask row 0 becomes [1, 0] with output [1, 0]; row 1 is unchanged. b. Attention $4\times512^2+4\times512=1{,}050{,}624$; MLP $2\times512\times2{,}048+2{,}048+512=2{,}099{,}712$; norms $4\times512=2{,}048$; total 3,152,384. One point for each part.

**Exercise 2.** Full marks need the two-block decoder returning (B, T, V), both invariance assertions, and the broken-mask run. Each block needs its own weights. The padding test must change a token at a padded position and compare logits only at real positions. With `np.tril(..., 1)` each position sees one token ahead, and the causal-invariance assertion fails at the first cut point. Two points for the decoder, one for the tests, one for the broken-mask report.

**Exercise 3.** Check the formulas against the configuration the student records. For Llama-3.1-8B the block is 218,112,000 and the model 8,030,261,248; at 16 bits that is 16.1 GB, which does not fit 15 GB, and at 4 bits 4.0 GB for the weights alone, which does. Common sources of a gap: tied or untied output embeddings, biases, position tables, and for encoder models the pooler and embedding norms. Accept any total within 1% of the published count, or a larger gap that the student explains. One point for the recorded configuration, one for the block count, one for the total and comparison, one for the memory calculation.

Marking anchors. Derivation (2): result 1, working 1. Implementation (4): working code 2, test 1, reproducible run 1. Investigation (4): configuration 1, counts 2, memory and interpretation 1. Give no marks for numbers that do not appear in an output or a shown calculation.

## Quiz 2 key

1. C. 2. B. 3. B. 4. C.

5. a. $e^{\ln 3}=3$ and $e^0=1$, so the weights are $(3/4, 1/4)$. b. $0.75\,(2,0)+0.25\,(0,4)=(1.5, 1)$. c. Query 0 may see only key 0: weights $(1, 0)$, output $(2, 0)$.

6. a. Attention $4\times100^2=40{,}000$; MLP $2\times100\times400=80{,}000$; norms $4\times100=400$; total 120,400. One point for the parts, one for the total. b. It does not change: heads slice the same matrices.
