# Preparation and diagnostic

[Course home](../README.md) · Allow 60–90 minutes before day 1.

Install the [core environment](SETUP.md), open tutorial 1, and run its setup cell. For a Python/PyTorch refresher, the earlier [PyTorch practical](../practicals/pytorch_intro_notebook.ipynb) is optional; its historical dependencies are separate from the new core.

## Unmarked diagnostic (20 minutes)

1. If X is 16×32 and W is 32×8, what is the shape of XW? What axis should a per-example softmax normalize?
2. Given probabilities [0.2, 0.3, 0.5] and target class 2 (zero-indexed), calculate natural-log cross-entropy.
3. Differentiate (w x − y)² with respect to w.
4. Explain why selecting a learning rate using final test accuracy invalidates that test estimate.
5. Write a Python function that counts words in a list of strings without changing the originals.
6. A model predicts the majority class on a 90/10 split. What is its accuracy, and what behavior is hidden?
7. Explain broadcasting of a (4, 1) array against a (4, 6) array.
8. List one reason to set a seed and one reason a seed alone does not ensure reproducibility.

## Self-check

1. 16×8; normalize the 8 class scores within each row.
2. −ln(0.5) ≈ 0.6931.
3. 2x(wx−y).
4. Test outcomes influenced selection, so they no longer estimate a fully held-out choice.
5. For example, `[len(s.split()) for s in strings]`; whitespace words are a declared convention.
6. 90%; the minority-class recall is zero.
7. Each row's single value is repeated across its six columns for elementwise operations.
8. A seed controls some random generators; data versions, software, nondeterministic kernels, and hardware can still differ.

If three or more items are unfamiliar, review vector/matrix operations, NumPy indexing, logarithms, and train/development/test separation before the first session. During week 1, pair with a colleague and use the derivation examples before attempting optional extensions.

## Notation used throughout

B: batch size; T: sequence length; d: hidden dimension; V: vocabulary size; N: number of examples or model parameters as defined locally; r: adapter rank; τ: contrastive temperature; β: preference-objective coefficient. Vectors are row vectors in NumPy examples. Logarithms are natural unless bits are explicitly requested. MB uses 10⁶ bytes; MiB uses 2²⁰ bytes; GiB uses 2³⁰ bytes.
