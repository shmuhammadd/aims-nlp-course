# Validation record

[Course home](../README.md)

Validation date: **28 September 2026**. Checks ran on macOS ARM64, Python 3.13.1, CPU. The [environment record](validation/environment.json) lists the installed package versions. Top-level course requirements are pinned; this is not a universal hardware or transitive-dependency lock.

## Checks completed

| Check | Result |
|---|---|
| Complete lesson bundles | 15 notes, 15 notebooks, 15 exercise sheets, 15 instructor guides |
| Core Python execution | All 15 notebooks / 60 code cells passed in fresh processes |
| Real Jupyter execution | All 15 notebooks / 60 code cells passed with nbclient and ipykernel |
| Notebook schema | All notebooks validated with nbformat 5.10.4 |
| Local navigation | All relative links in the new course and root README resolve |
| Python syntax | All notebook cells, extension scripts, and check scripts parse |
| Mathematical/behavioral checks | Causal invariance, finite-difference gradient, frozen base weights, DPO equality, memory estimates, CTC/WER, modality interventions, and other embedded assertions passed |
| Tiny Transformer | 100 training steps, development checkpoint selection, save, and future-token invariance passed |
| Text/LoRA integration | Both CLIs ran with a locally generated tiny random Qwen3 checkpoint; adapter B weights changed from zero |
| Sentiment pipeline | Train/dev selection/test output passed on the authored synthetic TSV fixture |
| Visual/audio API imports | Required image/audio classes import in the pinned environment |
| SVG illustrations | All five original SVG files parse as XML |

Executed outputs are committed as file content in the notebooks, so students can inspect a reference run without a model download. Notebook 15 was rerun after its result records gained an explicit experiment-condition field.

## Measured extension examples

The [tiny-decoder run](validation/tiny_decoder_run.json) records actual losses on the authored toy corpus. Development loss reached approximately 1.252 at step 60 and rose to approximately 1.405 at step 99 while training loss continued falling. This illustrates why checkpoint selection uses development data. It is not a language-model benchmark result.

The [sentiment smoke-test result](validation/sentiment_smoke_run.json) uses only the synthetic English fixture. Its four-item test accuracy must not be cited as AfriSenti or real-language performance.

## Reproduce the checks

```bash
python scripts/check_course.py
python scripts/run_notebooks.py
python scripts/run_notebooks.py --kernel --write-outputs
python scripts/check_model_extensions.py
python course/extensions/train_tiny_decoder.py --steps 100 --output outputs/validation-tiny
python course/extensions/afrisenti_baseline.py --train course/data/sentiment_train.tsv --dev course/data/sentiment_dev.tsv --test course/data/sentiment_test.tsv --output outputs/validation-sentiment.json
```

The first two need only the core environment. Kernel checks require `requirements-notebooks.txt`; model checks require `requirements-models.txt`. Local Jupyter kernels need permission to open local ports. The validation environment was temporary and did not alter the repository's existing practicals.

## Limits

Remote pretrained-checkpoint validation is listed separately below. The offline Qwen3 fixture validates API plumbing, masking, saving, and actual optimization; it does not establish pretrained-model quality. CUDA performance, memory estimates, every Python version in the supported range, Colab UI behavior, real AfriSenti data, and real speech/video datasets were not benchmarked. Historical notebooks and PDF slides were preserved without being revalidated.

Instructor preflight should verify the actual classroom device, chosen model revisions, data access, and runtime before assigning an optional model extension. The offline core remains fully usable without these extensions.
