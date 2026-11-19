# Validation record

[Course home](../README.md)

Last full check: **16 November 2026**, macOS ARM64, Python 3.13.1, NumPy 2.1.3, CPU. The pretrained-model checks further down date from 28 September 2026 and were not repeated.

## What was run on 16 November 2026

| Check | Result |
| --- | --- |
| Session bundles | 15 sessions, each with notes, slide source, built deck, notebook with two stub cells, solutions file, exercise sheet, instructor guide; quizzes for sessions 1–14 |
| Notebooks | All 16 (15 sessions and the optional video module) pass `scripts/run_notebooks.py` in fresh processes, with instructor solutions in place of the stubs |
| Slide builds | All 15 decks build from source with `scripts/build_slides.py`; the build checks that titles and subtitles fit one line and that text fits its box at 16 pt or larger |
| Slide geometry | No shape lies outside the slide's content area (python-pptx scan of all 408 slides) |
| Slide rendering | Every deck was previewed through macOS Quick Look and headless Chrome, and about a fifth of the slides were read by eye |
| Local links | All relative links in the course and the root README resolve (`scripts/check_course.py`) |
| arXiv links | All 129 distinct arXiv links were fetched and their titles matched the link text |
| Task facts | The five task cards were read from the organisers' pages on this date |

## What was not run or checked

- **PowerPoint.** No deck was opened in PowerPoint or Keynote. Quick Look is an approximation: open each deck once and look for anything that sits badly.
- **Pretrained models.** No new pretrained-model code was run. The forward-hook sketch on day 11 and the 4-bit loading sketch on day 13 are labelled as not run. Qwen3, LoRA, CLIP, Whisper and video extension runs remain unexecuted, as before.
- **Shared-task code and data.** The organisers' baselines for Tasks 6, 7, 9, 12 and AfriGuard were not run, and no task data was downloaded. Published baseline scores are quoted from the task pages.
- **Facts from memory.** Model configurations (for example Llama-3.1-8B: 4,096 wide, 32 layers, 8 key-value heads), some figures quoted from papers, and the section numbers in "read sections 1–3" were written from memory of the sources. The arXiv check confirms that each paper exists, not each number.
- **African-language text.** Short Hausa, Yoruba, Swahili and Amharic examples appear on slides and in notes. A speaker of each language should read them before they are shown.
- **Jupyter kernel and Colab.** This round used the plain-Python runner only.

## Measured extension examples

The [tiny-decoder run](validation/tiny_decoder_run.json) records actual losses on the authored toy corpus. Development loss reached approximately 1.252 at step 60 and rose to approximately 1.405 at step 99 while training loss continued falling. This illustrates why checkpoint selection uses development data. It is not a language-model benchmark result.

The [sentiment smoke-test result](validation/sentiment_smoke_run.json) uses only the synthetic English fixture. Its four-item test accuracy must not be cited as AfriSenti or real-language performance.

## Pretrained visual-QA check

The actual [SmolVLM run](validation/vlm_original_run.json) used revision `7e3e67edbbed1bf9888184d9df282b700a323964` on the [authored chart](validation/vlm_original_chart.png). It returned “A.” while the correct highest bar was B. This is a successful software-path check and a failed task prediction. It is retained as an error-analysis example rather than removed from the materials. No aggregate model-quality claim follows from one example.

The processor produced 17 image patches and 1,156 input-token positions in this run; preprocessing can enlarge a small input and should be included in resource planning.

## Reproduce the checks

```bash
python scripts/check_course.py
python scripts/run_notebooks.py
python -m pip install -r requirements-slides.txt && python scripts/build_slides.py
python scripts/check_model_extensions.py          # needs requirements-models.txt
python course/extensions/safety_eval.py --self-test
```

## Limits of the September 2026 model checks

The pretrained visual-QA path above was executed. Pretrained Qwen3 inference/LoRA, CLIP, Whisper, and video checkpoint runs were not executed; their syntax/API checks do not substitute for model-download and end-to-end classroom preflight. The offline Qwen3 fixture validates API plumbing, masking, saving, and actual optimization; it does not establish pretrained-model quality. CUDA performance, memory estimates, every Python version in the supported range, Colab UI behavior, real AfriSenti data, and real speech/video datasets were not benchmarked. Historical notebooks and PDF slides were preserved without being revalidated.

Instructor preflight should verify the actual classroom device, chosen model revisions, data access, and runtime before assigning an optional model extension. The offline core remains fully usable without these extensions.
