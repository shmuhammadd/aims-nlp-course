# Environment and execution

[Course home](../README.md)

## Core notebooks

Use Python 3.10–3.13. The core execution was checked with Python 3.13 and NumPy 2.1.3; see [validation](VALIDATION.md). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name aims-almm --display-name 'AIMS ALMM'
jupyter lab
```

On Windows activate with `.venv\Scripts\activate`. Select the AIMS ALMM kernel. Open `course/tutorials/01_tokenization.ipynb`, restart the kernel, and run all cells. Each notebook is self-contained; it does not depend on variables from a previous lecture. Setup prints the Python/NumPy version and fixed seed. Core notebooks use no network calls and read no external datasets.

For a minimal terminal-only installation, install `requirements.txt` and run:

```bash
python scripts/check_course.py
python scripts/run_notebooks.py
```

The runner executes plain Python cells in a fresh process per notebook and checks their assertions. It is useful without Jupyter; it does not test the browser UI. Add `--write-outputs` to save verified stdout into the notebooks. Add `--kernel` when nbclient/ipykernel are installed to execute through a real Jupyter kernel instead. To run one tutorial, use `--only 02_transformers`.

For Colab, upload a notebook through File → Upload notebook, run `import numpy; print(numpy.__version__)`, and use a CPU runtime for the core. Exact pinned reproduction is best done in a clean local environment. Existing repository Colab links point to the previous edition until these local changes are published. This course does not assume that the new files already exist on GitHub.

## Optional pretrained-model environment

Use a separate `.venv-models`, preferably Python 3.11 or 3.12. Install `requirements-models.txt`. On CUDA machines, install the matching PyTorch wheel for your driver using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then check the resulting versions. The pinned environment is a teaching baseline, not the latest release line; upgrading Transformers to 5.x is a separate compatibility exercise.

Allow several GB of disk for model caches. CPU float32 is the default in the scripts for portability; CUDA is selected only with `--device cuda`. Set small token and step limits before scaling. Model downloads occur on first execution. Scripts use standard model implementations with `trust_remote_code=False` and do not upload data or checkpoints. Record the resolved model revision printed in outputs. For exact reruns, pass that revision back with `--revision COMMIT_SHA`.

Use `python -m pip freeze > environment-lock.txt` in your project submission to capture transitive versions. The supplied requirement files pin top-level packages, not every transitive dependency or platform-specific wheel. Instructor preflight should record a platform-specific lock after validating the lab hardware.

## Troubleshooting

| Symptom | Action |
|---|---|
| NumPy missing | Install into the Python environment selected by the notebook kernel |
| Different notebook state after rerunning a cell | Restart and Run All; cells are intended to run in order |
| Model download unavailable | Continue with the offline core; retry downloads before a later optional session |
| Out of memory | Reduce batch/context/output length, choose a smaller checkpoint, or use the core path |
| CUDA unavailable | Use `--device cpu`; do not assume CUDA on Apple hardware |
| Pretrained result differs | Check checkpoint revision, prompt/template, software, seed, device, and decoding |
| Audio loading fails | Supply a short local WAV; check permissions and the optional soundfile/librosa installation |

Do not install packages mid-lecture unless necessary. An instructor should prepare an offline wheel cache and a tested environment image for the actual classroom platform where possible.
