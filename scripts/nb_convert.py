"""Edit a course notebook as plain text.

    python scripts/nb_convert.py to-py  course/tutorials/11_steering.ipynb   # writes 11_steering.py beside it
    python scripts/nb_convert.py to-nb  course/tutorials/11_steering.py      # writes 11_steering.ipynb, deletes the .py

Cells are separated by '# %%' (code) or '# %% [markdown]' lines; markdown cells are comment blocks ('# ' prefix).
Outputs are dropped: rerun `python scripts/run_notebooks.py --only STEM --write-outputs` afterwards.
"""
import json
import sys
from pathlib import Path

mode, path = sys.argv[1], Path(sys.argv[2])
if mode == "to-py":
    out = []
    for cell in json.loads(path.read_text())["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "markdown":
            out.append("# %% [markdown]\n" + "\n".join(("# " + l).rstrip() for l in src.splitlines()))
        else:
            out.append("# %%\n" + src.rstrip("\n"))
    path.with_suffix(".py").write_text("\n\n".join(out) + "\n")
else:
    stem, cells = path.stem, []
    for n, block in enumerate(("\n" + path.read_text()).split("\n# %%")[1:]):
        head, _, body = block.partition("\n")
        md = "[markdown]" in head
        body = body.strip("\n")
        if md:
            body = "\n".join(l[2:] if l.startswith("# ") else l[1:] for l in body.splitlines())
        cell = {"cell_type": "markdown" if md else "code", "id": f"{stem}-{n:02d}", "metadata": {}, "source": body.splitlines(keepends=True)}
        if not md:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"aims": {"core": "offline-numpy", "lecture": stem, "seed": 42},
          "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.with_suffix(".ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    path.unlink()
