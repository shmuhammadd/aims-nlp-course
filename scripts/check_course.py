"""Check course completeness, local links, notebook structure, and Python syntax."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "course/manifest.json").read_text())
assert [x["number"] for x in manifest] == list(range(1, 16))
assert len({x["slug"] for x in manifest}) == 15
checked_links = 0
for row in manifest:
    slug = row["slug"]
    for folder, suffix in [("lectures", ".md"), ("tutorials", ".ipynb"), ("exercises", ".md"), ("instructor", ".md")]:
        assert (ROOT / "course" / folder / (slug + suffix)).is_file(), (folder, slug)
    note = (ROOT / "course/lectures" / (slug + ".md")).read_text()
    assert "120 minutes" in note and "## Practical questions" in note and "## Worked example" in note
    exercise = (ROOT / "course/exercises" / (slug + ".md")).read_text()
    assert all(f"## {i}." in exercise for i in [1, 2, 3])

paths = [ROOT / "README.md"] + list((ROOT / "course").rglob("*.md")) + list((ROOT / "course/tutorials").glob("*.ipynb"))
for path in paths:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        assert nb["nbformat"] == 4
        assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
        texts = []
        for cell in nb["cells"]:
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            if cell["cell_type"] == "code":
                ast.parse(source, filename=str(path))
                assert isinstance(cell["outputs"], list)
            elif cell["cell_type"] == "markdown":
                texts.append(source)
        body = "\n".join(texts)
        try:
            import nbformat
        except ImportError:
            pass
        else:
            nbformat.validate(nb)
    else:
        body = path.read_text()
    for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", body):
        if target.startswith(("https://", "http://", "mailto:", "#", "data:")):
            continue
        target = unquote(target.split("#")[0])
        if not target:
            continue
        assert (path.parent / target).exists(), f"Broken link in {path.relative_to(ROOT)}: {target}"
        checked_links += 1
for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "course/extensions").glob("*.py")):
    ast.parse(path.read_text(), filename=str(path))
print(f"PASS: 15 complete lesson bundles; {checked_links} local links; notebook structure and Python syntax.")
