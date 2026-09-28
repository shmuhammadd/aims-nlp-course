"""Execute course notebooks in isolated processes; optionally use real kernels."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_one(path, write_outputs=False, kernel=False):
    start = time.perf_counter()
    if kernel:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
        if write_outputs:
            nbformat.write(nb, path)
        count = sum(c.cell_type == "code" for c in nb.cells)
    else:
        nb = json.loads(path.read_text())
        namespace = {"__name__": "__main__"}
        count = 0
        for index, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
            except Exception:
                print(stdout.getvalue(), end="")
                print(stderr.getvalue(), end="", file=sys.stderr)
                raise
            cell["execution_count"] = count
            cell["outputs"] = [
                {"output_type": "stream", "name": name, "text": stream.getvalue()}
                for name, stream in [("stdout", stdout), ("stderr", stderr)] if stream.getvalue()
            ]
        if write_outputs:
            path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"PASS {path.name}: {count} code cells ({time.perf_counter()-start:.2f}s, {'Jupyter' if kernel else 'Python'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Notebook stem, such as 02_transformers")
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument("--kernel", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_one(args.worker, args.write_outputs, args.kernel)
        return
    paths = sorted((ROOT / "course/tutorials").glob("*.ipynb"))
    if args.only:
        paths = [p for p in paths if p.stem == args.only]
    if not paths:
        parser.error("No matching notebook")
    failed = []
    for path in paths:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)]
        if args.write_outputs:
            cmd.append("--write-outputs")
        if args.kernel:
            cmd.append("--kernel")
        try:
            result = subprocess.run(cmd, cwd=path.parent, timeout=240, check=False)
            if result.returncode:
                failed.append(path.name)
        except subprocess.TimeoutExpired:
            failed.append(path.name + " (timeout)")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))
    print(f"All {len(paths)} notebooks passed.")


if __name__ == "__main__":
    main()
