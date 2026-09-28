"""Shared CLI and provenance helpers for optional model experiments."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform


def parser(description, model):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=model)
    p.add_argument("--revision", default=None, help="Use a resolved Hub commit for exact reruns")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="Optional result JSON file (fails if it already exists)")
    return p


def setup(args):
    import torch
    from transformers import set_seed
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu or the offline core.")
    set_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    return torch


def report(args, model, result):
    versions = {}
    for package in ["torch", "transformers", "peft", "numpy"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = getattr(model, "config", None)
    record = {"model": args.model, "requested_revision": args.revision,
              "resolved_revision": getattr(config, "_commit_hash", None),
              "seed": args.seed, "device": args.device, "python": platform.python_version(),
              "packages": versions, "result": result}
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return record
