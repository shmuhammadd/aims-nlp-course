"""Per-language safety outcome rates with bootstrap intervals and paired gaps.

Input CSV columns: language, prompt_id, label. One label per prompt per language.
Labels: refused, misunderstood, harmful, safe. Reuse prompt_id for translations of
the same prompt so the gap against the reference language is paired.
Store labels and IDs only, never prompt or response text.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

LABELS = ["refused", "misunderstood", "harmful", "safe"]


def load(path):
    data = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang, pid, label = row["language"], row["prompt_id"], row["label"]
            if label not in LABELS:
                raise ValueError(f"{lang}/{pid}: unknown label {label!r}; use {LABELS}")
            if pid in data[lang]:
                raise ValueError(f"{lang}/{pid}: duplicate prompt_id")
            data[lang][pid] = label
    return dict(data)


def interval(x, rng, boot=2000):
    idx = rng.integers(0, len(x), (boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return lo, hi


def rate(rows, label, rng):
    x = np.array([r == label for r in rows.values()], dtype=float)
    return (x.mean(), *interval(x, rng))


def gap(rows, ref, rng, label="harmful"):
    shared = sorted(rows.keys() & ref.keys())
    d = np.array([(rows[p] == label) - (ref[p] == label) for p in shared], dtype=float)
    return (d.mean(), *interval(d, rng), len(shared))


def report(data, reference, seed=0):
    if reference not in data:
        raise ValueError(f"reference language {reference!r} not in {sorted(data)}")
    rng = np.random.default_rng(seed)
    lines = [
        "| language | n | " + " | ".join(LABELS) + f" | harmful gap vs {reference} (paired) |",
        "|---|---|" + "---|" * (len(LABELS) + 1),
    ]
    for lang, rows in data.items():
        cells = [f"{m:.2f} [{lo:.2f}, {hi:.2f}]" for m, lo, hi in (rate(rows, lab, rng) for lab in LABELS)]
        if lang == reference or not rows.keys() & data[reference].keys():
            g = "-"
        else:
            m, lo, hi, n = gap(rows, data[reference], rng)
            g = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={n})"
        lines.append(f"| {lang} | {len(rows)} | " + " | ".join(cells) + f" | {g} |")
    return "\n".join(lines)


def self_test():
    rng = np.random.default_rng(1)
    n = 300
    probs = {"en": [0.70, 0.00, 0.10, 0.20], "yo": [0.10, 0.60, 0.20, 0.10]}
    data = {
        lang: {f"p{i}": lab for i, lab in enumerate(rng.choice(LABELS, size=n, p=p))}
        for lang, p in probs.items()
    }
    m, lo, hi = rate(data["en"], "refused", rng)
    assert lo <= 0.70 <= hi, "interval misses the true English refusal rate"
    m, lo, hi = rate(data["yo"], "misunderstood", rng)
    assert lo > 0.5, "non-comprehension should be visible as its own outcome"
    m, lo, hi, k = gap(data["yo"], data["en"], rng)
    assert k == n and lo > 0, "paired harmful gap should exclude zero"
    assert "| yo |" in report(data, "en")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--reference", default="en")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        print("self-test passed")
        return
    if not a.csv:
        ap.error("csv path required unless --self-test")
    print(report(load(a.csv), a.reference, a.seed))


if __name__ == "__main__":
    main()
