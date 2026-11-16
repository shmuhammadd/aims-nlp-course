"""Figures for lecture 15. Same twelve authored cases as tutorial 15 (seed 42), so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)


def render(values):
    im = np.full((48, 72, 3), 255, dtype=np.uint8)
    for j, h in enumerate(values):
        im[40 - int(h) * 4:40, 4 + 24 * j:16 + 24 * j] = [30, 60, 90]
    return im


def read(im, j):
    return int((im[:40, 8 + 24 * j, 0] < 100).sum() / 4)


cases = []
for i in range(12):
    values = rng.integers(1, 10, size=3); chosen = int(rng.integers(0, 3))
    cases.append({"policy": f"Use plot {'ABC'[chosen]}.", "chosen": chosen, "image": render(values), "gold": int(values[chosen])})

# 1. Three of the twelve cases: the policy picks the bar, the chart gives its height.
fig, axes = new_fig(ncols=3, height=3.4)
for ax, case in zip(axes, cases[:3]):
    ax.imshow(case["image"], interpolation="nearest")
    ax.set_xticks([10, 34, 58]); ax.set_xticklabels(["A", "B", "C"]); ax.set_yticks([])
    ax.set_title(f'"{case["policy"]}"\nanswer {case["gold"]}', fontsize=13)
save(fig, "15_cases")

# 2. The numbers in the slide table: accuracy and coverage under the four conditions of the lab.
def run(case, text=True, image=True, img=None):
    return read(case["image"] if img is None else img, case["chosen"]) if text and image else None

conds = {"both inputs": [run(c) for c in cases], "policy only": [run(c, image=False) for c in cases],
         "chart only": [run(c, text=False) for c in cases],
         "decoy chart": [run(c, img=cases[(i + 1) % 12]["image"]) for i, c in enumerate(cases)]}
for name, ans in conds.items():
    print(name, "accuracy", np.mean([a == c["gold"] for a, c in zip(ans, cases)]), "coverage", np.mean([a is not None for a in ans]))
