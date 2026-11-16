"""Figures for lecture 8. The charts and the three toy systems repeat tutorial 8 (seed 42)."""
import math

from figstyle import *

rng = np.random.default_rng(42)


def chart(values):
    im = np.full((64, 96, 3), 255, dtype=np.uint8)
    for i, h in enumerate(values):
        color = np.zeros(3, dtype=np.uint8); color[i] = 200
        im[56 - int(h) * 4:56, 8 + 28 * i:24 + 28 * i] = color
    return im


def read_heights(im):
    out = []
    for i in range(3):
        column = im[:56, 12 + 28 * i]
        out.append(int(((column.min(axis=1) < 100) & (column.max(axis=1) > 100)).sum()) / 4)
    return np.array(out)


def answer(im):
    h = read_heights(im)
    if h.max() == 0: return None
    if (h == h.max()).sum() > 1: return "tie"
    return "ABC"[int(h.argmax())]


original = chart([3, 7, 5]); blank = np.full_like(original, 255); swapped = chart([7, 3, 5])

# 1. The three images of the intervention, drawn from the same pixel arrays as the lab.
fig, axes = new_fig(wide=True, ncols=4, height=3.4)
for ax, (name, im) in zip(axes, [("original", original), ("blank", blank), ("counterfactual", swapped), ("shifted 20 px", np.roll(original, 20, axis=1))]):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}\nparser: {answer(im)}", fontsize=15)
    for s in ax.spines.values(): s.set_visible(True); s.set_color(GREY)
save(fig, "08_interventions")

# 2. Three toy systems with the real image and with a decoy (same random draws as the notebook).
n = 300
truth = rng.choice(3, size=n, p=[.25, .5, .25])
def heights_for(top):
    h = rng.permutation(np.arange(1, 9))[:3]; h[[top, int(h.argmax())]] = h[[int(h.argmax()), top]]; return h
charts = [chart(heights_for(t)) for t in truth]
decoy_of = rng.permutation(n)
def weak(im):
    if answer(im) is None: return "ABC"[rng.choice(3, p=[.25, .5, .25])]
    return answer(im) if rng.random() < .15 else "ABC"[rng.choice(3, p=[.25, .5, .25])]
gold = np.array(list("ABC"))[truth]
acc = {}
for name, f in {"prior": lambda im: "B", "weak": weak, "parser": answer}.items():
    real = np.array([f(charts[i]) for i in range(n)]); decoy = np.array([f(charts[decoy_of[i]]) for i in range(n)])
    [f(blank) for i in range(n)]            # the notebook also queries the blank image here; keep the random stream aligned
    acc[name] = (np.mean(real == gold), np.mean(decoy == gold))
    print(name, acc[name])
fig, ax = new_fig()
x = np.arange(3)
ax.bar(x - 0.2, [acc[k][0] for k in acc], 0.38, color=BLUE, label="real image")
ax.bar(x + 0.2, [acc[k][1] for k in acc], 0.38, color=PINK, label="decoy image")
for i, k in enumerate(acc):
    ax.text(i - 0.2, acc[k][0] + 0.02, f"{acc[k][0]:.2f}", ha="center", fontsize=13); ax.text(i + 0.2, acc[k][1] + 0.02, f"{acc[k][1]:.2f}", ha="center", fontsize=13)
ax.set_xticks(x, ["prior\n(never looks)", "weak\n(looks 15%)", "parser\n(always looks)"]); ax.set_ylabel("accuracy on 300 toy questions")
ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.5, 1]); ax.legend(loc="upper left", ncols=2)
save(fig, "08_decoy")

# 3. The null distribution behind the worked McNemar example: 16 discordant pairs, 12 in favour of the real image.
k = np.arange(17); pmf = np.array([math.comb(16, int(i)) for i in k]) / 2 ** 16
print("p-value", pmf[12:].sum())
fig, ax = new_fig()
ax.bar(k, pmf, color=[PINK if i >= 12 else GREY for i in k])
ax.set_xlabel("pairs won by the real image, out of 16"); ax.set_ylabel("probability if the image did not matter")
ax.text(13.6, 0.06, f"p = {pmf[12:].sum():.3f}", color=PINK, fontsize=15, ha="center")
ax.set_title("12 of 16 or more: 2,517 of 65,536")
save(fig, "08_mcnemar")

# 4. Visual tokens grow with the square of the image side (patch 14, no pooling).
sides = np.arange(112, 897, 14); tokens = (sides // 14) ** 2
fig, ax = new_fig()
ax.plot(sides, tokens, color=BLUE)
for s in [224, 336, 448, 896]:
    ax.plot(s, (s // 14) ** 2, "o", color=PINK); ax.text(s - 20, (s // 14) ** 2 + 130, f"{(s // 14) ** 2:,}", ha="right", fontsize=14)
ax.set_xlabel("image side in pixels (patch 14)"); ax.set_ylabel("patches"); ax.set_xticks([224, 336, 448, 896])
ax.set_title("Twice the side, four times the patches")
save(fig, "08_tokens")
