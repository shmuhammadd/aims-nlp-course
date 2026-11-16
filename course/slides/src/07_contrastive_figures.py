"""Figures for lecture 7. The similarity matrix repeats tutorial 7 (seed 42), so the heatmap matches the notebook."""
from figstyle import *

rng = np.random.default_rng(42)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


# The tutorial's world: six 8x8 images, label vectors, blue-right held out.
colors, names = np.eye(3), ["red", "green", "blue"]
images, text, labels = [], [], []
for c in range(3):
    for side in range(2):
        im = np.zeros((8, 8, 3)); im[2:6, side * 4:side * 4 + 4] = colors[c]
        images.append(im); text.append(np.r_[colors[c], np.eye(2)[side]]); labels.append(names[c] + (" left", " right")[side])
X = np.array(images).reshape(6, -1); X /= np.linalg.norm(X, axis=1, keepdims=True)
T = np.array(text); T /= np.linalg.norm(T, axis=1, keepdims=True)
train, tau = np.arange(5), 0.2
W = rng.normal(0, 0.05, (X.shape[1], T.shape[1]))
for step in range(201):
    Z = X[train] @ W; norms = np.linalg.norm(Z, axis=1, keepdims=True); U = Z / norms
    S = U @ T[train].T / tau; row, col = softmax(S), softmax(S.T)
    dU = (row - np.eye(5) + col.T - np.eye(5)) / 10 @ T[train] / tau
    W -= 0.1 * X[train].T @ ((dU - U * (dU * U).sum(axis=1, keepdims=True)) / norms)
Z = X @ W; S = (Z / np.linalg.norm(Z, axis=1, keepdims=True)) @ T.T / tau

# 1. Similarity matrix after training, with the held-out row and column marked.
fig, ax = new_fig(height=5.4)
ax.imshow(S, cmap="Blues", vmin=-4.5, vmax=4.5)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=12, color="white" if S[i, j] > 2 else "black")
ax.set_xticks(range(6), labels, rotation=45, ha="right", fontsize=12); ax.set_yticks(range(6), labels, fontsize=12)
ax.axhline(4.5, color=PINK, linewidth=2.5); ax.axvline(4.5, color=PINK, linewidth=2.5)
ax.set_xlabel("caption"); ax.set_ylabel("image"); ax.set_title("Blue right was never trained")
for s in ax.spines.values(): s.set_visible(False)
save(fig, "07_similarity")

# 2. One image against three captions (cosines 0.8, 0.6, 0.1): the temperature decides how sharp the softmax is.
cos = np.array([0.8, 0.6, 0.1]); taus = [1.0, 0.2, 0.07]
fig, ax = new_fig()
for k, (t, colour) in enumerate(zip(taus, [GREY, VIOLET, BLUE])):
    p = softmax(cos / t)
    print("tau", t, "probabilities", np.round(p, 3))
    ax.bar(np.arange(3) + (k - 1) * 0.27, p, 0.25, color=colour, label=rf"$\tau = {t}$")
ax.set_xticks(range(3), ["own caption\ncos 0.8", "near miss\ncos 0.6", "unrelated\ncos 0.1"])
ax.set_ylabel("softmax probability"); ax.set_ylim(0, 1.05); ax.legend(loc="upper right")
ax.set_title("Lower temperature, sharper choice")
save(fig, "07_temperature")

# 3. Patch grids for a 224-pixel image: patch size 32 gives 49 tokens, patch size 16 gives 196.
fig, axes = new_fig(ncols=2, height=3.6)
for ax, patch in zip(axes, [32, 16]):
    n = 224 // patch
    for k in range(n + 1):
        ax.plot([0, 224], [k * patch] * 2, color=BLUE, linewidth=1); ax.plot([k * patch] * 2, [0, 224], color=BLUE, linewidth=1)
    ax.add_patch(plt.Rectangle((0, 224 - patch), patch, patch, color=PINK))
    ax.set_xlim(-2, 226); ax.set_ylim(-2, 226); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(rf"patch {patch}: ${n}\times{n} = {n * n}$", fontsize=15)
save(fig, "07_patches")

# 4. Retrieval when the caption embeddings get noisier (stand-in for a weaker text tower). 200 pairs, 32 dimensions.
def recall_at_k(S, k):
    d = np.diag(S)
    return ((S > d[:, None]).sum(1) < k).mean(), ((S > d[None, :]).sum(0) < k).mean()

rng = np.random.default_rng(42)
N, d = 200, 32
img = rng.normal(size=(N, d)); img /= np.linalg.norm(img, axis=1, keepdims=True)
noise = rng.normal(size=(N, d))
sigmas = np.linspace(0, 0.6, 25); r1, r5 = [], []
for s in sigmas:
    txt = img + s * noise; txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    r1.append(recall_at_k(img @ txt.T, 1)[0]); r5.append(recall_at_k(img @ txt.T, 5)[0])
fig, ax = new_fig(wide=True, height=4.4)
ax.plot(sigmas, r1, color=BLUE, label="image-to-text recall@1")
ax.plot(sigmas, r5, color=PINK, label="image-to-text recall@5")
ax.axhline(5 / N, color=GREY, linestyle=":", linewidth=1.5); ax.text(0.6, 5 / N + 0.03, "random ranker, k = 5", color=GREY, fontsize=13, ha="right")
ax.set_xlabel(r"noise added to each caption embedding, $\sigma$ (toy: 200 pairs, 32 dimensions)"); ax.set_ylabel("recall")
ax.set_ylim(0, 1.05); ax.legend(loc="upper right")
save(fig, "07_recall_noise")
