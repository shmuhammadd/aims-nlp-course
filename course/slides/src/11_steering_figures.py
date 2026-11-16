"""Figures for lecture 11. Same toy as tutorial 11 (seed 42), so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)
d, K = 32, 4
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
topic_dirs, read = Q[:K], Q[K]
u = Q[K] + 0.6 * Q[0]; u /= np.linalg.norm(u)


def activations(lang, n):
    topic = np.repeat(np.arange(K), n)
    return 3 * topic_dirs[topic] + (1.5 if lang else -1.5) * u + rng.normal(0, 0.6, (len(topic), d)), topic


H_en, _ = activations(0, 200); H_sw, _ = activations(1, 200)
D_en, t_en = activations(0, 100); D_sw, t_sw = activations(1, 100)
v = H_sw.mean(0) - H_en.mean(0); v_hat = v / np.linalg.norm(v)

# 1. Projections on the difference-of-means direction separate the two languages.
fig, ax = new_fig()
bins = np.linspace(-4, 4, 41)
ax.hist(D_en @ v_hat, bins, color=BLUE, alpha=0.85, label="English")
ax.hist(D_sw @ v_hat, bins, color=PINK, alpha=0.85, label="Swahili")
ax.set_xlabel(r"projection on $\hat{v}$"); ax.set_ylabel("held-out examples"); ax.legend()
ax.set_title("One direction separates the languages")
save(fig, "11_projection_hist")

# 2. Compliance rises with alpha, relevance falls: the trade-off Task 7 scores.
alphas = np.linspace(0, 4, 33)
comp = [((D_en + a * v) @ read > 0).mean() for a in alphas]
rel = [((((D_en + a * v) @ topic_dirs.T).argmax(1)) == t_en).mean() for a in alphas]
harm = [2 * c * r / (c + r + 1e-12) for c, r in zip(comp, rel)]
fig, ax = new_fig(wide=True, height=4.6)
ax.plot(alphas, comp, color=BLUE, label="compliance (answer is in Swahili)")
ax.plot(alphas, rel, color=PINK, label="relevance (answer is on topic)")
ax.plot(alphas, harm, color="#000000", linestyle="--", label="harmonic mean")
ax.axvspan(0.9, 1.6, color=VIOLET, alpha=0.15)
ax.text(1.25, 0.08, "usable\nrange", ha="center", color=NAVY, fontsize=14)
ax.set_xlabel(r"steering strength $\alpha$   ($h' = h + \alpha v$)"); ax.set_ylabel("share of held-out examples")
ax.set_ylim(0, 1.05); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
save(fig, "11_tradeoff")

# 3. Two-dimensional picture of add, clamp and ablate for one activation.
fig, ax = new_fig()
h = np.array([-1.2, 1.6]); vv = np.array([2.4, 0.9]); vh = vv / np.linalg.norm(vv)
ax.axline((0, 0), vh, color=GREY, linewidth=1, linestyle=":")
for end, colour, name in [(h, "#000000", "h"), (h + vv, BLUE, r"h + v  (add)"), (h - (h @ vh) * vh, PINK, "ablate")]:
    ax.annotate("", xy=end, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=2.5))
    ax.text(end[0] + 0.08, end[1] + 0.12, name, color=colour, fontsize=15)
ax.annotate("", xy=h + vv, xytext=h, arrowprops=dict(arrowstyle="-|>", color=VIOLET, linewidth=2, linestyle="--"))
ax.text(0.2, 2.35, "v", color=VIOLET, fontsize=15)
ax.text(2.2, 0.55, r"$\hat{v}$ axis", color=GREY, fontsize=14)
ax.set_xlim(-2, 3.2); ax.set_ylim(-0.6, 3.2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Add moves along v; ablate removes it")
save(fig, "11_geometry")
