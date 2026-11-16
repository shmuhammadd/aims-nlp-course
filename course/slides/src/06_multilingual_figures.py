"""Figures for lecture 06. Same invented tables and seed as tutorial 06, so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)
language = np.array(["English"] * 100 + ["Hausa"] * 10)
correct_a = np.array([1] * 90 + [0] * 10 + [1] * 3 + [0] * 7)
correct_b = np.array([1] * 86 + [0] * 14 + [1] * 7 + [0] * 3)

# 1. The same overall accuracy hides different per-language accuracy (invented outcome table).
fig, ax = new_fig()
groups = ["overall", "English\n(n = 100)", "Hausa\n(n = 10)", "mean over\nlanguages"]
for j, (c, colour, name) in enumerate([(correct_a, BLUE, "system A"), (correct_b, PINK, "system B")]):
    per = [c[language == l].mean() for l in ("English", "Hausa")]
    vals = [c.mean(), per[0], per[1], np.mean(per)]
    ax.bar(np.arange(4) + (j - 0.5) * 0.38, vals, 0.36, color=colour, label=name)
    for x, v in zip(np.arange(4) + (j - 0.5) * 0.38, vals):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=12)
ax.set_xticks(range(4)); ax.set_xticklabels(groups, fontsize=13); ax.set_ylim(0, 1.15); ax.set_ylabel("accuracy")
ax.legend(loc="upper center", ncols=2); ax.set_title("Invented table: equal overall scores")
save(fig, "06_micro_macro")

# 2. Paired bootstrap over items and over authors (tutorial section 2; same order of random draws).
idx = rng.integers(0, len(language), size=(2000, len(language)))
diff = (correct_b[idx] - correct_a[idx]).mean(axis=1)
hausa = np.where(language == "Hausa")[0]
idx_h = rng.integers(0, len(hausa), size=(2000, len(hausa)))            # drawn to keep the stream aligned with the notebook
author = np.arange(len(language)) // 5
gain = np.array([(correct_b - correct_a)[author == a].sum() for a in range(22)])
idx_c = rng.integers(0, 22, size=(2000, 22))
diff_c = gain[idx_c].sum(axis=1) / len(language)
fig, ax = new_fig(wide=True, height=4.4)
bins = (np.arange(-18, 20) - 0.5) / 110            # one bin per possible value, k/110
for d, colour, name in [(diff_c, PINK, "resampling 22 authors"), (diff, BLUE, "resampling 110 items")]:
    lo, hi = np.quantile(d, [.025, .975])
    ax.hist(d, bins, color=colour, alpha=0.75, label=f"{name}: 95% interval [{lo:.3f}, {hi:.3f}]")
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("accuracy of B minus accuracy of A, in 2,000 resamples"); ax.set_ylabel("resamples")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
save(fig, "06_bootstrap")

# 3. Temperature-smoothed language sampling for corpora of 10,000, 1,000 and 100 examples.
fig, ax = new_fig()
sizes = np.array([10000, 1000, 100]); alphas = np.linspace(0, 1, 101)
q = np.array([sizes ** a / (sizes ** a).sum() for a in alphas])
for j, (colour, name) in enumerate([(BLUE, "10,000 examples"), (VIOLET, "1,000"), (PINK, "100")]):
    ax.plot(alphas, q[:, j], color=colour, label=name)
ax.set_xlabel(r"$\alpha$   (0 = uniform, 1 = follows corpus size)"); ax.set_ylabel(r"sampling probability $q_l$")
ax.legend(); ax.set_title(r"$q_l = n_l^\alpha / \sum_j n_j^\alpha$")
save(fig, "06_sampling")

# 4. Kappa for two annotators who are each right 90% of the time, as the positive class gets rarer (formula).
fig, ax = new_fig()
prev = np.linspace(0.01, 0.5, 200); acc = 0.9
says_pos = prev * acc + (1 - prev) * (1 - acc)
p_o = acc ** 2 + (1 - acc) ** 2
p_e = says_pos ** 2 + (1 - says_pos) ** 2
ax.plot(prev, (p_o - p_e) / (1 - p_e), color=BLUE, label=r"Cohen's $\kappa$")
ax.axhline(p_o, color=GREY, linestyle=":", label=f"raw agreement {p_o:.2f}")
ax.set_xlabel("share of items in the positive class"); ax.set_ylabel("agreement"); ax.set_ylim(0, 1); ax.legend(loc="center right")
ax.set_title("Same annotators, rarer class")
save(fig, "06_kappa_prevalence")

# 5. Steps in building an annotated dataset (schematic, no data).
fig, ax = new_fig(wide=True, height=3.2)
steps = [("Collect", "sources, consent,\nlicence"), ("Guidelines", "label definitions\nwith examples"), ("Pilot", "2 annotators,\n100 items, kappa"),
         ("Annotate", "3 native speakers\nper item"), ("Adjudicate", "majority vote,\nthen an expert"), ("Release", "splits, licence,\ndata statement")]
for i, (name, detail) in enumerate(steps):
    x = i * 2.05
    ax.add_patch(plt.Rectangle((x, 0.6), 1.75, 1.5, facecolor=[BLUE, BLUE, VIOLET, VIOLET, PINK, PINK][i], edgecolor="none"))
    ax.text(x + 0.875, 1.72, name, ha="center", va="center", color="white", fontsize=14.5, fontweight="bold")
    ax.text(x + 0.875, 1.08, detail, ha="center", va="center", color="white", fontsize=10.5)
    if i < 5:
        ax.annotate("", xy=(x + 2.03, 1.35), xytext=(x + 1.77, 1.35), arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.8))
ax.annotate("", xy=(2.9, 2.15), xytext=(4.95, 2.15),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=1.8, linestyle="--", connectionstyle="arc3,rad=0.5"))
ax.text(3.95, 2.78, "low kappa: revise the guidelines", ha="center", color=NAVY, fontsize=12.5)
ax.set_xlim(-0.1, 12.1); ax.set_ylim(0.4, 3.05); ax.axis("off")
save(fig, "06_data_pipeline")
