"""Figures for lecture 10. Same simulations as tutorial 10 (seed 42, same draw order), so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)

# Tutorial section 2: an honest and an overconfident model with the same decisions.
z = rng.normal(0, 1.5, 5000)
p_true = 1 / (1 + np.exp(-z)); y = (rng.random(5000) < p_true).astype(int)
p_over = 1 / (1 + np.exp(-2.5 * z))
rng.integers(0, 10, size=(1000, 10))            # the tutorial's bootstrap draw, kept so later draws match

# 1. Reliability: accuracy against confidence in ten bins.
fig, ax = new_fig()
centres = np.linspace(0.525, 0.975, 10)
for q, colour, name in [(p_true, BLUE, "honest"), (p_over, PINK, "overconfident")]:
    conf = np.maximum(q, 1 - q); ok = (q >= 0.5) == y
    idx = np.minimum(((conf - 0.5) * 20).astype(int), 9)
    filled = [b for b in range(10) if (idx == b).sum() >= 20]
    ece = sum((idx == b).mean() * abs(ok[idx == b].mean() - conf[idx == b].mean()) for b in range(10) if (idx == b).any())
    ax.plot(centres[filled], [ok[idx == b].mean() for b in filled], "o-", color=colour, label=f"{name}, ECE {ece:.3f}")
ax.plot([0.5, 1], [0.5, 1], color=GREY, linestyle=":", linewidth=1.5)
ax.set_xlabel("confidence"); ax.set_ylabel("accuracy in the bin"); ax.legend(loc="upper left")
ax.set_title("Same accuracy (0.732), different calibration")
save(fig, "10_reliability")

# 2. Selective prediction for the overconfident model: accuracy on the answered share.
conf = np.maximum(p_over, 1 - p_over); ok = (p_over >= 0.5) == y
order = np.argsort(-conf); coverage = np.arange(1, 5001) / 5000; sel_acc = np.cumsum(ok[order]) / np.arange(1, 5001)
fig, ax = new_fig()
ax.plot(coverage[50:], sel_acc[50:], color=BLUE)
for t in [0.9, 0.99]:
    c = (conf >= t).mean(); a = ok[conf >= t].mean()
    ax.plot(c, a, "o", color=PINK, markersize=10); ax.text(c + 0.03, a + 0.005, f"threshold {t}", color=PINK, fontsize=14)
ax.set_xlabel("coverage (share answered)"); ax.set_ylabel("accuracy on answered")
ax.set_title("Answer less, score more")
save(fig, "10_coverage")

# 3. Tutorial section 4: the best of 50 equal systems, on the leaderboard and on fresh items.
board = rng.binomial(200, 0.70, size=(2000, 50)) / 200
winner_board = board.max(1); winner_fresh = rng.binomial(200, 0.70, size=2000) / 200
fig, ax = new_fig(wide=True, height=4.4)
bins = np.arange(0.5875, 0.8626, 0.005)
ax.hist(winner_fresh, bins, color=BLUE, alpha=0.85, label=f"winner on 200 fresh items (mean {winner_fresh.mean():.3f})")
ax.hist(winner_board, bins, color=PINK, alpha=0.85, label=f"winner on the leaderboard (mean {winner_board.mean():.3f})")
ax.axvline(0.70, color="#000000", linestyle="--", linewidth=2); ax.text(0.702, ax.get_ylim()[1] * 0.92, "true accuracy 0.70", fontsize=14)
ax.set_xlabel("accuracy of the best of 50 equal systems"); ax.set_ylabel("simulated leaderboards"); ax.legend(loc="upper left")
save(fig, "10_winner")

# 4. Tutorial section 6: three frames, per-frame F1 with micro and macro averages.
counts = {"A (common)": (40, 10, 10), "B (rare)": (2, 2, 6), "C (rare)": (1, 0, 3)}
f1 = lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn)
per = [f1(*v) for v in counts.values()]; TP, FP, FN = np.array(list(counts.values())).sum(0)
fig, ax = new_fig()
ax.bar(list(counts), per, color=[BLUE, VIOLET, VIOLET])
for value, colour, name in [(f1(TP, FP, FN), PINK, "micro"), (np.mean(per), NAVY, "macro")]:
    ax.axhline(value, color=colour, linestyle="--"); ax.text(2.45, value + 0.02, f"{name} {value:.3f}", color=colour, ha="right", fontsize=15)
ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.set_title("Micro follows the common frame")
save(fig, "10_micro_macro")

# 5. Worked example from the lecture notes: greedy matching on a 2 x 3 cosine matrix.
S = np.array([[0.9, 0.2, 0.1], [0.3, 0.8, 0.4]])
fig, ax = new_fig(height=4.2)
ax.imshow(S, cmap="Blues", vmin=0, vmax=1.2)
for i in range(2):
    for j in range(3):
        best_row, best_col = S[i].argmax() == j, S[:, j].argmax() == i
        ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=20, fontweight="bold" if best_row or best_col else "normal",
                color=PINK if best_row else ("#000000" if best_col else GREY))
ax.set_xticks(range(3), [r"$\hat{x}_1$", r"$\hat{x}_2$", r"$\hat{x}_3$"]); ax.set_yticks(range(2), [r"$x_1$", r"$x_2$"])
ax.set_xlabel("candidate tokens"); ax.set_ylabel("reference tokens")
ax.set_title(f"R = {S.max(1).mean():.2f}, P = {S.max(0).mean():.2f}")
for s in ax.spines.values(): s.set_visible(False)
save(fig, "10_greedy")
