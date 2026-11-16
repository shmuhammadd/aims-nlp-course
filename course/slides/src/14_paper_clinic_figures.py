"""Figures for lecture 14. Same synthetic predictions as tutorial 14 (seed 42), so slide numbers match the notebook."""
from figstyle import *

EMOTIONS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]
K = 8
prior = np.array([0.22, 0.18, 0.20, 0.08, 0.04, 0.06, 0.10, 0.12])
confusable = np.array([3, 2, 1, 0, 5, 4, 7, 6])


def simulate(rng, n):
    y = rng.choice(K, size=n, p=prior)
    ease = rng.uniform(size=n)

    def predict(cut):
        right = 0.5 * ease + 0.5 * rng.uniform(size=n) < cut
        slip = np.where(rng.uniform(size=n) < 0.6, confusable[y], rng.choice(K, size=n, p=prior))
        return np.where(right, y, slip)
    return y, predict(0.508), predict(0.513)


def confusion(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def macro_f1(y, p):
    C = confusion(y, p)
    tp, denom = np.diag(C), C.sum(0) + C.sum(1)
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def paired(rng, y, a, b, n_boot=1000):
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out[i] = macro_f1(y[idx], b[idx]) - macro_f1(y[idx], a[idx])
    return out


rng = np.random.default_rng(42)
n = 2000
y, pred_a, pred_b = simulate(rng, n)
observed = macro_f1(y, pred_b) - macro_f1(y, pred_a)
diffs = paired(rng, y, pred_a, pred_b)                  # same draws as section 3 of the notebook
unpaired = np.empty(1000)
for i in range(1000):
    ia, ib = rng.integers(0, n, n), rng.integers(0, n, n)
    unpaired[i] = macro_f1(y[ib], pred_b[ib]) - macro_f1(y[ia], pred_a[ia])
lo, hi = np.quantile(diffs, [0.025, 0.975])
print(f"observed {observed:+.3f} paired [{lo:+.3f}, {hi:+.3f}] unpaired {np.quantile(unpaired, [0.025, 0.975]).round(3)}")

# 1. The bootstrap distribution of the macro-F1 difference, paired against unpaired.
fig, ax = new_fig(wide=True, height=4.6)
bins = np.linspace(-0.08, 0.06, 57)
ax.hist(unpaired, bins, color=GREY, alpha=0.45, label="separate draws for A and B (wrong)")
ax.hist(diffs, bins, color=BLUE, alpha=0.85, label="paired: one draw scores both systems")
ax.axvline(0, color="#000000", linewidth=1.5)
for edge in (lo, hi):
    ax.axvline(edge, color=PINK, linestyle="--")
ax.text(lo - 0.002, ax.get_ylim()[1] * 0.92, f"{lo:+.3f}", color=PINK, ha="right", fontsize=14)
ax.text(hi + 0.002, ax.get_ylim()[1] * 0.92, f"{hi:+.3f}", color=PINK, ha="left", fontsize=14)
ax.set_xlabel("macro-F1 of B minus macro-F1 of A, per resample"); ax.set_ylabel("resamples (of 1,000)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
save(fig, "14_bootstrap_hist")

# 2. Interval width against the number of items: nested subsets of one 8,000-item pool.
rng2 = np.random.default_rng(42)
Y, A, B = simulate(rng2, 8000)
sizes = [250, 500, 1000, 2000, 4000, 8000]
rows = []
for m in sizes:
    d = paired(rng2, Y[:m], A[:m], B[:m])
    rows.append((macro_f1(Y[:m], B[:m]) - macro_f1(Y[:m], A[:m]), *np.quantile(d, [0.025, 0.975])))
rows = np.array(rows)
print("n, diff, lo, hi, width"); [print(m, r.round(3), round(r[2] - r[1], 3)) for m, r in zip(sizes, rows)]
fig, ax = new_fig(wide=True, height=4.6)
ax.errorbar(sizes, rows[:, 0], yerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]], fmt="o", color=BLUE, capsize=6, markersize=8)
for m, r in zip(sizes, rows):
    ax.text(m, r[2] + 0.006, f"width {r[2] - r[1]:.3f}", ha="center", fontsize=13, color=NAVY)
ax.axhline(0, color="#000000", linewidth=1.5)
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([f"{m:,}" for m in sizes]); ax.minorticks_off()
ax.set_xlabel("development items scored"); ax.set_ylabel("B minus A, 95% paired interval")
save(fig, "14_interval_vs_n")

# 3. Where system B's errors are: the five largest confusion pairs.
C = confusion(y, pred_b)
errors = int(C.sum() - np.trace(C))
pairs = sorted(((int(C[g, p]), g, p) for g in range(K) for p in range(K) if g != p), reverse=True)[:5][::-1]
fig, ax = new_fig()
ax.barh([f"{EMOTIONS[g]}\n→ {EMOTIONS[p]}" for _, g, p in pairs], [c for c, _, _ in pairs], color=PINK)
for i, (c, _, _) in enumerate(pairs):
    ax.text(c + 2, i, f"{c}  ({c / errors:.0%})", va="center", fontsize=14)
ax.set_xlim(0, 185); ax.set_xlabel(f"errors (of {errors})")
ax.set_title("Five pairs hold 57% of the errors")
save(fig, "14_confusions")
print("errors", errors, pairs)
