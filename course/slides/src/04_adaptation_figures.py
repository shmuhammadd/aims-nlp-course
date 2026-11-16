"""Figures for lecture 04. Same toy and seed as tutorial 04, so slide numbers match the notebook."""
import itertools

from figstyle import *

rng = np.random.default_rng(42)
n, din, dout, r = 128, 12, 8, 2
X = rng.normal(size=(n, din)); Xtest = rng.normal(size=(64, din))
Wbase = rng.normal(0, .2, (din, dout))
true_delta = rng.normal(0, .2, (din, r)) @ rng.normal(0, .2, (r, dout))
Y = X @ (Wbase + true_delta); Ytest = Xtest @ (Wbase + true_delta)
A0 = rng.normal(0, .1, (din, r))


def train(A):
    A, B, curve = A.copy(), np.zeros((A.shape[1], dout)), []
    for step in range(401):
        err = X @ (Wbase + A @ B) - Y
        G = 2 * X.T @ err / err.size
        dA, dB = G @ B.T, A.T @ G
        A -= .5 * dA; B -= .5 * dB
        curve.append(np.mean((Xtest @ (Wbase + A @ B) - Ytest) ** 2))
    return curve


# 1. Held-out error of the adapter: rank 2 matches the rank-2 target, rank 1 cannot.
fig, ax = new_fig()
ax.semilogy(train(A0), color=BLUE, label="rank 2 (40 numbers)")
ax.semilogy(train(A0[:, :1]), color=PINK, label="rank 1 (20 numbers)")
ax.axhline(np.mean((Xtest @ Wbase - Ytest) ** 2), color=GREY, linestyle=":", label="frozen base")
ax.set_xlabel("gradient step"); ax.set_ylabel("held-out MSE"); ax.legend()
ax.set_title("The target change has rank 2")
save(fig, "04_lora_fit")

# 2. Demonstration order in the toy in-context classifier (notebook section 4).
k, d_feat = 4, 6
mu = np.zeros(d_feat); mu[0] = 1.0


def draw(m):
    y = rng.choice([-1, 1], size=m)
    return y[:, None] * mu + rng.normal(0, 1.0, (m, d_feat)), y


demo_y = np.array([1, 1, -1, -1])
demo_x = demo_y[:, None] * mu + rng.normal(0, 1.0, (k, d_feat))
dev_x, dev_y = draw(40); test_x, test_y = draw(2000)


def icl_predict(queries, order, gamma):
    xs, ys = demo_x[list(order)], demo_y[list(order)]
    w = gamma ** np.arange(k - 1, -1, -1)
    sim = np.exp(-((queries[:, None, :] - xs[None]) ** 2).sum(-1) / (2 * d_feat))
    return np.where((sim * w * ys).sum(1) >= 0, 1, -1)


orders = list(itertools.permutations(range(k)))
fig, ax = new_fig(wide=True, height=4.6)
for gamma, colour, name in [(1.0, BLUE, r"$\gamma=1$: every position counts the same"),
                            (0.5, PINK, r"$\gamma=0.5$: later demonstrations count more")]:
    acc = np.sort([(icl_predict(test_x, o, gamma) == test_y).mean() for o in orders])
    ax.plot(range(1, 25), acc, "o-", color=colour, label=name)
ax.axhline(0.5, color=GREY, linestyle=":"); ax.text(24.2, 0.505, "chance", color=GREY, fontsize=14, ha="right")
ax.set_xlabel("the 24 orderings of the same four demonstrations, sorted by accuracy")
ax.set_ylabel("test accuracy"); ax.set_ylim(0.4, 0.9)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2)
save(fig, "04_order")

# 3. Adapter size as a share of one 4096 x 4096 matrix: r(d+d)/d^2.
fig, ax = new_fig()
ranks = np.array([1, 2, 4, 8, 16, 32, 64])
ax.plot(ranks, 100 * ranks * 2 * 4096 / 4096 ** 2, "o-", color=BLUE)
ax.annotate("rank 8: 0.39%", xy=(8, 0.39), xytext=(14, 0.25), color=NAVY, fontsize=14,
            arrowprops=dict(arrowstyle="-", color=NAVY))
ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels(ranks)
ax.set_xlabel("rank r"); ax.set_ylabel("trainable share of the matrix (%)")
ax.set_title(r"$r(d_{in}+d_{out})\,/\,d_{in}d_{out}$,  $d=4096$")
save(fig, "04_param_share")

# 4. Which positions carry loss in response-only fine-tuning (schematic, no data).
tokens = ["<user>", "Translate", "to", "Hausa:", "good", "morning", "<assistant>", "Ina", "kwana", "<eos>", "<pad>", "<pad>"]
supervised = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
fig, ax = new_fig(wide=True, height=2.9)
for i, (tok, s) in enumerate(zip(tokens, supervised)):
    for row, text, fill in [(1, tok, SMOKE), (0, tok if s else "−100", BLUE if s else "#D9D9D9")]:
        ax.add_patch(plt.Rectangle((i, row), 0.94, 0.8, facecolor=fill, edgecolor=GREY))
        ax.text(i + 0.47, row + 0.4, text, ha="center", va="center", fontsize=11.5,
                color="white" if (row == 0 and s) else "black")
ax.text(-0.15, 1.4, "input", ha="right", va="center"); ax.text(-0.15, 0.4, "label", ha="right", va="center")
ax.set_xlim(-1.3, len(tokens)); ax.set_ylim(-0.1, 1.9); ax.axis("off")
save(fig, "04_label_mask")
