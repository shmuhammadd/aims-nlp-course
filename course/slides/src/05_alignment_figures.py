"""Figures for lecture 05. The refusal toy is the one in tutorial 05 (seed 42), so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 1. The post-training pipeline (schematic, no data).
fig, ax = new_fig(wide=True, height=3.6)
stages = [("Pretrain", "raw text", "day 3", GREY),
          ("SFT", "prompt and\ntarget response", "day 4", BLUE),
          ("Preference / RL", "comparisons\nor rewards", "today", VIOLET),
          ("Safety evaluation", "refusal, over-refusal,\nper language", "today, days 6 and 10", PINK)]
for i, (name, signal, day, colour) in enumerate(stages):
    x = i * 3.1
    ax.add_patch(plt.Rectangle((x, 1.0), 2.4, 1.5, facecolor=colour, edgecolor="none"))
    ax.text(x + 1.2, 2.05, name, ha="center", va="center", color="white", fontsize=16, fontweight="bold")
    ax.text(x + 1.2, 1.45, signal, ha="center", va="center", color="white", fontsize=12.5)
    ax.text(x + 1.2, 0.72, day, ha="center", va="center", color=NAVY, fontsize=13)
    if i < 3:
        ax.annotate("", xy=(x + 3.05, 1.75), xytext=(x + 2.45, 1.75), arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2))
ax.annotate("", xy=(7.4, 2.55), xytext=(10.5, 2.55),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, linewidth=1.8, linestyle="--", connectionstyle="arc3,rad=0.45"))
ax.text(8.95, 3.62, "failures become new training pairs", ha="center", color=NAVY, fontsize=13)
ax.set_xlim(-0.2, 11.9); ax.set_ylim(0.4, 3.9); ax.axis("off")
save(fig, "05_pipeline")

# 2. The DPO loss as a function of how far the policy has moved from the reference.
fig, ax = new_fig()
gap = np.linspace(-3, 3, 241)
for beta, colour in [(0.5, VIOLET), (2.0, BLUE), (5.0, PINK)]:
    ax.plot(gap, np.logaddexp(0, -beta * gap), color=colour, label=rf"$\beta={beta:g}$")
ax.plot([0], [np.log(2)], "o", color="black"); ax.text(0.15, np.log(2) + 0.12, r"$\log 2$", fontsize=15)
ax.annotate("", xy=(0.3, np.logaddexp(0, -2 * 0.3)), xytext=(0.88, 0.8), arrowprops=dict(arrowstyle="-|>", color=NAVY)); ax.text(0.9, 0.75, "worked example:\n0.4375", fontsize=13, color=NAVY)
ax.set_xlabel(r"$\Delta_\theta-\Delta_{ref}$"); ax.set_ylabel("DPO loss for one pair"); ax.set_ylim(0, 3); ax.legend()
ax.set_title("Loss falls as the policy prefers the winner")
save(fig, "05_dpo_loss")

# 3. Refusal and over-refusal for three preference sets (tutorial section 4).
kinds = {"harmful": (1.5, 1.5), "plain safe": (-1.5, -1.5), "safe, sounds risky": (1.5, -1.5)}


def prompts(n, sounds_risky, harmful):
    return np.c_[rng.normal(sounds_risky, 1.0, n), rng.normal(harmful, 1.0, n)]


train = {name: prompts(200, *centre) for name, centre in kinds.items()}
test = {name: prompts(1000, *centre) for name, centre in kinds.items()}


def train_refusal_policy(names, beta=2.0, steps=300, lr=0.1):
    X = np.r_[tuple(train[name] for name in names)]
    sign = np.r_[tuple(np.full(len(train[name]), 1.0 if name == "harmful" else -1.0) for name in names)]
    w, b = np.zeros(2), 0.0
    for _ in range(steps):
        delta = sign * (X @ w + b)
        g = beta * (sigmoid(beta * delta) - 1) * sign
        w -= lr * (g[:, None] * X).mean(0); b -= lr * g.mean()
    return w, b


sets = {"refusals only": ["harmful"], "plus plain safe": ["harmful", "plain safe"],
        "plus safe that\nsounds risky": ["harmful", "plain safe", "safe, sounds risky"]}
fig, ax = new_fig(wide=True, height=4.4)
bars = [("harmful", BLUE, "harmful prompts refused (want high)"),
        ("safe, sounds risky", PINK, "safe prompts that sound risky, refused (want low)"),
        ("plain safe", GREY, "plain safe prompts refused (want low)")]
for s, names in enumerate(sets.values()):
    w, b = train_refusal_policy(names)
    for j, (name, colour, label) in enumerate(bars):
        rate = (test[name] @ w + b > 0).mean()
        ax.bar(s + (j - 1) * 0.27, rate, 0.25, color=colour, label=label if s == 0 else None)
        ax.text(s + (j - 1) * 0.27, rate + 0.02, f"{rate:.2f}", ha="center", fontsize=13)
ax.set_xticks(range(3)); ax.set_xticklabels(sets.keys()); ax.set_ylim(0, 1.12); ax.set_ylabel("share refused")
ax.set_xlabel("preference set used for training")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=12)
save(fig, "05_overrefusal")

# 4. Disagreeing annotators: the mean pair loss has an interior minimum (tutorial section 2).
fig, ax = new_fig()
theta = np.linspace(-1, 3, 201)
for share, colour in [(1.0, BLUE), (0.7, PINK)]:
    loss = share * np.logaddexp(0, -2 * theta) + (1 - share) * np.logaddexp(0, 2 * theta)
    ax.plot(theta, loss, color=colour, label=f"{int(share * 10)} of 10 prefer A")
best = np.log(0.7 / 0.3) / 2
ax.axvline(best, color=PINK, linestyle=":"); ax.text(best + 0.08, 1.5, rf"minimum at $\theta={best:.2f}$", color=PINK, fontsize=14)
ax.set_xlabel(r"$\theta$ (log-odds of response A)"); ax.set_ylabel(r"mean DPO loss, $\beta=2$"); ax.legend(loc="upper center")
ax.set_title("Disagreement stops the policy")
save(fig, "05_disagreement")
