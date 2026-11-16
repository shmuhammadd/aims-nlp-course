"""Figures for lecture 3. The decoding figures use the five-word distribution of tutorial 3; the training curve is the
measured run stored in course/validation/tiny_decoder_run.json."""
import json

from figstyle import *

ROOT = Path(__file__).resolve().parents[3]

# 1. Measured run of extensions/train_tiny_decoder.py: training loss falls, development loss turns.
run = json.loads((ROOT / "course/validation/tiny_decoder_run.json").read_text())
steps = [r["step"] for r in run["history"]]
train = [r["train_loss"] for r in run["history"]]; dev = [r["dev_loss"] for r in run["history"]]
best = int(np.argmin(dev))
fig, ax = new_fig()
ax.plot(steps, train, color=BLUE, marker="o", markersize=4, label="training loss")
ax.plot(steps, dev, color=PINK, marker="o", markersize=4, label="development loss")
ax.axvline(steps[best], color=GREY, linestyle=":", linewidth=1.5)
ax.text(steps[best] + 2, 0.55, f"best checkpoint\nstep {steps[best]}, loss {dev[best]:.2f}", color=NAVY, fontsize=13)
ax.set_xlabel("training step"); ax.set_ylabel("loss (nats per token)"); ax.legend(loc="upper right")
ax.set_title("Keep the checkpoint at the turn")
save(fig, "03_curves")

# 2 and 3. One distribution over five next words, reshaped by temperature and by truncation.
words = ["leo", "kesho", "sasa", "hapa", "tena"]
base = np.array([0.50, 0.25, 0.15, 0.07, 0.03])


def temperature(p, T):
    q = p ** (1 / T)
    return q / q.sum()


def top_k(p, k):
    out = np.zeros_like(p); keep = np.argsort(-p)[:k]; out[keep] = p[keep]
    return out / out.sum()


def top_p(p, mass):
    order = np.argsort(-p); n = int(np.searchsorted(np.cumsum(p[order]), mass - 1e-12)) + 1
    out = np.zeros_like(p); out[order[:n]] = p[order[:n]]
    return out / out.sum()


def grouped(rows, name, title):
    fig, ax = new_fig()
    width = 0.8 / len(rows)
    for i, (label, probs, colour) in enumerate(rows):
        ax.bar(np.arange(5) + (i - (len(rows) - 1) / 2) * width, probs, width, color=colour, label=label)
    ax.set_xticks(range(5), words); ax.set_ylabel("probability"); ax.set_ylim(0, 0.8); ax.legend(); ax.set_title(title)
    save(fig, name)


grouped([("T = 0.5", temperature(base, 0.5), BLUE), ("T = 1 (the model)", base, GREY), ("T = 2", temperature(base, 2.0), PINK)],
        "03_temperature", "Temperature sharpens or flattens")
grouped([("the model", base, GREY), ("top-k, k = 2", top_k(base, 2), BLUE), ("top-p, p = 0.8", top_p(base, 0.8), PINK)],
        "03_truncation", "Truncation removes the tail")

# 4. C = 6ND, converted to days at a sustained 10 TFLOP/s (the rate used in the worked example).
D = np.logspace(8, 12, 50)
fig, ax = new_fig(wide=True, height=4.4)
for N, label, colour in [(1e8, "100 million parameters", BLUE), (1e9, "1 billion", VIOLET), (8e9, "8 billion", PINK)]:
    ax.plot(D, 6 * N * D / 1e13 / 86400, color=colour, label=label)
ax.axhline(1, color=GREY, linestyle=":", linewidth=1.5); ax.text(2e11, 1.4, "one day", color=GREY, fontsize=13)
ax.axhline(365, color=GREY, linestyle=":", linewidth=1.5); ax.text(1.2e8, 500, "one year", color=GREY, fontsize=13)
ax.scatter([2e9], [6 * 1e8 * 2e9 / 1e13 / 86400], color="#000000", zorder=5)
ax.text(2.4e9, 0.55, "worked example: 33.3 hours", fontsize=13)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training tokens D"); ax.set_ylabel("days at a sustained 10 TFLOP/s")
ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))
save(fig, "03_compute")
