"""Figures for lecture 12. Computed figures repeat tutorial 12 (seed 42, same draw order), so slide numbers match the notebook."""
import re
from collections import Counter

from matplotlib.patches import FancyBboxPatch

from figstyle import *

rng = np.random.default_rng(42)


def box(ax, x, y, w, h, text, colour=BLUE, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", edgecolor=colour, facecolor=fill, linewidth=2.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=15, color="#000000")


def arrow(ax, start, end, colour="#000000", style="-|>"):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color=colour, linewidth=2.2))


def blank(ax, xmax, ymax):
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")


# 1. The pipeline, with the measurement that belongs to each stage.
fig, ax = new_fig(wide=True, height=3.6)
blank(ax, 12.4, 3.6)
stages = [(0.2, "Question"), (2.7, "Retriever\n(index of chunks\nwith source IDs)"), (5.6, "Top-k chunks"), (8.1, "Reader\n(language model)"), (10.5, "Answer + source,\nor abstain")]
widths = [1.9, 2.3, 1.9, 1.9, 1.8]
for (x, text), w in zip(stages, widths):
    box(ax, x, 1.7, w, 1.5, text, fill="white" if "Retriever" not in text and "Reader" not in text else "#E6E6FA")
for (x, _), w, (nx, _) in zip(stages, widths, stages[1:]):
    arrow(ax, (x + w + 0.03, 2.45), (nx - 0.03, 2.45))
ax.text(4.4, 0.75, "Score retrieval here:\nhit@k, recall@k, MRR", ha="center", color=BLUE, fontsize=15)
ax.text(9.9, 0.75, "Score the answer here: correct?\nsupported by the cited source? abstained?", ha="center", color=PINK, fontsize=15)
arrow(ax, (5.9, 1.25), (6.4, 1.68), BLUE); arrow(ax, (10.9, 1.25), (11.3, 1.68), PINK)
save(fig, "12_pipeline")

# The tutorial's TF-IDF index (sections 1 to 4).
docs = {"D1": "The library opens at eight on weekdays.", "D2": "The computing lab closes at eighteen on weekdays.",
        "D3": "Borrowed books must be returned within seven days.", "D4": "The study room contains a projector and a whiteboard."}
tokens = lambda s: re.findall(r"\w+", s.lower())
vocab = sorted(set(t for d in docs.values() for t in tokens(d))); vid = {t: i for i, t in enumerate(vocab)}


def bow(s):
    x = np.zeros(len(vocab))
    for t, c in Counter(tokens(s)).items():
        if t in vid: x[vid[t]] = c
    return x


counts = np.stack([bow(d) for d in docs.values()])
idf = np.log((1 + len(docs)) / (1 + (counts > 0).sum(0))) + 1
X = counts * idf; X /= np.linalg.norm(X, axis=1, keepdims=True)
docids = list(docs)


def scores(q):
    v = bow(q) * idf; v /= max(np.linalg.norm(v), 1e-12)
    return dict(zip(docids, X @ v))


all_q = [("When does the library open?", "D1"), ("When does the lab close?", "D2"), ("How long may books be borrowed?", "D3"),
         ("What equipment is in the study room?", "D4"), ("What time can I start reading in the building with books?", "D1"),
         ("Until what hour can I use the computers?", "D2"), ("When must I bring back what I borrowed?", "D3"),
         ("Is there a screen for presentations?", "D4"), ("Where can I project slides?", "D4"), ("Maktaba inafunguliwa saa ngapi?", "D1")]
conditions = {"retrieved": lambda g: set(docids), "oracle": lambda g: {g},
              "shuffled": lambda g: {docids[(docids.index(g) + 1) % 4]}, "missing": lambda g: set(docids) - {g}}

# 2. Four evidence conditions: correct, abstained, wrong source, out of ten questions.
fig, ax = new_fig()
for n, (name, allowed) in enumerate(conditions.items()):
    cited = []
    for q, gold in all_q:
        s = scores(q); best = max(allowed(gold), key=lambda d: (s[d], -docids.index(d)))
        cited.append(best if s[best] >= 0.2 else None)
    correct = sum(c == g for c, (_, g) in zip(cited, all_q)); abstained = sum(c is None for c in cited); wrong = 10 - correct - abstained
    for bottom, value, colour, label in [(0, correct, BLUE, "correct source"), (correct, abstained, GREY, "abstained"), (correct + abstained, wrong, PINK, "wrong source")]:
        ax.bar(name, value, bottom=bottom, color=colour, label=label if n == 0 else None)
        if value: ax.text(n, bottom + value / 2, str(value), ha="center", va="center", color="white", fontsize=16, fontweight="bold")
ax.set_ylabel("questions (of 10)"); ax.set_ylim(0, 13.5); ax.set_yticks([0, 5, 10]); ax.legend(loc="upper center", ncols=3, fontsize=12, columnspacing=0.8, handlelength=1)
ax.set_title("Same questions, four contexts")
save(fig, "12_ablation")

# 3. Tutorial section 8: majority vote and pass@k against the number of samples.
ks = [1, 3, 5, 9, 17, 33]; rows = []
for k in ks:
    out = []
    for m in (4, 1):
        c = rng.multinomial(k, [0.4] + [0.6 / m] * m, size=4000)
        out += [(c[:, 0] > c[:, 1:].max(1)).mean(), (c[:, 0] > 0).mean()]
    rows.append((out[0], out[2], out[1]))
spread, agree, pass_k = zip(*rows)
fig, ax = new_fig(wide=True, height=4.4)
ax.plot(ks, pass_k, "o--", color="#000000", label="pass@k (an oracle picks)")
ax.plot(ks, spread, "o-", color=BLUE, label="vote, errors spread over 4 values")
ax.plot(ks, agree, "o-", color=PINK, label="vote, errors agree")
ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks]); ax.set_ylim(0, 1.05)
ax.set_xlabel("samples per question, k   (each sample right with probability 0.4)"); ax.set_ylabel("share of questions right")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
save(fig, "12_vote")

# 4. The bounded loop: every call crosses the dispatcher; observations come back as data.
fig, ax = new_fig(height=5.2)
blank(ax, 6, 5.2)
box(ax, 0.3, 3.9, 2.3, 1.0, "Controller\n(model)", fill="#E6E6FA")
box(ax, 3.4, 3.9, 2.3, 1.0, "Dispatcher\nchecks the call", colour=PINK)
box(ax, 3.4, 1.9, 2.3, 1.0, "Tool or\nretriever")
box(ax, 0.3, 1.9, 2.3, 1.0, "Observation\n(data only)")
arrow(ax, (2.65, 4.4), (3.35, 4.4)); arrow(ax, (4.55, 3.85), (4.55, 2.95)); arrow(ax, (3.35, 2.4), (2.65, 2.4)); arrow(ax, (1.45, 2.95), (1.45, 3.85))
ax.text(3.0, 4.62, "call", ha="center", fontsize=13); ax.text(4.7, 3.4, "if valid", fontsize=13, color=PINK)
ax.text(3.0, 0.95, "Stop when the answer is ready\nor the step budget is spent", ha="center", fontsize=15, color=NAVY)
ax.text(3.0, 0.2, "Log every call and observation", ha="center", fontsize=14, color=GREY)
save(fig, "12_loop")
