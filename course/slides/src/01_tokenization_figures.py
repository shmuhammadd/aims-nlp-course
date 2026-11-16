"""Figures for lecture 1. Same strings, corpus and bigram model as tutorial 1, so slide numbers match the notebook."""
import unicodedata
from collections import Counter

from figstyle import *

# 1. UTF-8 bytes per code point for the notebook's strings.
strings = [("English\nsentence", "The class starts today."), ("Hausa\nsentence", "Darasi ya fara yau."),
           ("Hausa word\nwith hooked k", "ƙasa"), ("Yoruba\ngreeting", "Ẹ káàárọ̀"),
           ("Amharic\ngreeting", "ሰላም")]
ratio = [len(t.encode("utf-8")) / len(unicodedata.normalize("NFC", t)) for _, t in strings]
fig, ax = new_fig()
ax.bar(range(len(strings)), ratio, color=[BLUE, BLUE, VIOLET, VIOLET, PINK])
for i, r in enumerate(ratio):
    ax.text(i, r + 0.05, f"{r:.2f}", ha="center", fontsize=14)
ax.set_xticks(range(len(strings)), [n for n, _ in strings], fontsize=11)
ax.set_ylabel("UTF-8 bytes per code point"); ax.set_ylim(0, 3.4)
ax.set_title("Bytes are not letters")
save(fig, "01_bytes")

# 2. A BPE tokeniser learned on three English sentences, applied to held-out English, Hausa and Swahili.
train_docs = ["the class starts today", "the lab starts today", "the class ends soon"]
dev_docs = ["the lab ends soon"]


def pair_counts(words):
    counts = Counter()
    for symbols, n in words.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += n
    return counts


def apply_pair(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if symbols[i:i + 2] == pair: out.append(symbols[i] + symbols[i + 1]); i += 2
        else: out.append(symbols[i]); i += 1
    return tuple(out)


words, merges = Counter(tuple(w) for d in train_docs for w in d.split()), []
while not all(len(s) == 1 for s in words):
    counts = pair_counts(words)
    best = min(counts, key=lambda p: (-counts[p], p))
    new = Counter()
    for s, n in words.items():
        new[apply_pair(s, best)] += n
    words = new; merges.append(best)


def tokens_per_letter(text, k):
    total = 0
    for word in text.split():
        s = tuple(word)
        for pair in merges[:k]: s = apply_pair(s, pair)
        total += len(s)
    return total / len(text.replace(" ", ""))


ks = range(len(merges) + 1)
fig, ax = new_fig(wide=True, height=4.4)
for name, text, colour in [("English, held out", dev_docs[0], BLUE), ("Hausa: darasi ya fara yau", "darasi ya fara yau", PINK),
                           ("Swahili: darasa linaanza leo", "darasa linaanza leo", VIOLET)]:
    ax.plot(ks, [tokens_per_letter(text, k) for k in ks], color=colour, marker="o", markersize=4, label=name)
ax.set_xlabel("merges learned on three English sentences"); ax.set_ylabel("tokens per letter"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower left")
save(fig, "01_bpe_cost")

# 3 and 4. The character bigram baseline of the notebook.
vocab = ["<unk>", "<bos>", "<eos>"] + sorted(set("".join(train_docs)))
ids = {c: i for i, c in enumerate(vocab)}
encode = lambda doc: [1] + [ids.get(c, 0) for c in doc] + [2]


def count(docs):
    c = np.zeros((len(vocab), len(vocab)))
    for doc in docs:
        s = encode(doc)
        for a, b in zip(s, s[1:]): c[a, b] += 1
    return c


def ppl(c, doc, alpha):
    p = (c + alpha) / (c.sum(1, keepdims=True) + alpha * len(vocab))
    s = encode(doc)
    return float(np.exp(-np.mean([np.log(p[a, b]) for a, b in zip(s, s[1:])])))


c = count(train_docs)
alphas = np.logspace(-3, 0.3, 40)
fig, ax = new_fig()
ax.plot(alphas, [ppl(c, dev_docs[0], a) for a in alphas], color=BLUE, label="every transition seen in training")
ax.plot(alphas, [ppl(c, "the test ends today", a) for a in alphas], color=PINK, label="two unseen transitions")
ax.set_xscale("log"); ax.set_xlabel(r"smoothing strength $\alpha$"); ax.set_ylabel("development perplexity")
ax.set_ylim(0, 14); ax.legend(loc="upper left"); ax.set_title(r"The best $\alpha$ depends on the data")
save(fig, "01_smoothing")

s = encode(dev_docs[0]); alpha = 0.5
unigram = (c.sum(0) + alpha) / (c.sum() + alpha * len(vocab))
rungs = [("uniform", float(len(vocab)), GREY), ("unigram", float(np.exp(-np.mean(np.log(unigram[s[1:]])))), VIOLET),
         ("bigram", ppl(c, dev_docs[0], alpha), BLUE), ("bigram,\ndev leaked", ppl(count(train_docs + dev_docs), dev_docs[0], alpha), PINK)]
fig, ax = new_fig()
ax.bar(range(4), [r[1] for r in rungs], color=[r[2] for r in rungs])
for i, r in enumerate(rungs):
    ax.text(i, r[1] + 0.3, f"{r[1]:.2f}", ha="center", fontsize=14)
ax.set_xticks(range(4), [r[0] for r in rungs]); ax.set_ylabel("development perplexity")
ax.set_title("A ladder of baselines, and one leak")
save(fig, "01_ladder")
