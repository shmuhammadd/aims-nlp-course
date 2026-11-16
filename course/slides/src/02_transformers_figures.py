"""Figures for lecture 2. The decoder is the one in tutorial 2 (seed 42, same draw order), so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


T, d, V = 6, 8, 12
emb = rng.normal(0, .2, (V, d)); pos = rng.normal(0, .2, (T, d))
Wq, Wk, Wv, Wo = [rng.normal(0, .2, (d, d)) for _ in range(4)]
W1 = rng.normal(0, .2, (d, 2 * d)); W2 = rng.normal(0, .2, (2 * d, d))
head = rng.normal(0, .2, (d, V))
norm = lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def decoder(tokens, causal=True):
    x = emb[tokens] + pos[:len(tokens)]
    z = norm(x); q, k, v = z @ Wq, z @ Wk, z @ Wv
    scores = q @ k.T / np.sqrt(d)
    if causal: scores = np.where(np.triu(np.ones(scores.shape, dtype=bool), 1), -np.inf, scores)
    a = softmax(scores)
    h = x + (a @ v) @ Wo
    h = h + np.maximum(norm(h) @ W1, 0) @ W2
    return norm(h) @ head, a


seq = np.array([1, 3, 5, 7, 2, 4])
logits, a = decoder(seq)

# 1. The causal attention weights of the notebook's block.
fig, ax = new_fig()
ax.imshow(a, cmap="Blues", vmin=0, vmax=1)
for i in range(T):
    for j in range(T):
        ax.text(j, i, f"{a[i, j]:.2f}" if j <= i else "0", ha="center", va="center", fontsize=12,
                color="white" if a[i, j] > 0.5 else ("#000000" if j <= i else GREY))
ax.set_xlabel("key position (attended to)"); ax.set_ylabel("query position")
ax.set_xticks(range(T)); ax.set_yticks(range(T)); ax.spines[:].set_visible(False)
ax.set_title("Each row sees its prefix only")
save(fig, "02_attention")

# 2. Change the tokens from position `cut` on; measure the largest change in the logits before `cut`.
cuts = range(1, T)
leak = {True: [], False: []}
for causal in (True, False):
    base = decoder(seq, causal)[0]
    for cut in cuts:
        alt = seq.copy(); alt[cut:] = (alt[cut:] + 1) % V
        leak[causal].append(np.abs(base[:cut] - decoder(alt, causal)[0][:cut]).max())
fig, ax = new_fig()
ax.plot(cuts, leak[False], color=PINK, marker="o", label="no mask")
ax.plot(cuts, leak[True], color=BLUE, marker="o", label="causal mask")
ax.set_xlabel("first changed position"); ax.set_ylabel("largest change in earlier logits")
ax.set_xticks(list(cuts)); ax.legend(); ax.set_title("The mask makes the change exactly 0")
save(fig, "02_leak")

# 3. Largest attention weight over 16 random keys, with and without the 1/sqrt(d_k) factor.
dims = [4, 16, 64, 256, 1024]
r = np.random.default_rng(42)
raw, scaled = [], []
for dk in dims:
    q, k = r.normal(size=(500, 1, dk)), r.normal(size=(500, 16, dk))
    s = (q * k).sum(-1)
    raw.append(softmax(s).max(-1).mean()); scaled.append(softmax(s / np.sqrt(dk)).max(-1).mean())
fig, ax = new_fig()
ax.plot(dims, raw, color=PINK, marker="o", label="no scaling")
ax.plot(dims, scaled, color=BLUE, marker="o", label=r"divided by $\sqrt{d_k}$")
ax.axhline(1 / 16, color=GREY, linestyle=":", linewidth=1.5); ax.text(4, 0.09, "uniform: 1/16", color=GREY, fontsize=13)
ax.set_xscale("log", base=2); ax.set_xticks(dims, [str(x) for x in dims]); ax.set_ylim(0, 1.05)
ax.set_xlabel(r"head width $d_k$"); ax.set_ylabel("mean largest weight, 16 keys"); ax.legend(loc="center left")
ax.set_title("Unscaled scores saturate the softmax")
save(fig, "02_scaling")

# 4. RoPE: the score of a fixed query and key as a function of their distance, for three query positions.
def rope(x, position, base=10000.0):
    half = len(x) // 2
    theta = position * base ** (-np.arange(half) / half)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * np.cos(theta) - x[1::2] * np.sin(theta)
    out[1::2] = x[0::2] * np.sin(theta) + x[1::2] * np.cos(theta)
    return out


r = np.random.default_rng(42)
qv, kv = r.normal(size=64), r.normal(size=64)
fig, ax = new_fig()
offsets = np.arange(0, 33)
for m, colour, style, width in [(40, BLUE, "-", 5), (100, PINK, "--", 3), (400, "#000000", ":", 2)]:
    ax.plot(offsets, [rope(qv, m) @ rope(kv, m - o) for o in offsets], color=colour, linestyle=style, linewidth=width, label=f"query at position {m}")
ax.set_xlabel("distance between query and key"); ax.set_ylabel("score"); ax.legend()
ax.set_title("With RoPE the three curves coincide")
save(fig, "02_rope")

# 5. Where the parameters of one block sit. Counts follow from the published configurations.
def shares(parts):
    total = sum(parts.values())
    return {k: 100 * v / total for k, v in parts.items()}, total


gpt2, gpt2_total = shares({"attention": 4 * 768 * 768 + 4 * 768, "MLP": 2 * 768 * 3072 + 3072 + 768, "norms": 4 * 768})
llama, llama_total = shares({"attention": 2 * 4096 * 4096 + 2 * 4096 * 1024, "MLP": 3 * 4096 * 14336, "norms": 2 * 4096})
fig, ax = new_fig(wide=True, height=3.6)
for y, (name, s, total) in enumerate([("Llama-3.1-8B block\n218.1 M", llama, llama_total), ("GPT-2 small block\n7.09 M", gpt2, gpt2_total)]):
    left = 0
    for part, colour in [("attention", BLUE), ("MLP", VIOLET), ("norms", PINK)]:
        ax.barh(y, s[part], left=left, color=colour, label=part if y == 0 else None)
        if s[part] > 5:
            ax.text(left + s[part] / 2, y, f"{part} {s[part]:.0f}%", ha="center", va="center", color="white", fontsize=14)
        left += s[part]
ax.set_yticks([0, 1], ["Llama-3.1-8B block\n218.1 M", "GPT-2 small block\n7.09 M"]); ax.set_xlabel("share of the block's parameters (%)"); ax.set_xlim(0, 100)
save(fig, "02_params")
