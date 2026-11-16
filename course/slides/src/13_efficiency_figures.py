"""Figures for lecture 13. Arithmetic and simulations repeat tutorial 13 (seed 42, same draw order), so slide numbers match the notebook."""
from figstyle import *

rng = np.random.default_rng(42)

# 1. Ideal weight memory, parameters x bits / 8, against a free Colab T4.
models = {"SmolVLM 2.2B": 2.2, "Gemma 3 4B": 4.0, "Llama-3.1-8B": 8.03, "Gemma 3 12B": 12.0}
fig, ax = new_fig(wide=True, height=4.4)
for i, (bits, colour) in enumerate([(16, BLUE), (8, VIOLET), (4, PINK)]):
    gb = [n * bits / 8 for n in models.values()]
    bars = ax.bar(np.arange(len(models)) + (i - 1) * 0.27, gb, 0.27, color=colour, label=f"{bits}-bit")
    for b, g in zip(bars, gb):
        ax.text(b.get_x() + b.get_width() / 2, g + 0.3, f"{g:.1f}", ha="center", fontsize=13)
ax.axhline(15, color="#000000", linestyle="--", linewidth=2); ax.text(-0.45, 15.4, "free Colab T4: 15 GB", fontsize=14)
ax.set_xticks(range(len(models)), list(models)); ax.set_ylabel("weights (GB)"); ax.set_ylim(0, 27); ax.legend(loc="upper center", ncols=3)
save(fig, "13_weights")

# 2. KV cache of Llama-3.1-8B (32 layers, head dimension 128, 16-bit) against context length.
T = 2 ** np.arange(10, 18)
fig, ax = new_fig()
for heads, colour, name in [(32, GREY, "32 KV heads (no sharing)"), (8, BLUE, "8 KV heads (as built)")]:
    ax.plot(T, 2 * T * 32 * heads * 128 * 2 / 2 ** 30, "o-", color=colour, label=name)
ax.plot(4096, 0.5, "o", color=PINK, markersize=11); ax.text(5200, 0.3, "4,096 tokens: 0.5 GiB", color=PINK, fontsize=14)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xticks([1024, 4096, 16384, 131072], ["1k", "4k", "16k", "128k"]); ax.set_yticks([0.125, 0.5, 2, 16, 64], ["0.125", "0.5", "2", "16", "64"])
ax.set_xlabel("tokens in context"); ax.set_ylabel("KV cache (GiB), batch 1"); ax.legend(loc="upper left")
ax.set_title("The cache grows with the context")
save(fig, "13_kv")

# Tutorial section 3: a 64 x 32 matrix whose output channel 0 is 30 times larger.
W = rng.normal(size=(64, 32)); W[:, 0] *= 30
X = rng.normal(size=(128, 64))


def quantize(W, bits, per_channel=True):
    qmax = 2 ** (bits - 1) - 1
    maxabs = np.max(np.abs(W), axis=0, keepdims=True) if per_channel else np.max(np.abs(W))
    scale = np.where(maxabs == 0, 1, maxabs / qmax)
    return np.clip(np.rint(W / scale), -qmax, qmax) * scale


# 3. Weight error for per-tensor and per-channel scales at 4 and 8 bits.
fig, ax = new_fig()
for i, (pc, colour, name) in enumerate([(False, PINK, "one scale for the matrix"), (True, BLUE, "one scale per channel")]):
    mse = [np.mean((W - quantize(W, b, pc)) ** 2) for b in (4, 8)]
    bars = ax.bar(np.arange(2) + (i - 0.5) * 0.38, mse, 0.38, color=colour, label=name)
    for b, m in zip(bars, mse):
        ax.text(b.get_x() + b.get_width() / 2, m * 1.25, f"{m:.2g}", ha="center", fontsize=14)
ax.set_yscale("log"); ax.set_ylim(1e-4, 30); ax.set_xticks([0, 1], ["4 bits", "8 bits"]); ax.set_ylabel("weight mean squared error")
ax.legend(loc="upper right"); ax.set_title("One large channel ruins a shared scale")
save(fig, "13_quant")

# 4. Tutorial section 5: cosine between the full-precision direction and the quantised-model direction.
d = 128
W1, W2 = rng.normal(size=(d, d)) / np.sqrt(d), rng.normal(size=(d, d)) / np.sqrt(d)
lang = rng.normal(size=d) / np.sqrt(d); content = rng.normal(size=(400, d))
X_a, X_b = content - lang, content + lang


def direction(A, B):
    h = lambda x: np.tanh(np.tanh(x @ A) @ B)
    return h(X_b).mean(0) - h(X_a).mean(0)


v_full = direction(W1, W2); bits_list = [8, 6, 4, 3, 2]; cos = []
for bits in bits_list:
    v = direction(quantize(W1, bits), quantize(W2, bits))
    cos.append(v @ v_full / np.linalg.norm(v) / np.linalg.norm(v_full))
fig, ax = new_fig()
ax.plot(bits_list, cos, "o-", color=BLUE)
for b, c in zip(bits_list, cos):
    ax.text(b, c + 0.025, f"{c:.3f}", fontsize=14, color=NAVY, ha="center")
ax.set_xticks(bits_list); ax.set_xlim(1.6, 9); ax.set_ylim(0.4, 1.1); ax.invert_xaxis()
ax.set_xlabel("bits per weight"); ax.set_ylabel("cosine with full-precision direction")
ax.set_title("Toy: a direction under quantisation")
save(fig, "13_direction")

# 5. Tutorial section 6: speculative sampling returns the target distribution.
p = np.array([0.50, 0.25, 0.15, 0.07, 0.03]); q = np.array([0.30, 0.40, 0.10, 0.10, 0.10]); n = 200_000
x = rng.choice(5, size=n, p=q); accept = rng.random(n) < np.minimum(1, p[x] / q[x])
left = np.maximum(p - q, 0); left /= left.sum()
out = np.where(accept, x, rng.choice(5, size=n, p=left)); emp = np.bincount(out, minlength=5) / n
fig, ax = new_fig()
for i, (vals, colour, name) in enumerate([(q, GREY, "draft q"), (p, BLUE, "target p"), (emp, PINK, "speculative output")]):
    ax.bar(np.arange(5) + (i - 1) * 0.28, vals, 0.28, color=colour, label=name)
ax.set_xticks(range(5), [f"token {i + 1}" for i in range(5)]); ax.set_ylabel("probability"); ax.legend()
ax.set_title(f"Accepted {accept.mean():.0%} of drafts; output matches p")
save(fig, "13_spec")
