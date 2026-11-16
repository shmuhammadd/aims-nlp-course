"""Figures for lecture 9. Same tone, CTC matrix and voice toy as tutorial 9 (seed 42, same order of random draws)."""
from figstyle import *

rng = np.random.default_rng(42)
sr = 16000; t = np.arange(sr) / sr
wave = .5 * np.sin(2 * np.pi * 440 * t) + .01 * rng.normal(size=sr)
frame_length, hop = 400, 160
frames = np.stack([wave[i:i + frame_length] for i in range(0, len(wave) - frame_length + 1, hop)])
spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1))
freqs = np.fft.rfftfreq(frame_length, 1 / sr)

# 1. One 25 ms frame under its window, and the spectrum it gives.
fig, axes = new_fig(wide=True, ncols=2, height=4.2)
ms = np.arange(frame_length) / sr * 1000
axes[0].plot(ms, frames[0], color=GREY, linewidth=1.5, label="samples")
axes[0].plot(ms, frames[0] * np.hanning(frame_length), color=BLUE, label="after the window")
axes[0].set_xlabel("time in one frame (ms)"); axes[0].set_ylabel("amplitude"); axes[0].legend(loc="upper right"); axes[0].set_ylim(-0.6, 0.95)
axes[0].set_title("400 samples = 25 ms")
axes[1].plot(freqs[:60], np.log(spectrum.mean(0) + 1e-8)[:60], color=PINK)
axes[1].axvline(440, color=GREY, linestyle=":", linewidth=1.5); axes[1].text(480, 2.5, "440 Hz", color=GREY, fontsize=14)
axes[1].set_xlabel("frequency (Hz), bins 40 Hz apart"); axes[1].set_ylabel("log magnitude"); axes[1].set_title(f"{spectrum.shape[0]} frames of {spectrum.shape[1]} bins")
save(fig, "09_stft")

# 2. CTC frame probabilities from the notebook, with the best symbol in each frame outlined.
symbols = ["_", "a", "b"]
P = np.array([[.1, .8, .1], [.6, .3, .1], [.2, .1, .7], [.5, .1, .4]])
fig, ax = new_fig(height=4.2)
ax.imshow(P.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for f in range(4):
    for k in range(3):
        ax.text(f, k, f"{P[f, k]:.1f}", ha="center", va="center", fontsize=15, color="white" if P[f, k] > .5 else "black")
    ax.add_patch(plt.Rectangle((f - .5, P[f].argmax() - .5), 1, 1, fill=False, edgecolor=PINK, linewidth=3))
ax.set_xticks(range(4), [f"frame {f + 1}" for f in range(4)]); ax.set_yticks(range(3), ["blank", "a", "b"])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title(r"Best path a _ b _  $\rightarrow$  ab")
save(fig, "09_ctc")

# 3. One voice in training, new voices at test (the notebook's section 5, same draws).
ratios = [1.5, 2.0, 2.5]; clip_t = np.arange(4000) / sr
def magnitude(word, voice):
    f0 = 200 * voice
    x = np.sin(2 * np.pi * f0 * clip_t) + np.sin(2 * np.pi * f0 * ratios[word] * clip_t) + .3 * rng.normal(size=len(clip_t))
    return np.abs(np.fft.rfft(x * np.hanning(len(x))))
def clip(word, voice):
    s = np.log(magnitude(word, voice) + 1e-3); return s - s.mean()
def ratio_guess(word, voice):
    m = magnitude(word, voice); i = int(m.argmax()); m[max(i - 5, 0):i + 6] = 0; j = int(m.argmax())
    return int(np.abs(np.array(ratios) - max(i, j) / min(i, j)).argmin())
def ratio_accuracy(voices, n=300):
    w = rng.integers(0, 3, n); v = rng.choice(voices, n); return float(np.mean([ratio_guess(a, b) == a for a, b in zip(w, v)]))
def dataset(voices, n):
    w = rng.integers(0, 3, n); v = rng.choice(voices, n); return np.array([clip(a, b) for a, b in zip(w, v)]), w
def train(voices, n=150):
    F, w = dataset(voices, n); return np.array([F[w == k].mean(0) for k in range(3)])
def accuracy(centroids, voices, n=300):
    F, w = dataset(voices, n); return float(((F @ centroids.T / np.linalg.norm(centroids, axis=1)).argmax(1) == w).mean())
new_voices = np.linspace(.8, 1.3, 11)
one_voice = train([1.0]); many_voices = train(np.linspace(.8, 1.3, 6))
rows = [(accuracy(one_voice, [1.0]), accuracy(one_voice, new_voices)), (accuracy(many_voices, [1.0]), accuracy(many_voices, new_voices)),
        (ratio_accuracy([1.0]), ratio_accuracy(new_voices))]
print("voice toy", rows)
fig, ax = new_fig()
x = np.arange(3)
ax.bar(x - .2, [r[0] for r in rows], .38, color=BLUE, label="training voice"); ax.bar(x + .2, [r[1] for r in rows], .38, color=PINK, label="new voices")
for i, r in enumerate(rows):
    ax.text(i - .2, r[0] + .02, f"{r[0]:.2f}", ha="center", fontsize=13); ax.text(i + .2, r[1] + .02, f"{r[1]:.2f}", ha="center", fontsize=13)
ax.axhline(1 / 3, color=GREY, linestyle=":", linewidth=1.5); ax.text(2.6, 1 / 3 + .02, "chance", color=GREY, fontsize=13, ha="right")
ax.set_xticks(x, ["trained on\none voice", "trained on\nsix voices", "voice-invariant\nfeature"]); ax.set_ylabel("accuracy on 300 toy clips")
ax.set_ylim(0, 1.3); ax.set_yticks([0, .5, 1]); ax.legend(loc="upper left", ncols=2)
save(fig, "09_shift")

# 4. Cascade arithmetic: answers correct = p*0.6 + (1-p)*0.2 (assumed rates), and whole-question accuracy (1-w)^L.
fig, axes = new_fig(wide=True, ncols=2, height=4.3)
p = np.linspace(0, 1, 101)
axes[0].plot(p, p * .6 + (1 - p) * .2, color=BLUE)
for q in [.9, .7]:
    axes[0].plot(q, q * .6 + (1 - q) * .2, "o", color=PINK, markersize=9); axes[0].text(q - .03, q * .6 + (1 - q) * .2 + .03, f"{q * .6 + (1 - q) * .2:.2f}", ha="right", fontsize=14)
axes[0].set_xlabel("share of questions transcribed correctly"); axes[0].set_ylabel("share of answers correct"); axes[0].set_ylim(0, .7)
axes[0].set_title("Assumed: 0.6 after a right transcript, 0.2 after a wrong one", fontsize=13)
L = np.arange(1, 16)
for w, colour in [(.05, BLUE), (.10, VIOLET), (.20, PINK)]:
    axes[1].plot(L, (1 - w) ** L, color=colour, marker="o", markersize=4, label=f"word error rate {w:.2f}")
axes[1].set_xlabel("words in the question"); axes[1].set_ylabel("whole question correct"); axes[1].set_ylim(0, 1.05); axes[1].legend(loc="upper right")
axes[1].set_title(r"$(1-w)^L$ if word errors are independent", fontsize=13)
save(fig, "09_cascade")
