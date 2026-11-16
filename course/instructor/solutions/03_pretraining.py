def top_p_filter(probs, p):
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    n_keep = int(np.searchsorted(cumulative, p - 1e-12)) + 1      # first position where the mass reaches p
    out = np.zeros_like(probs)
    out[order[:n_keep]] = probs[order[:n_keep]]
    return out / out.sum()


def train_with_selection(W0, x, y, held_x, held_y, lr, steps, every):
    W = W0.copy()
    best_W, best_step, best_loss = W.copy(), 0, float("inf")
    for step in range(steps + 1):
        if step % every == 0:
            held = float(loss(W, held_x, held_y))
            if held < best_loss:
                best_W, best_step, best_loss = W.copy(), step, held
        W -= lr * gradient(W, x, y)
    return best_W, best_step, best_loss
