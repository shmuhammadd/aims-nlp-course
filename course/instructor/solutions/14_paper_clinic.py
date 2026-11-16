def paired_bootstrap(metric, y_true, pred_a, pred_b, n, rng):
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs.append(metric(y_true[idx], pred_b[idx]) - metric(y_true[idx], pred_a[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def mcnemar_exact(b, c):
    return sum(math.comb(b + c, k) for k in range(b, b + c + 1)) / 2 ** (b + c)
