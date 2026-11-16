def micro_macro_f1(Y_true, Y_pred):
    Y_true, Y_pred = np.asarray(Y_true).astype(bool), np.asarray(Y_pred).astype(bool)
    tp, fp, fn = (Y_true & Y_pred).sum(0), (~Y_true & Y_pred).sum(0), (Y_true & ~Y_pred).sum(0)
    denom = 2 * tp + fp + fn                                    # zero when a label has no positives at all
    per_label = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    micro = 2 * tp.sum() / denom.sum() if denom.sum() else 0.0
    return float(micro), float(per_label.mean())


def log_loss_bits(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log2(p) + (1 - y) * np.log2(1 - p))))
