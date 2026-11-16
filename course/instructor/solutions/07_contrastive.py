def contrastive_loss(S):
    rows = S - np.log(np.exp(S - S.max(1, keepdims=True)).sum(1, keepdims=True)) - S.max(1, keepdims=True)
    cols = S - np.log(np.exp(S - S.max(0, keepdims=True)).sum(0, keepdims=True)) - S.max(0, keepdims=True)
    return -(np.diag(rows).mean() + np.diag(cols).mean()) / 2


def recall_at_k(S, k):
    d = np.diag(S)
    return float(((S > d[:, None]).sum(1) < k).mean()), float(((S > d[None, :]).sum(0) < k).mean())
