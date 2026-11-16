def cohen_kappa(a, b):
    labels = np.unique(np.r_[a, b])
    p_o = np.mean(a == b)
    p_e = sum(np.mean(a == c) * np.mean(b == c) for c in labels)
    return 1.0 if np.isclose(p_e, 1) else (p_o - p_e) / (1 - p_e)


def macro_f1_by_language(y, p, lang, classes):
    out = {}
    for l in np.unique(lang):
        yl, pl, scores = y[lang == l], p[lang == l], []
        for c in classes:
            tp = np.sum((yl == c) & (pl == c)); fp = np.sum((yl != c) & (pl == c)); fn = np.sum((yl == c) & (pl != c))
            if tp + fp + fn:
                scores.append(2 * tp / (2 * tp + fp + fn))
        out[str(l)] = float(np.mean(scores))
    return out
