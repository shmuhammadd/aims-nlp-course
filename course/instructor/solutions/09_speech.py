def ctc_collapse(path, blank="_"):
    merged = [c for i, c in enumerate(path) if i == 0 or c != path[i - 1]]
    return "".join(c for c in merged if c != blank)


def cer(ref, hyp):
    if not ref:
        if not hyp:
            return 0.0
        raise ValueError("CER undefined for empty reference and nonempty hypothesis")
    return edit_distance(ref, hyp) / len(ref)
