import math


def decoy_test(real_correct, decoy_correct):
    b = int((real_correct & ~decoy_correct).sum())
    c = int((~real_correct & decoy_correct).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, sum(math.comb(b + c, k) for k in range(b, b + c + 1)) / 2 ** (b + c)


def visual_tokens(height, width, patch, pool=1):
    return (height // patch // pool) * (width // patch // pool)
