def kv_cache_bytes(B, T, L, H_kv, d_h, bytes_per=2):
    return 2 * B * T * L * H_kv * d_h * bytes_per                 # the leading 2 counts keys and values


def quantise_rows(W, bits):
    qmax = 2 ** (bits - 1) - 1
    maxabs = np.abs(W).max(axis=1, keepdims=True)
    scale = np.where(maxabs == 0, 1.0, maxabs / qmax)             # an all-zero row keeps scale 1
    return np.clip(np.rint(W / scale), -qmax, qmax).astype(np.int8), scale


def dequantise(q, scale):
    return q * scale
