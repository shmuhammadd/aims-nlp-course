def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads, mask):
    B, T, d = x.shape
    assert d % n_heads == 0
    dh = d // n_heads
    split = lambda z: z.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)       # (B, H, T, dh)
    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)                         # (B, H, T, T)
    attn = softmax(np.where(mask, scores, -np.inf))
    assert attn.shape == (B, n_heads, T, T)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ Wo
    assert out.shape == (B, T, d)
    return out, attn


def block_params(d_model, d_ff, bias=True):
    attention = 4 * d_model * d_model + (4 * d_model if bias else 0)
    mlp = 2 * d_model * d_ff + (d_ff + d_model if bias else 0)
    return attention + mlp + 4 * d_model
