def response_only_labels(input_ids, attention_mask, prompt_lens):
    labels = input_ids.copy()
    for row, n in enumerate(prompt_lens):
        labels[row, :n] = -100
    labels[attention_mask == 0] = -100
    return labels


def lora_param_count(d_in, d_out, r):
    return r * (d_in + d_out)


def merge_lora(W, A, B, scale):
    return W + scale * A @ B
