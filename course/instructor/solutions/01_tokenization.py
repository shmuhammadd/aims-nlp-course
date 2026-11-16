def bpe_merge_step(words):
    counts = pair_counts(words)
    best = min(counts, key=lambda pair: (-counts[pair], pair))
    new_words = Counter()
    for symbols, n in words.items():
        new_words[apply_pair(symbols, best)] += n
    return best, dict(new_words)


def bits_per_byte(token_logprobs, text):
    return -np.sum(token_logprobs) / np.log(2) / len(text.encode("utf-8"))
