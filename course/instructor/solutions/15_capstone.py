def selective_scores(answers, gold):
    answered = [a is not None for a in answers]
    right = [a is not None and a == g for a, g in zip(answers, gold)]
    coverage = sum(answered) / len(answers)
    selective = sum(right) / sum(answered) if any(answered) else None
    return coverage, selective, sum(right) / len(answers)


def with_decoy(cases, shift=1):
    return [{**c, "image": cases[(i + shift) % len(cases)]["image"]} for i, c in enumerate(cases)]
