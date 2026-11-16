def steering_vector(H, labels, source, target):
    return H[labels == target].mean(0) - H[labels == source].mean(0)


def clamp(h, direction, target):
    return h + np.outer(target - h @ direction, direction)
