def sequence_dpo_loss(logp_w, logp_l, ref_w, ref_l, mask_w, mask_l, beta):
    delta_theta = (logp_w * mask_w).sum(1) - (logp_l * mask_l).sum(1)
    delta_ref = (ref_w * mask_w).sum(1) - (ref_l * mask_l).sum(1)
    return np.logaddexp(0, -beta * (delta_theta - delta_ref)).mean()


def group_advantages(rewards):
    centred = rewards - rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return np.where(std > 0, centred / np.where(std > 0, std, 1.0), 0.0)
