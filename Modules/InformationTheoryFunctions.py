import numpy as np

# Mutual Information in bits
def mutualinformation(py, px, pxgy):
    card_y = len(py)
    card_x = len(px)
    
    if pxgy.shape != (card_x, card_y):
        raise ValueError("Dimensionality of p(x|y) does not match p(x), p(y)!")
    
    MI = 0
    for i in range(card_y):
        MI += py[i] * kl_divergence_bits(pxgy[:, i], px)
    
    return MI

# Entropy in bits
def entropybits(p):
    # return -np.sum(p * np.log2(p))
    return -np.sum(np.where(p > 0.000001, p * np.log2(p), 0))

# Kullback-Leibler Divergence in bits
def kl_divergence_bits(p_x, p0_x, eps=1e-15):
    if np.any((p0_x == 0) & (p_x!=0)):
        print(p_x)
        print(p0_x)
        raise ValueError("Zeros in denominator before kl_divergence computation!")
    
    p0_x_safe = np.maximum(p0_x, eps)

    # if np.any((p0_x_safe < 0.000001)):
    #     print(p_x)
    #     print(p0_x)
    #     raise ValueError("Zeros in denominator before kl_divergence computation!")

    # kl_div = np.sum(p_x * np.log2(p_x / p0_x))
    with np.errstate(divide='ignore', invalid='ignore'):
        kl_div = np.sum(np.where(p_x > 0.000001, p_x * (np.log2(p_x) - np.log2(p0_x_safe)), 0))

    return kl_div