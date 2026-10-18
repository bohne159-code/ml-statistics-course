"""
LÖSUNG 06: Informationstheorie für Machine Learning & Deep Learning
"""

import math
from typing import List


def _log_base(val: float, base: float) -> float:
    if base == 2.0:
        return math.log2(val)
    elif base == math.e:
        return math.log(val)
    elif base == 10.0:
        return math.log10(val)
    return math.log(val) / math.log(base)


def shannon_entropy(probabilities: List[float], base: float = 2.0, eps: float = 1e-15) -> float:
    if not probabilities:
        raise ValueError("Wahrscheinlichkeitsliste darf nicht leer sein.")
    if abs(sum(probabilities) - 1.0) > 1e-4:
        raise ValueError(f"Wahrscheinlichkeiten müssen sich zu 1 aufsummieren.")

    h = 0.0
    for p in probabilities:
        if p > eps:
            h -= p * _log_base(p, base)
    return h


def cross_entropy(p_true: List[float], q_pred: List[float], base: float = 2.0, eps: float = 1e-15) -> float:
    if len(p_true) != len(q_pred):
        raise ValueError("p_true und q_pred müssen dieselbe Länge haben.")
        
    ce = 0.0
    for p, q in zip(p_true, q_pred):
        if p > 0:
            q_clipped = max(eps, min(1.0, q))
            ce -= p * _log_base(q_clipped, base)
    return ce


def kl_divergence(p: List[float], q: List[float], base: float = 2.0, eps: float = 1e-15) -> float:
    if len(p) != len(q):
        raise ValueError("Verteilungen p und q müssen dieselbe Länge haben.")

    d_kl = 0.0
    for pi, qi in zip(p, q):
        if pi > eps:
            qi_clipped = max(eps, qi)
            d_kl += pi * (_log_base(pi, base) - _log_base(qi_clipped, base))
    return d_kl


def kl_divergence_two_gaussians_1d(mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
    if sigma_p <= 0 or sigma_q <= 0:
        raise ValueError("Standardabweichungen müssen positiv sein.")
        
    term1 = math.log(sigma_q / sigma_p)
    term2 = ((sigma_p ** 2) + ((mu_p - mu_q) ** 2)) / (2.0 * (sigma_q ** 2))
    return term1 + term2 - 0.5


if __name__ == "__main__":
    fair_coin = [0.5, 0.5]
    biased_coin = [0.9, 0.1]
    print(f"Entropie fair:   {shannon_entropy(fair_coin):.4f} Bits")
    print(f"Entropie biased: {shannon_entropy(biased_coin):.4f} Bits")

    p = [0.7, 0.3]
    q = [0.5, 0.5]
    h_p = shannon_entropy(p, base=math.e)
    d_kl = kl_divergence(p, q, base=math.e)
    h_cross = cross_entropy(p, q, base=math.e)
    print(f"H(P) = {h_p:.4f}, D_KL = {d_kl:.4f}, H(P, Q) = {h_cross:.4f}")
    print(f"Summe H(P) + D_KL = {h_p + d_kl:.4f}")
