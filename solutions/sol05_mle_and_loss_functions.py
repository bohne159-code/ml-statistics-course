"""
LÖSUNG 05: Maximum Likelihood Estimation (MLE) & Loss-Funktionen
"""

import math
from typing import List, Tuple


def estimate_normal_mle(data: List[float]) -> Tuple[float, float]:
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    n = len(data)
    mu_hat = sum(data) / n
    sigma_sq_hat = sum((x - mu_hat) ** 2 for x in data) / n
    sigma_hat = math.sqrt(sigma_sq_hat)
    return mu_hat, sigma_hat


def bernoulli_log_likelihood(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true und y_prob müssen dieselbe Länge haben.")
    
    total_ll = 0.0
    for y, p in zip(y_true, y_prob):
        p_clipped = max(eps, min(1.0 - eps, p))
        total_ll += y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped)
    return total_ll


def binary_cross_entropy_loss(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:
    ll = bernoulli_log_likelihood(y_true, y_prob, eps=eps)
    return - (1.0 / len(y_true)) * ll


def gaussian_nll_regression(
    y_true: List[float],
    y_pred: List[float],
    sigma: float = 1.0
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("Listen müssen dieselbe Länge haben.")
    if sigma <= 0:
        raise ValueError("sigma muss > 0 sein.")

    n = len(y_true)
    term1 = 0.5 * n * math.log(2.0 * math.pi * (sigma ** 2))
    sse = sum((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred))
    term2 = sse / (2.0 * (sigma ** 2))
    return term1 + term2


if __name__ == "__main__":
    data = [2.1, 2.5, 1.9, 2.3, 2.2]
    mu_mle, sig_mle = estimate_normal_mle(data)
    print(f"MLE Parameter: mu={mu_mle:.3f}, sigma={sig_mle:.3f}")

    y_true = [1, 0, 1, 1]
    y_pred = [0.9, 0.1, 0.8, 0.95]
    ll = bernoulli_log_likelihood(y_true, y_pred)
    bce = binary_cross_entropy_loss(y_true, y_pred)
    print(f"Bernoulli Log-Likelihood: {ll:.4f}")
    print(f"Binary Cross Entropy Loss: {bce:.4f}")
