"""
LÖSUNG 03: Wahrscheinlichkeitsverteilungen & Zentraler Grenzwertsatz (CLT)
"""

import math
import random
from typing import List, Tuple


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma <= 0:
        raise ValueError("Standardabweichung sigma muss positiv sein.")
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * (((x - mu) / sigma) ** 2)
    return coeff * math.exp(exponent)


def standard_error_of_mean(sample_std: float, n: int) -> float:
    if n <= 0:
        raise ValueError("Stichprobengröße n muss > 0 sein.")
    return sample_std / math.sqrt(n)


def bernoulli_pmf(k: int, p: float) -> float:
    if k not in (0, 1):
        raise ValueError("k muss 0 oder 1 sein.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p muss zwischen 0 und 1 liegen.")
    return p if k == 1 else (1.0 - p)


def simulate_clt(
    distribution_type: str,
    sample_size: int,
    num_experiments: int,
    seed: int = 42
) -> List[float]:
    random.seed(seed)
    sample_means = []

    for _ in range(num_experiments):
        if distribution_type == "uniform":
            sample = [random.random() for _ in range(sample_size)]
        elif distribution_type == "exponential":
            sample = [random.expovariate(1.0) for _ in range(sample_size)]
        else:
            raise ValueError(f"Unbekannter Verteilungstyp: {distribution_type}")
        
        sample_means.append(sum(sample) / sample_size)

    return sample_means


def confidence_interval_95(sample_mean: float, sem: float) -> Tuple[float, float]:
    z = 1.96
    lower = sample_mean - z * sem
    upper = sample_mean + z * sem
    return lower, upper


if __name__ == "__main__":
    pdf_0 = normal_pdf(0.0, mu=0.0, sigma=1.0)
    print(f"Normal PDF bei x=0: {pdf_0:.4f}")
    means = simulate_clt("exponential", sample_size=30, num_experiments=1000)
    mean_of_means = sum(means) / len(means)
    print(f"Mittelwert der 1000 Stichprobenmittel: {mean_of_means:.4f}")
