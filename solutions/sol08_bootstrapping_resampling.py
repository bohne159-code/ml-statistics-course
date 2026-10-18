"""
LÖSUNG 08: Bootstrapping, Resampling & Ensemble-Statistik
"""

import math
import random
from typing import Callable, List, Tuple


def bootstrap_sample(data: List[float], seed: int = None) -> List[float]:
    if seed is not None:
        random.seed(seed)
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    n = len(data)
    return [random.choice(data) for _ in range(n)]


def bootstrap_confidence_interval(
    data: List[float],
    statistic_func: Callable[[List[float]], float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level muss zwischen 0 und 1 liegen.")
    random.seed(seed)

    boot_stats = []
    for _ in range(num_bootstraps):
        sample = bootstrap_sample(data)
        val = statistic_func(sample)
        boot_stats.append(val)

    boot_stats.sort()
    
    alpha = 1.0 - confidence_level
    lower_idx = int((alpha / 2.0) * num_bootstraps)
    upper_idx = int((1.0 - alpha / 2.0) * num_bootstraps)
    
    # Sicherstellen, dass Indizes im Array liegen
    lower_idx = max(0, min(num_bootstraps - 1, lower_idx))
    upper_idx = max(0, min(num_bootstraps - 1, upper_idx))
    
    return boot_stats[lower_idx], boot_stats[upper_idx]


def calculate_ensemble_variance(
    base_variance: float,
    correlation_rho: float,
    num_models_m: int
) -> float:
    if num_models_m < 1:
        raise ValueError("num_models_m muss mindestens 1 sein.")
    if not (0.0 <= correlation_rho <= 1.0):
        raise ValueError("correlation_rho muss im Intervall [0, 1] liegen.")

    return correlation_rho * base_variance + ((1.0 - correlation_rho) / num_models_m) * base_variance


if __name__ == "__main__":
    data = [12.0, 15.0, 14.0, 16.0, 35.0, 18.0, 19.0, 22.0]
    sample = bootstrap_sample(data, seed=42)
    median_func = lambda x: sorted(x)[len(x)//2]
    ci_low, ci_high = bootstrap_confidence_interval(data, median_func, num_bootstraps=500)
    print(f"95% Bootstrap CI für Median: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"Ensemble Varianz (sigma=10, rho=0.2, M=50 Bäume): {calculate_ensemble_variance(10.0, 0.2, 50):.3f}")
