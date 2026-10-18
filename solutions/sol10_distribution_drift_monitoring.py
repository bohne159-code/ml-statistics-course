"""
LÖSUNG 10: Data Drift & Monitoring in Produktion (MLOps-Statistik)
"""

import math
from typing import List, Tuple


def compute_ecdf(data: List[float], eval_points: List[float]) -> List[float]:
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    n = len(data)
    ecdf_vals = []
    for x in eval_points:
        count = sum(1 for val in data if val <= x)
        ecdf_vals.append(count / n)
    return ecdf_vals


def compute_ks_statistic(baseline_data: List[float], production_data: List[float]) -> float:
    if not baseline_data or not production_data:
        raise ValueError("Datenlisten dürfen nicht leer sein.")

    all_points = sorted(list(set(baseline_data + production_data)))
    ecdf_base = compute_ecdf(baseline_data, all_points)
    ecdf_prod = compute_ecdf(production_data, all_points)

    max_diff = max(abs(b - p) for b, p in zip(ecdf_base, ecdf_prod))
    return max_diff


def compute_psi(
    expected_proportions: List[float],
    actual_proportions: List[float],
    eps: float = 1e-6
) -> Tuple[float, str]:
    if len(expected_proportions) != len(actual_proportions):
        raise ValueError("Beide Listen müssen die gleiche Anzahl an Bins haben.")

    psi = 0.0
    for exp_p, act_p in zip(expected_proportions, actual_proportions):
        e_safe = max(eps, exp_p)
        a_safe = max(eps, act_p)
        psi += (a_safe - e_safe) * math.log(a_safe / e_safe)

    if psi < 0.10:
        status = "NO_DRIFT"
    elif psi <= 0.25:
        status = "MODERATE_DRIFT"
    else:
        status = "SIGNIFICANT_DRIFT"

    return psi, status


if __name__ == "__main__":
    baseline = [30000.0, 35000.0, 40000.0, 45000.0, 50000.0]
    prod_drift = [45000.0, 50000.0, 60000.0, 70000.0, 85000.0]
    ks = compute_ks_statistic(baseline, prod_drift)
    print(f"KS D-Statistik: {ks:.3f}")

    p_train = [0.4, 0.4, 0.2]
    p_prod = [0.1, 0.3, 0.6]
    psi_val, status = compute_psi(p_train, p_prod)
    print(f"PSI: {psi_val:.4f} -> {status}")
