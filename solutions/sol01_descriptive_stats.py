"""
LÖSUNG 01: Deskriptive Statistik & Daten-Exploration für ML
"""

import math
from typing import List, Tuple


def calculate_mean(data: List[float]) -> float:
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    return sum(data) / len(data)


def calculate_median(data: List[float]) -> float:
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    else:
        return (s[mid - 1] + s[mid]) / 2.0


def calculate_variance(data: List[float], sample: bool = True) -> float:
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    n = len(data)
    if sample and n < 2:
        raise ValueError("Stichproben-Varianz erfordert mindestens 2 Datenpunkte.")
    
    mean_val = calculate_mean(data)
    sq_diff_sum = sum((x - mean_val) ** 2 for x in data)
    divisor = (n - 1) if sample else n
    return sq_diff_sum / divisor


def calculate_std_dev(data: List[float], sample: bool = True) -> float:
    return math.sqrt(calculate_variance(data, sample=sample))


def calculate_covariance(x: List[float], y: List[float], sample: bool = True) -> float:
    if len(x) != len(y):
        raise ValueError("Listen x und y müssen die gleiche Länge haben.")
    if not x:
        raise ValueError("Listen dürfen nicht leer sein.")
    n = len(x)
    if sample and n < 2:
        raise ValueError("Stichproben-Kovarianz erfordert mindestens 2 Datenpunkte.")

    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)
    cov_sum = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    divisor = (n - 1) if sample else n
    return cov_sum / divisor


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    std_x = calculate_std_dev(x, sample=True)
    std_y = calculate_std_dev(y, sample=True)
    if std_x == 0.0 or std_y == 0.0:
        raise ValueError("Korrelation undefiniert, wenn die Standardabweichung 0 ist.")
    cov_xy = calculate_covariance(x, y, sample=True)
    return cov_xy / (std_x * std_y)


def detect_outliers_iqr(data: List[float], factor: float = 1.5) -> Tuple[List[float], Tuple[float, float]]:
    if len(data) < 4:
        raise ValueError("Mindestens 4 Datenpunkte für IQR benötigt.")
    
    s = sorted(data)
    n = len(s)
    mid = n // 2

    # Untere und obere Hälfte bestimmen
    if n % 2 == 0:
        lower_half = s[:mid]
        upper_half = s[mid:]
    else:
        lower_half = s[:mid]
        upper_half = s[mid + 1:]

    q1 = calculate_median(lower_half)
    q3 = calculate_median(upper_half)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers, (lower_bound, upper_bound)


if __name__ == "__main__":
    beispiel_daten = [10.0, 12.0, 12.0, 13.0, 12.0, 14.0, 25.0]
    print(f"Mittelwert: {calculate_mean(beispiel_daten):.2f}")
    print(f"Median:     {calculate_median(beispiel_daten):.2f}")
    print(f"Varianz:    {calculate_variance(beispiel_daten, sample=True):.2f}")
    print(f"Std-Abw:    {calculate_std_dev(beispiel_daten, sample=True):.2f}")
    outliers, (lb, ub) = detect_outliers_iqr(beispiel_daten)
    print(f"Ausreißer:  {outliers} (Grenzen: {lb:.2f} bis {ub:.2f})")
