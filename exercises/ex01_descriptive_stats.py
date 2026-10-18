"""
ÜBUNG 01: Deskriptive Statistik & Daten-Exploration für ML
===========================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Bevor ein Modell trainiert wird, musst du deine Daten verstehen (Exploratory Data Analysis - EDA):
- Mittelwert & Median: Erkennen von Schiefe (Skewness). Wenn Mean >> Median, gibt es starke Ausreißer nach oben.
- Varianz & Standardabweichung: Maß für die Streuung von Features. Feature Scaling (z.B. Z-Score Normalization: (x - mu) / sigma)
  ist essenziell für Gradient Descent, k-NN, SVMs und neuronale Netze.
- Kovarianz & Korrelation: Erkennen von Multikollinearität (zwei Features enthalten dieselbe Information) und
  Feature Selection (starke Korrelation mit dem Target).
- Ausreißer-Erkennung (IQR): Schützt Loss Functions wie MSE (L2), die quadratisch auf Ausreißer reagieren.

Deine Aufgabe:
--------------
Implementiere die folgenden Funktionen mit nativem Python (ohne externe Bibliotheken wie numpy/pandas).
Dadurch verstehst du die Mathematik hinter den Standard-Funktionen wirklich im Detail.
"""

import math
from typing import List, Tuple

def calculate_mean(data: List[float]) -> float:
    """
    Berechnet das arithmetische Mittel (Mean) der Daten.
    Formel: mu = (1 / n) * sum(x_i)
    """
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    return  (1/len(data)) * sum(data)



def calculate_median(data: List[float]) -> float:
    """
    Berechnet den Median (Zentralwert).
    Bei ungerader Anzahl n: das mittlere Element der sortierten Daten.
    Bei gerader Anzahl n: der Durchschnitt der beiden mittleren Elemente.
    """
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    if len(data) % 2 == 1:
        return sorted(data)[len(data) // 2]
    else:
        mid1 = sorted(data)[len(data) // 2 - 1]
        mid2 = sorted(data)[len(data) // 2]
        return (mid1 + mid2) / 2
    raise NotImplementedError("Bitte implementieren!")


def calculate_variance(data: List[float], sample: bool = True) -> float:
    """
    Berechnet die Varianz (Streuung um den Mittelwert).
    Formel:
      - Stichproben-Varianz (sample=True, Bessel-Korrektur): s^2 = sum((x_i - mean)^2) / (n - 1)
      - Populations-Varianz (sample=False): sigma^2 = sum((x_i - mean)^2) / n
    Hinweis: Bei sample=True und n < 2 soll ein ValueError geworfen werden.
    """
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    if sample and len(data) < 2:
        raise ValueError("Stichproben-Varianz erfordert mindestens 2 Datenpunkte.")
    
    # TODO: Berechne den Mittelwert und anschließend die Varianz (sample oder population)
    raise NotImplementedError("Bitte implementieren!")


def calculate_std_dev(data: List[float], sample: bool = True) -> float:
    """
    Berechnet die Standardabweichung: sigma = sqrt(Varianz).
    Gleiche Einheit wie die Originaldaten (im Gegensatz zur Varianz, die quadrierten Einheiten hat).
    """
    # TODO: Nutze calculate_variance und math.sqrt
    raise NotImplementedError("Bitte implementieren!")


def calculate_covariance(x: List[float], y: List[float], sample: bool = True) -> float:
    """
    Berechnet die Kovarianz zwischen zwei Merkmalen x und y.
    Formel (Stichprobe): Cov(x, y) = sum((x_i - mean_x) * (y_i - mean_y)) / (n - 1)
    Formel (Population): Cov(x, y) = sum((x_i - mean_x) * (y_i - mean_y)) / n
    Beide Listen müssen die gleiche Länge n haben (n >= 2 bei sample=True).
    """
    if len(x) != len(y):
        raise ValueError("Listen x und y müssen die gleiche Länge haben.")
    if not x:
        raise ValueError("Listen dürfen nicht leer sein.")
    if sample and len(x) < 2:
        raise ValueError("Stichproben-Kovarianz erfordert mindestens 2 Datenpunkte.")

    # TODO: Berechne die Kovarianz
    raise NotImplementedError("Bitte implementieren!")


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Berechnet den Pearson-Korrelationskoeffizienten r (-1 bis +1).
    Formel: r = Cov(x, y) / (std_dev(x) * std_dev(y))
    Normalisiert die Kovarianz unabhängig von den Skalen der Features.
    """
    # TODO: Berechne r. Achte darauf, dass keine Division durch 0 auftritt, falls std == 0!
    raise NotImplementedError("Bitte implementieren!")


def detect_outliers_iqr(data: List[float], factor: float = 1.5) -> Tuple[List[float], Tuple[float, float]]:
    """
    Findet Ausreißer mithilfe der IQR-Methode (Interquartilsabstand).
    1. Sortiere die Daten.
    2. Berechne Q1 (25. Perzentil) und Q3 (75. Perzentil).
       (Vereinfachter Ansatz: Q1 = Median der unteren Hälfte, Q3 = Median der oberen Hälfte)
    3. IQR = Q3 - Q1
    4. Untere Grenze = Q1 - factor * IQR
       Obere Grenze = Q3 + factor * IQR
    5. Rückgabe: (Liste aller Werte außerhalb der Grenzen, (untere_grenze, obere_grenze))
    """
    if len(data) < 4:
        raise ValueError("Mindestens 4 Datenpunkte für IQR benötigt.")

    # TODO: Berechne Q1, Q3, IQR, Grenzen und filtere die Ausreißer
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 01 - Deskriptive Statistik")
    print("=" * 60)
    beispiel_daten = [10.0, 12.0, 12.0, 13.0, 12.0, 14.0, 25.0]
    
    # 1. Mean
    try:
        m = calculate_mean(beispiel_daten)
        print(f"  [OK]   calculate_mean:           Mittelwert = {m:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] calculate_mean:           {e}")
    except Exception as e:
        print(f"  [FAIL] calculate_mean:           {e}")

    # 2. Median
    try:
        med = calculate_median(beispiel_daten)
        print(f"  [OK]   calculate_median:         Median = {med:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] calculate_median:         {e}")
    except Exception as e:
        print(f"  [FAIL] calculate_median:         {e}")

    # 3. Variance
    try:
        var = calculate_variance(beispiel_daten, sample=True)
        print(f"  [OK]   calculate_variance:       Varianz = {var:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] calculate_variance:       {e}")
    except Exception as e:
        print(f"  [FAIL] calculate_variance:       {e}")

    # 4. Std Dev
    try:
        std = calculate_std_dev(beispiel_daten, sample=True)
        print(f"  [OK]   calculate_std_dev:        Std-Abw = {std:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] calculate_std_dev:        {e}")
    except Exception as e:
        print(f"  [FAIL] calculate_std_dev:        {e}")

    # 5. Covariance
    try:
        cov = calculate_covariance(beispiel_daten, beispiel_daten, sample=True)
        print(f"  [OK]   calculate_covariance:     Kovarianz = {cov:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] calculate_covariance:     {e}")
    except Exception as e:
        print(f"  [FAIL] calculate_covariance:     {e}")

    # 6. Correlation
    try:
        r = calculate_pearson_correlation(beispiel_daten, beispiel_daten)
        print(f"  [OK]   calculate_pearson_corr:   r = {r:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] calculate_pearson_corr:   {e}")
    except Exception as e:
        print(f"  [FAIL] calculate_pearson_corr:   {e}")

    # 7. Outliers IQR
    try:
        outliers, (lb, ub) = detect_outliers_iqr(beispiel_daten)
        print(f"  [OK]   detect_outliers_iqr:      Ausreisser = {outliers} (Grenzen: {lb:.1f} bis {ub:.1f})")
    except NotImplementedError as e:
        print(f"  [TODO] detect_outliers_iqr:      {e}")
    except Exception as e:
        print(f"  [FAIL] detect_outliers_iqr:      {e}")
    print("=" * 60)
