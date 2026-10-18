"""
ÜBUNG 08: Bootstrapping, Resampling & Ensemble-Statistik
=========================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
1. Nicht-parametrische Konfidenzintervalle:
   In der Praxis sind Metriken wie ROC-AUC, F1-Score oder Modell-Latenzen fast nie normalverteilt.
   Mit Bootstrapping (Ziehen mit Zurücklegen) kannst du für JEDE beliebige Metrik exakte Konfidenzintervalle
   berechnen, ohne eine Normalverteilungsannahme treffen zu müssen!
2. Bagging (Bootstrap Aggregating):
   Das statistische Fundament von Random Forests! Durch das Trainieren von M Modellen auf M Bootstrap-Stichproben
   und anschließendes Mitteln der Vorhersagen wird die Modellvarianz drastisch gesenkt.
   Formel für die Ensemble-Varianz:
     Var(Ensemble) = rho * sigma^2 + ((1 - rho) / M) * sigma^2
     (wobei rho die Korrelation zwischen den Modellen und M die Anzahl der Bäume ist).
3. Out-of-Bag (OOB) Error:
   Statistisch gesehen landen bei N Ziehungen mit Zurücklegen aus N Datenpunkten ca. (1 - 1/e) ≈ 63.2%
   der Datenpunkte im Bootstrap-Sample. Die restlichen 36.8% dienen als kostenloser Validierungsdatensatz!

Deine Aufgabe:
--------------
Implementiere Bootstrapping, Konfidenzintervall-Schätzung und die Ensemble-Varianzreduktion.
"""

import math
import random
from typing import Callable, List, Tuple


def bootstrap_sample(data: List[float], seed: int = None) -> List[float]:
    """
    Zieht eine Bootstrap-Stichprobe der gleichen Länge wie 'data' mit Zurücklegen (with replacement).
    """
    if seed is not None:
        random.seed(seed)
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    n = len(data)
    # TODO: Ziehe n Elemente zufällig aus data (mit Zurücklegen)
    raise NotImplementedError("Bitte implementieren!")


def bootstrap_confidence_interval(
    data: List[float],
    statistic_func: Callable[[List[float]], float],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Berechnet das nicht-parametrische Konfidenzintervall einer beliebigen statistischen Kennzahl
    (z.B. Median, 90. Perzentil, Mean) mittels Perzentil-Bootstrap-Methode:
    
    1. Wiederhole 'num_bootstraps' mal:
       a) Erzeuge eine Bootstrap-Stichprobe aus 'data'.
       b) Berechne die Kennzahl mittels statistic_func(sample).
       c) Speichere das Ergebnis in einer Liste.
    2. Sortiere die Liste der berechneten Kennzahlen.
    3. Alpha = 1 - confidence_level (z.B. 0.05 für 95% Konfidenz).
    4. Untere Grenze = Wert am Perzentil (alpha / 2) (z.B. 2.5. Perzentil).
       Obere Grenze = Wert am Perzentil (1 - alpha / 2) (z.B. 97.5. Perzentil).
       
    Rückgabe: (untere_grenze, obere_grenze)
    """
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level muss zwischen 0 und 1 liegen.")
    random.seed(seed)

    # TODO: Führe num_bootstraps durch, berechne jeweils statistic_func und ermittle die Perzentile
    raise NotImplementedError("Bitte implementieren!")


def calculate_ensemble_variance(
    base_variance: float,
    correlation_rho: float,
    num_models_m: int
) -> float:
    """
    Berechnet die theoretische Varianz eines Bagging-Ensembles (z.B. Random Forest mit M Bäumen):
    
      Var(Ensemble) = rho * sigma^2 + ((1 - rho) / M) * sigma^2
      
    Parameter:
      - base_variance (sigma^2): Varianz eines einzelnen unkorrelierten Basis-Modells
      - correlation_rho (rho): Paarweise Korrelation zwischen den Modellen (0 <= rho <= 1)
      - num_models_m (M): Anzahl der Modelle im Ensemble (M >= 1)
      
    Erkenntnis für ML:
      Wenn M gegen unendlich geht, sinkt die Varianz auf rho * sigma^2.
      Deshalb verwendet Random Forest 'Feature Bagging' (Zufallsauswahl von Merkmalen pro Split),
      um die Korrelation rho zwischen den Bäumen aktiv zu senken!
    """
    if num_models_m < 1:
        raise ValueError("num_models_m muss mindestens 1 sein.")
    if not (0.0 <= correlation_rho <= 1.0):
        raise ValueError("correlation_rho muss im Intervall [0, 1] liegen.")

    # TODO: Berechne und gib die Ensemble-Varianz zurück
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 08 - Bootstrapping & Resampling")
    print("=" * 60)
    data = [12.0, 15.0, 14.0, 16.0, 35.0, 18.0, 19.0, 22.0]

    # 1. Bootstrap Sample
    try:
        sample = bootstrap_sample(data, seed=42)
        print(f"  [OK]   bootstrap_sample:         Länge = {len(sample)}: {sample[:4]}...")
    except NotImplementedError as e:
        print(f"  [TODO] bootstrap_sample:         {e}")
    except Exception as e:
        print(f"  [FAIL] bootstrap_sample:         {e}")

    # 2. Bootstrap CI
    try:
        median_func = lambda x: sorted(x)[len(x)//2]
        ci_l, ci_u = bootstrap_confidence_interval(data, median_func, num_bootstraps=200, seed=42)
        print(f"  [OK]   bootstrap_ci:             95% CI = [{ci_l:.2f}, {ci_u:.2f}]")
    except NotImplementedError as e:
        print(f"  [TODO] bootstrap_ci:             {e}")
    except Exception as e:
        print(f"  [FAIL] bootstrap_ci:             {e}")

    # 3. Ensemble Variance
    try:
        v_ens = calculate_ensemble_variance(10.0, 0.2, 50)
        print(f"  [OK]   ensemble_variance:        Var(Ensemble) = {v_ens:.3f}")
    except NotImplementedError as e:
        print(f"  [TODO] ensemble_variance:        {e}")
    except Exception as e:
        print(f"  [FAIL] ensemble_variance:        {e}")
    print("=" * 60)
