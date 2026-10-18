"""
ÜBUNG 10: Data Drift & Monitoring in Produktion (MLOps-Statistik)
================================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Ein Machine-Learning-Modell verschlechtert sich in Produktion fast immer über die Zeit.
Der Grund: Die reale Welt verändert sich, während das Modell auf historischen Daten trainiert wurde.
1. Covariate Shift (Data Drift):
   - Die Verteilung der Eingabe-Features P(X) ändert sich (z.B. durch neue Nutzergruppen, verändertes Nutzerverhalten).
2. Concept Drift:
   - Der Zusammenhang zwischen Features und Zielgröße P(Y | X) ändert sich (z.B. neue Betrugsmaschen bei Kreditkarten).
3. Population Stability Index (PSI):
   - Der weltweite Industrie-Standard (Finanzwesen, E-Commerce, Tech) zur Erkennung von Merkmals-Drift.
   - Basiert auf diskretisierter relativer Entropie (Symmetrisierte KL-Divergenz):
     PSI = sum_i ( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
   - Daumenregel:
     * PSI < 0.10: Kein signifikanter Drift (Modell stabil).
     * 0.10 <= PSI <= 0.25: Moderater Drift (Monitoring verstärken).
     * PSI > 0.25: Kritischer Drift -> Alarm auslösen & Retraining anstoßen!
4. Kolmogorov-Smirnov (KS) Statistik:
   - Misst den maximalen vertikalen Abstand zwischen zwei empirischen Verteilungsfunktionen (ECDFs):
     D = max_x | F_baseline(x) - F_production(x) |

Deine Aufgabe:
--------------
Implementiere die ECDF-Berechnung, die KS-Statistik und den Population Stability Index (PSI).
"""

import math
from typing import List, Tuple


def compute_ecdf(data: List[float], eval_points: List[float]) -> List[float]:
    """
    Berechnet die Empirische Kumulative Verteilungsfunktion (ECDF) an gegebenen Auswertungspunkten.
    
    Für jeden Punkt x in eval_points:
      F(x) = (Anzahl der Werte in data <= x) / len(data)
      
    Rückgabe: Liste der ECDF-Werte zwischen 0.0 und 1.0.
    """
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    n = len(data)
    sorted_data = sorted(data)
    
    # TODO: Berechne F(x) für jedes x in eval_points
    raise NotImplementedError("Bitte implementieren!")


def compute_ks_statistic(baseline_data: List[float], production_data: List[float]) -> float:
    """
    Berechnet die 2-Stichproben-Kolmogorov-Smirnov-Statistik D:
      D = max | F_baseline(x) - F_production(x) |
      
    Als Auswertungspunkte x dient die Vereinigung aller Werte aus beiden Stichproben.
    Rückgabe: Maximaler Abstand D im Bereich [0.0, 1.0].
    """
    if not baseline_data or not production_data:
        raise ValueError("Datenlisten dürfen nicht leer sein.")

    # TODO: Ermittle alle eindeutigen Punkte beider Datensätze, berechne deren ECDFs und finde das Maximum der absoluten Differenz
    raise NotImplementedError("Bitte implementieren!")


def compute_psi(
    expected_proportions: List[float],
    actual_proportions: List[float],
    eps: float = 1e-6
) -> Tuple[float, str]:
    """
    Berechnet den Population Stability Index (PSI) zwischen zwei diskreten Verteilungen
    (z.B. Trainings-Baseline vs. Produktions-Woche):
    
      PSI = sum_i ( (Actual_i - Expected_i) * ln(Actual_i / Expected_i) )
      
    Parameter:
      - expected_proportions: Relative Häufigkeiten der Bins in Baseline (Summe = 1.0)
      - actual_proportions: Relative Häufigkeiten der Bins in Produktion (Summe = 1.0)
      - eps: Minimalwert gegen Division durch 0 oder log(0)
      
    Rückgabe:
      (psi_wert, status_meldung)
      Status:
        - "NO_DRIFT" wenn PSI < 0.10
        - "MODERATE_DRIFT" wenn 0.10 <= PSI <= 0.25
        - "SIGNIFICANT_DRIFT" wenn PSI > 0.25
    """
    if len(expected_proportions) != len(actual_proportions):
        raise ValueError("Beide Listen müssen die gleiche Anzahl an Bins haben.")

    # TODO: Berechne den PSI-Wert und bestimme die Status-Meldung
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 10 - Data Drift & MLOps Monitoring")
    print("=" * 60)
    baseline = [30000.0, 35000.0, 40000.0, 45000.0, 50000.0]
    prod_drift = [45000.0, 50000.0, 60000.0, 70000.0, 85000.0]

    # 1. ECDF
    try:
        ecdf = compute_ecdf(baseline, [35000.0, 45000.0])
        print(f"  [OK]   compute_ecdf:             ECDF = {ecdf}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_ecdf:             {e}")
    except Exception as e:
        print(f"  [FAIL] compute_ecdf:             {e}")

    # 2. KS Statistic
    try:
        ks = compute_ks_statistic(baseline, prod_drift)
        print(f"  [OK]   compute_ks_statistic:     D = {ks:.3f}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_ks_statistic:     {e}")
    except Exception as e:
        print(f"  [FAIL] compute_ks_statistic:     {e}")

    # 3. PSI
    try:
        p_train = [0.4, 0.4, 0.2]
        p_prod = [0.1, 0.3, 0.6]
        psi_val, status = compute_psi(p_train, p_prod)
        print(f"  [OK]   compute_psi:              PSI = {psi_val:.4f} -> {status}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_psi:              {e}")
    except Exception as e:
        print(f"  [FAIL] compute_psi:              {e}")
    print("=" * 60)
