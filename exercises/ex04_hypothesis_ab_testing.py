"""
ÜBUNG 04: Hypothesentests & A/B-Testing für Machine Learning Modelle
===================================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Als ML-Engineer trainierst du ein neues Modell (Treatment B), das offline bessere Metriken liefert als das
aktuelle Produktionsmodell (Control A). Doch erst der Online A/B-Test entscheidet über den Rollout:
- Bringt das neue Modell eine echte Verbesserung (z.B. +1.5% Klickrate) oder ist das reines Zufallsrauschen?
- Type I Error (Alpha, Signifikanzniveau, meist 5%): Wahrscheinlichkeit, dass wir fälschlicherweise behaupten,
  Modell B sei besser, obwohl in Wahrheit kein Unterschied existiert.
- p-Wert: Die Wahrscheinlichkeit, ein mindestens so extremes Ergebnis wie das beobachtete zu erhalten,
  unter der Annahme, dass die Nullhypothese H0 (kein Unterschied) wahr ist. Wenn p < alpha -> H0 ablehnen!
- Statistische Power (1 - Beta, meist 80%): Die Fähigkeit des Experiments, einen echten Effekt auch tatsächlich
  als signifikant nachzuweisen.
- Sample Size Calculation: Wie viele Tausend Nutzer müssen wir testen, um einen Effekt von z.B. 1% nachweisen zu können?

Deine Aufgabe:
--------------
Implementiere den Zwei-Stichproben-Z-Test für Konversionsraten und eine Stichprobengrößen-Abschätzung.
"""

import math
from typing import Dict, Tuple


def z_score_proportions(
    successes_a: int, trials_a: int,
    successes_b: int, trials_b: int
) -> float:
    """
    Berechnet den Z-Score für den Vergleich zweier Anteile (z.B. Click-Through-Rate von Modell A vs. Modell B).
    
    1. Berechne Konversionsraten:
       p_a = successes_a / trials_a
       p_b = successes_b / trials_b
    2. Berechne gepoolten Anteil (Pooled Proportion):
       p_pool = (successes_a + successes_b) / (trials_a + trials_b)
    3. Berechne den gepoolten Standardfehler:
       se_pool = sqrt( p_pool * (1 - p_pool) * (1/trials_a + 1/trials_b) )
    4. Z-Score:
       z = (p_b - p_a) / se_pool
    """
    if trials_a <= 0 or trials_b <= 0:
        raise ValueError("Stichprobengrößen müssen positiv sein.")
    # TODO: Berechne p_a, p_b, p_pool, se_pool und den Z-Score
    raise NotImplementedError("Bitte implementieren!")


def approximate_p_value_two_tailed(z: float) -> float:
    """
    Berechnet den zweiseitigen p-Wert aus dem Z-Score.
    Mithilfe der komplementären Fehlerfunktion math.erfc:
      p = erfc( |z| / sqrt(2) )
    """
    # TODO: Berechne den p-Wert
    raise NotImplementedError("Bitte implementieren!")


def evaluate_ab_test(
    successes_a: int, trials_a: int,
    successes_b: int, trials_b: int,
    alpha: float = 0.05
) -> Dict[str, object]:
    """
    Führt einen vollständigen A/B-Test aus und liefert eine strukturierte Entscheidung:
    - cr_a: Conversion Rate Gruppe A
    - cr_b: Conversion Rate Gruppe B
    - relative_lift: (cr_b - cr_a) / cr_a
    - z_score: Z-Wert
    - p_value: p-Wert
    - is_significant: True wenn p_value < alpha
    - decision: 'DEPLOY_MODEL_B' wenn signifikant und lift > 0,
                'KEEP_MODEL_A' wenn nicht signifikant oder schlechter
    """
    # TODO: Nutze z_score_proportions und approximate_p_value_two_tailed
    raise NotImplementedError("Bitte implementieren!")


def estimate_required_sample_size(
    baseline_conversion: float,
    minimum_detectable_effect_relative: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Schätzt die benötigte Stichprobengröße PRO GRUPPE für einen A/B-Test.
    
    Faustformel von Evan Miller für zwei Anteile (alpha=0.05, power=0.80):
      n_per_variant ≈ 16 * p * (1 - p) / delta^2
      wobei:
        p = baseline_conversion
        delta = baseline_conversion * minimum_detectable_effect_relative (absoluter Unterschied)
        
    Beispiel:
      baseline = 0.10 (10% Klickrate), relativer Lift = 0.10 (10% Steigerung -> delta = 0.01)
      n ≈ 16 * 0.10 * 0.90 / (0.01^2) = 1.44 / 0.0001 = 14.400 Nutzer pro Variante!
    """
    if not (0 < baseline_conversion < 1):
        raise ValueError("Baseline Conversion muss zwischen 0 und 1 liegen.")
    if minimum_detectable_effect_relative <= 0:
        raise ValueError("Relativer Effekt muss > 0 sein.")
        
    # TODO: Berechne delta und schätze n (aufrunden auf die nächste Ganzzahl mit math.ceil)
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 04 - Hypothesentests & A/B-Testing")
    print("=" * 60)

    # 1. Z-Score Proportions
    try:
        z = z_score_proportions(1000, 20000, 1150, 20000)
        print(f"  [OK]   z_score_proportions:      z = {z:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] z_score_proportions:      {e}")
    except Exception as e:
        print(f"  [FAIL] z_score_proportions:      {e}")

    # 2. Approximate p-value
    try:
        p_val = approximate_p_value_two_tailed(2.0)
        print(f"  [OK]   approximate_p_value:      p = {p_val:.5f}")
    except NotImplementedError as e:
        print(f"  [TODO] approximate_p_value:      {e}")
    except Exception as e:
        print(f"  [FAIL] approximate_p_value:      {e}")

    # 3. Evaluate A/B Test
    try:
        res = evaluate_ab_test(1000, 20000, 1150, 20000, alpha=0.05)
        print(f"  [OK]   evaluate_ab_test:         Entscheidung = {res['decision']} (p={res['p_value']:.4f})")
    except NotImplementedError as e:
        print(f"  [TODO] evaluate_ab_test:         {e}")
    except Exception as e:
        print(f"  [FAIL] evaluate_ab_test:         {e}")

    # 4. Sample Size
    try:
        n_req = estimate_required_sample_size(0.05, 0.10)
        print(f"  [OK]   estimate_sample_size:     n = {n_req:,} Nutzer pro Gruppe")
    except NotImplementedError as e:
        print(f"  [TODO] estimate_sample_size:     {e}")
    except Exception as e:
        print(f"  [FAIL] estimate_sample_size:     {e}")
    print("=" * 60)
