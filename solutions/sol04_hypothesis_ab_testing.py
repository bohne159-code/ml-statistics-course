"""
LÖSUNG 04: Hypothesentests & A/B-Testing für Machine Learning Modelle
"""

import math
from typing import Dict, Tuple


def z_score_proportions(
    successes_a: int, trials_a: int,
    successes_b: int, trials_b: int
) -> float:
    if trials_a <= 0 or trials_b <= 0:
        raise ValueError("Stichprobengrößen müssen positiv sein.")
    
    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    
    p_pool = (successes_a + successes_b) / (trials_a + trials_b)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * ((1.0 / trials_a) + (1.0 / trials_b)))
    
    if se_pool == 0:
        return 0.0
    return (p_b - p_a) / se_pool


def approximate_p_value_two_tailed(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def evaluate_ab_test(
    successes_a: int, trials_a: int,
    successes_b: int, trials_b: int,
    alpha: float = 0.05
) -> Dict[str, object]:
    cr_a = successes_a / trials_a
    cr_b = successes_b / trials_b
    relative_lift = (cr_b - cr_a) / cr_a if cr_a > 0 else 0.0
    
    z = z_score_proportions(successes_a, trials_a, successes_b, trials_b)
    p_val = approximate_p_value_two_tailed(z)
    
    is_sig = p_val < alpha
    decision = "DEPLOY_MODEL_B" if (is_sig and relative_lift > 0) else "KEEP_MODEL_A"
    
    return {
        "cr_a": cr_a,
        "cr_b": cr_b,
        "relative_lift": relative_lift,
        "z_score": z,
        "p_value": p_val,
        "is_significant": is_sig,
        "decision": decision
    }


def estimate_required_sample_size(
    baseline_conversion: float,
    minimum_detectable_effect_relative: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    if not (0 < baseline_conversion < 1):
        raise ValueError("Baseline Conversion muss zwischen 0 und 1 liegen.")
    if minimum_detectable_effect_relative <= 0:
        raise ValueError("Relativer Effekt muss > 0 sein.")
        
    p = baseline_conversion
    delta = baseline_conversion * minimum_detectable_effect_relative
    n = 16.0 * p * (1.0 - p) / (delta ** 2)
    return math.ceil(n)


if __name__ == "__main__":
    res = evaluate_ab_test(1000, 20000, 1150, 20000, alpha=0.05)
    print(f"Modell A CR: {res['cr_a']*100:.2f}% | Modell B CR: {res['cr_b']*100:.2f}%")
    print(f"Relativer Zuwachs: {res['relative_lift']*100:.2f}%")
    print(f"Z-Score: {res['z_score']:.4f} | p-Wert: {res['p_value']:.5f}")
    print(f"Statistisch signifikant? {res['is_significant']} -> Entscheidung: {res['decision']}")
