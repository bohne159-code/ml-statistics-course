"""
ÜBUNG 03: Wahrscheinlichkeitsverteilungen & Zentraler Grenzwertsatz (CLT)
========================================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
1. Normalverteilung (Gauß):
   - Gewichtsinitialisierung in Neuronalen Netzen (He / Xavier Initialization)
   - Latenter Raum in VAEs (Variational Autoencoders: z ~ N(0, I))
   - Diffusion Models: Fügen schrittweise Gaußsches Rauschen hinzu und lernen die Umkehrung
2. Zentraler Grenzwertsatz (Central Limit Theorem - CLT):
   - Egal wie schief oder seltsam die Ausgangsdaten verteilt sind: Der Durchschnitt von N unabhängigen
     Zufallsvariablen nähert sich mit wachsendem N einer Normalverteilung an!
   - Das ist das Fundament für Konfidenzintervalle, Batch Normalization und A/B-Tests.
3. Standardfehler (Standard Error of the Mean - SEM):
   - SEM = sigma / sqrt(n). Erklärt, warum größere Datensätze oder größere Minibatches
     die Schätzvarianz verringern!

Deine Aufgabe:
--------------
Implementiere die Dichtefunktionen (PDF), berechne den Standardfehler und simuliere den CLT.
"""

import math
import random
from typing import List, Tuple


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Berechnet die Wahrscheinlichkeitsdichte (PDF) der Gaußschen Normalverteilung an der Stelle x.
    Formel:
      f(x) = (1 / (sigma * sqrt(2 * pi))) * exp( -0.5 * ((x - mu) / sigma)^2 )
    """
    if sigma <= 0:
        raise ValueError("Standardabweichung sigma muss positiv sein.")
    # TODO: Implementiere die Gaußsche Dichtefunktion
    raise NotImplementedError("Bitte implementieren!")


def standard_error_of_mean(sample_std: float, n: int) -> float:
    """
    Berechnet den Standardfehler des Mittelwerts:
    SEM = sample_std / sqrt(n)
    """
    if n <= 0:
        raise ValueError("Stichprobengröße n muss > 0 sein.")
    # TODO: Berechne den Standardfehler
    raise NotImplementedError("Bitte implementieren!")


def bernoulli_pmf(k: int, p: float) -> float:
    """
    Wahrscheinlichkeitsfunktion (PMF) der Bernoulli-Verteilung:
    P(K = 1) = p
    P(K = 0) = 1 - p
    k muss 0 oder 1 sein. p muss im Intervall [0, 1] liegen.
    """
    if k not in (0, 1):
        raise ValueError("k muss 0 oder 1 sein.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p muss zwischen 0 und 1 liegen.")
    # TODO: Gib p für k=1 und (1-p) für k=0 zurück
    raise NotImplementedError("Bitte implementieren!")


def simulate_clt(
    distribution_type: str,
    sample_size: int,
    num_experiments: int,
    seed: int = 42
) -> List[float]:
    """
    Simuliert den Zentralen Grenzwertsatz:
    1. Ziehe in jedem Experiment 'sample_size' Werte aus einer stark NICHT-normalen Verteilung:
       - 'uniform': Gleichverteilung im Bereich [0, 1] (Mittelwert 0.5)
       - 'exponential': Exponentialverteilung mit lambda=1.0 (sehr schief!)
    2. Berechne den Mittelwert dieser Stichprobe.
    3. Wiederhole das 'num_experiments' mal.
    4. Rückgabe: Liste aller berechneten Stichproben-Mittelwerte.
       (Diese Liste nähert sich nach CLT einer Glockenkurve an!)
    """
    random.seed(seed)
    sample_means = []

    # TODO: Für jedes Experiment:
    #   Ziehe 'sample_size' Werte entsprechend 'distribution_type'
    #   (nutze random.random() für 'uniform', random.expovariate(1.0) für 'exponential')
    #   Berechne den Mittelwert dieser Stichprobe und hänge ihn an sample_means an.
    
    raise NotImplementedError("Bitte implementieren!")


def confidence_interval_95(sample_mean: float, sem: float) -> Tuple[float, float]:
    """
    Berechnet das klassische 95%-Konfidenzintervall unter Annahme einer Normalverteilung.
    Der Z-Wert für 95% Konfidenz (2-seitig) beträgt 1.96.
    Intervall = [mean - 1.96 * SEM, mean + 1.96 * SEM]
    """
    # TODO: Berechne untere und obere Grenze
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 03 - Verteilungen & CLT")
    print("=" * 60)

    # 1. Normal PDF
    try:
        pdf_0 = normal_pdf(0.0, mu=0.0, sigma=1.0)
        print(f"  [OK]   normal_pdf:               f(0) = {pdf_0:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] normal_pdf:               {e}")
    except Exception as e:
        print(f"  [FAIL] normal_pdf:               {e}")

    # 2. SEM
    try:
        sem = standard_error_of_mean(10.0, 100)
        print(f"  [OK]   standard_error_of_mean:   SEM = {sem:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] standard_error_of_mean:   {e}")
    except Exception as e:
        print(f"  [FAIL] standard_error_of_mean:   {e}")

    # 3. Bernoulli PMF
    try:
        p1 = bernoulli_pmf(1, 0.7)
        print(f"  [OK]   bernoulli_pmf:            P(K=1) = {p1:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] bernoulli_pmf:            {e}")
    except Exception as e:
        print(f"  [FAIL] bernoulli_pmf:            {e}")

    # 4. CLT Simulation
    try:
        means = simulate_clt("exponential", sample_size=30, num_experiments=500)
        print(f"  [OK]   simulate_clt:             {len(means)} Stichproben generiert (Mittelwert: {sum(means)/len(means):.3f})")
    except NotImplementedError as e:
        print(f"  [TODO] simulate_clt:             {e}")
    except Exception as e:
        print(f"  [FAIL] simulate_clt:             {e}")

    # 5. Confidence Interval 95
    try:
        ci_l, ci_u = confidence_interval_95(100.0, 2.0)
        print(f"  [OK]   confidence_interval_95:   [{ci_l:.2f}, {ci_u:.2f}]")
    except NotImplementedError as e:
        print(f"  [TODO] confidence_interval_95:   {e}")
    except Exception as e:
        print(f"  [FAIL] confidence_interval_95:   {e}")
    print("=" * 60)
