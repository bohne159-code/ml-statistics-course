"""
ÜBUNG 06: Informationstheorie für Machine Learning & Deep Learning
===================================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Informationstheorie ist die moderne mathematische Sprache von Deep Learning, LLMs und generativer KI:
1. Shannon-Entropie H(P):
   - Maß für die Unordnung / Unsicherheit einer Verteilung.
   - Kriterium für Decision Trees (Random Forests, XGBoost) via Information Gain (Informationsgewinn).
2. Cross-Entropy H(P, Q):
   - Der Standard-Loss für Klassifikation und Large Language Models (LLMs wie GPT, Gemini, LLaMA).
3. Kullback-Leibler (KL) Divergenz D_KL(P || Q):
   - Misst den "Abstand" (relative Entropie) zwischen zwei Wahrscheinlichkeitsverteilungen.
   - VAEs (Variational Autoencoders): Zwingt den latenten Raum, der Standardnormalverteilung zu ähneln.
   - RLHF (Reinforcement Learning from Human Feedback für LLMs): Verhindert durch einen KL-Penalty,
     dass das ausgerichtete Modell zu weit vom Ausgangs-Modell abweicht (Policy Drift).
   - Wichtige Eigenschaft: D_KL(P || Q) >= 0 (Gibbs'sche Ungleichung) und D_KL(P || Q) != D_KL(Q || P) (nicht symmetrisch!).
4. Der fundamentale Zusammenhang:
     H(P, Q) = H(P) + D_KL(P || Q)
     Da H(P) (die Entropie der echten Labels) konstant ist, minimiert das Minimieren der Cross-Entropy
     direkt die KL-Divergenz zwischen Vorhersage und Realität!

Deine Aufgabe:
--------------
Implementiere Entropie, Cross-Entropie, KL-Divergenz und teste die Identität.
"""

import math
from typing import List


def shannon_entropy(probabilities: List[float], base: float = 2.0, eps: float = 1e-15) -> float:
    """
    Berechnet die Shannon-Entropie einer diskreten Verteilung P:
      H(P) = - sum( p_i * log_base(p_i) ) für p_i > 0
    Bei base=2 ist die Einheit 'Bits' (oder Shannons), bei base=e ist es 'Nats'.
    """
    if not probabilities:
        raise ValueError("Wahrscheinlichkeitsliste darf nicht leer sein.")
    
    # Prüfe ob Summe der Wahrscheinlichkeiten ca. 1 ist
    if abs(sum(probabilities) - 1.0) > 1e-4:
        raise ValueError(f"Wahrscheinlichkeiten müssen sich zu 1 aufsummieren (Summe={sum(probabilities)}).")

    # TODO: Berechne H(P)
    raise NotImplementedError("Bitte implementieren!")


def cross_entropy(p_true: List[float], q_pred: List[float], base: float = 2.0, eps: float = 1e-15) -> float:
    """
    Berechnet die Cross-Entropie zwischen der wahren Verteilung P und der Modellvorhersage Q:
      H(P, Q) = - sum( p_i * log_base(q_i) )
    Clippe q_i mit eps, um log(0) zu verhindern.
    """
    if len(p_true) != len(q_pred):
        raise ValueError("p_true und q_pred müssen dieselbe Länge haben.")
        
    # TODO: Berechne die Cross-Entropie
    raise NotImplementedError("Bitte implementieren!")


def kl_divergence(p: List[float], q: List[float], base: float = 2.0, eps: float = 1e-15) -> float:
    """
    Berechnet die Kullback-Leibler (KL) Divergenz D_KL(P || Q):
      D_KL(P || Q) = sum( p_i * log_base(p_i / q_i) ) für p_i > 0
      
    Alternativ über Logarithmengesetze:
      D_KL(P || Q) = sum( p_i * (log_base(p_i) - log_base(q_i)) )
    """
    if len(p) != len(q):
        raise ValueError("Verteilungen p und q müssen dieselbe Länge haben.")

    # TODO: Berechne D_KL(P || Q)
    raise NotImplementedError("Bitte implementieren!")


def kl_divergence_two_gaussians_1d(mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
    """
    Berechnet die geschlossene analytische Formel der KL-Divergenz zwischen zwei 1D-Normalverteilungen
    P = N(mu_p, sigma_p^2) und Q = N(mu_q, sigma_q^2) in Nats (Basis e):
    
      D_KL(P || Q) = log(sigma_q / sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2 * sigma_q^2) - 0.5
      
    Dies ist exakt die Formel, die in jedem VAE (Variational Autoencoder) für den Latent Loss genutzt wird
    (mit Q = N(0, 1))!
    """
    if sigma_p <= 0 or sigma_q <= 0:
        raise ValueError("Standardabweichungen müssen positiv sein.")
        
    # TODO: Berechne die geschlossene analytische KL-Divergenz
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 06 - Informationstheorie")
    print("=" * 60)

    # 1. Shannon Entropy
    try:
        h = shannon_entropy([0.5, 0.5], base=2.0)
        print(f"  [OK]   shannon_entropy:          H(fair) = {h:.4f} Bits")
    except NotImplementedError as e:
        print(f"  [TODO] shannon_entropy:          {e}")
    except Exception as e:
        print(f"  [FAIL] shannon_entropy:          {e}")

    # 2. Cross Entropy
    try:
        ce = cross_entropy([0.7, 0.3], [0.5, 0.5], base=2.0)
        print(f"  [OK]   cross_entropy:            H(P, Q) = {ce:.4f} Bits")
    except NotImplementedError as e:
        print(f"  [TODO] cross_entropy:            {e}")
    except Exception as e:
        print(f"  [FAIL] cross_entropy:            {e}")

    # 3. KL Divergence
    try:
        kl = kl_divergence([0.7, 0.3], [0.5, 0.5], base=2.0)
        print(f"  [OK]   kl_divergence:            D_KL(P||Q) = {kl:.4f} Bits")
    except NotImplementedError as e:
        print(f"  [TODO] kl_divergence:            {e}")
    except Exception as e:
        print(f"  [FAIL] kl_divergence:            {e}")

    # 4. KL Gaussians
    try:
        kl_g = kl_divergence_two_gaussians_1d(0.0, 1.0, 0.0, 1.0)
        print(f"  [OK]   kl_two_gaussians_1d:      D_KL = {kl_g:.4f} Nats")
    except NotImplementedError as e:
        print(f"  [TODO] kl_two_gaussians_1d:      {e}")
    except Exception as e:
        print(f"  [FAIL] kl_two_gaussians_1d:      {e}")
    print("=" * 60)
