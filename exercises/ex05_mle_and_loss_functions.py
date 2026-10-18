"""
ÜBUNG 05: Maximum Likelihood Estimation (MLE) & Loss-Funktionen
==============================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Warum nutzen wir eigentlich 'Mean Squared Error' (MSE) bei Regression und 'Cross-Entropy' bei Klassifikation?
Die Antwort lautet: Es sind direkte mathematische Ableitungen der Maximum Likelihood Estimation (MLE)!

1. Normalverteiltes Rauschen -> MSE:
   Wenn wir annehmen, dass die Zielvariable y um die Modellvorhersage f(x) mit normalverteiltem Rauschen
   epsilon ~ N(0, sigma^2) streut, führt das Maximieren der Likelihood exakt zur Minimierung von MSE!
   
2. Bernoulli-Verteilung -> Binary Cross-Entropy (Log-Loss):
   Wenn y in {0, 1} liegt und das Modell die Wahrscheinlichkeit p = P(y=1|x) ausgibt,
   ist der Negative Log-Likelihood (NLL) exakt die Binary Cross-Entropy:
     L = - [ y * log(p) + (1 - y) * log(1 - p) ]

3. MAP (Maximum A Posteriori) & Regularisierung:
   - Fügen wir einen Gaußschen Prior N(0, tau^2) auf die Modellgewichte hinzu, erhalten wir L2-Regularisierung (Ridge)!
   - Fügen wir einen Laplace-Prior hinzu, erhalten wir L1-Regularisierung (Lasso)!

Deine Aufgabe:
--------------
Implementiere die Log-Likelihood-Berechnungen und erkenne die Brücke zu modernen Deep-Learning Loss Functions.
"""

import math
from typing import List, Tuple


def estimate_normal_mle(data: List[float]) -> Tuple[float, float]:
    """
    Berechnet die analytischen Maximum-Likelihood-Schätzer für eine Normalverteilung N(mu, sigma^2):
      mu_hat = (1 / n) * sum(x_i)
      sigma_sq_hat = (1 / n) * sum((x_i - mu_hat)^2)  <-- Wichtig: Divisor n (verzerrter MLE), nicht n - 1!
      sigma_hat = sqrt(sigma_sq_hat)
      
    Rückgabe:
      (mu_hat, sigma_hat)
    """
    if not data:
        raise ValueError("Datenliste darf nicht leer sein.")
    # TODO: Berechne mu_hat und sigma_hat (mit Divisor n)
    raise NotImplementedError("Bitte implementieren!")


def bernoulli_log_likelihood(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:
    """
    Berechnet die Log-Likelihood für binäre Beobachtungen:
      LL = sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )
      
    Um log(0) (Undefined) zu verhindern, clippe p_i in den Bereich [eps, 1 - eps].
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true und y_prob müssen dieselbe Länge haben.")
    
    # TODO: Clippe p_i und berechne die Summe der Log-Likelihoods
    raise NotImplementedError("Bitte implementieren!")


def binary_cross_entropy_loss(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:
    """
    Berechnet die Binary Cross-Entropy (BCE) Loss:
      BCE = - (1 / n) * Log-Likelihood
      
    In ML minimieren Optimierer (wie Adam / SGD) diesen Verlust.
    """
    # TODO: Nutze bernoulli_log_likelihood
    raise NotImplementedError("Bitte implementieren!")


def gaussian_nll_regression(
    y_true: List[float],
    y_pred: List[float],
    sigma: float = 1.0
) -> float:
    """
    Berechnet den Negative Log-Likelihood (NLL) für Regressionsvorhersagen unter Annahme
    eines konstanten Gaußschen Rauschens sigma:
    
    NLL = (n / 2) * log(2 * pi * sigma^2) + (1 / (2 * sigma^2)) * sum((y_i - y_hat_i)^2)
    
    Beachte: Der zweite Term ist exakt proportional zum Sum of Squared Errors (SSE)!
    Daher ist das Minimieren des MSE identisch mit dem Maximieren der Likelihood.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Listen müssen dieselbe Länge haben.")
    if sigma <= 0:
        raise ValueError("sigma muss > 0 sein.")

    # TODO: Berechne den Gaußschen NLL
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 05 - MLE & Loss-Funktionen")
    print("=" * 60)

    # 1. MLE Normal
    try:
        mu_m, sig_m = estimate_normal_mle([2.1, 2.5, 1.9, 2.3, 2.2])
        print(f"  [OK]   estimate_normal_mle:      mu={mu_m:.3f}, sigma={sig_m:.3f}")
    except NotImplementedError as e:
        print(f"  [TODO] estimate_normal_mle:      {e}")
    except Exception as e:
        print(f"  [FAIL] estimate_normal_mle:      {e}")

    # 2. Bernoulli Log-Likelihood
    try:
        ll = bernoulli_log_likelihood([1, 0, 1], [0.9, 0.1, 0.8])
        print(f"  [OK]   bernoulli_log_likelihood: LL = {ll:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] bernoulli_log_likelihood: {e}")
    except Exception as e:
        print(f"  [FAIL] bernoulli_log_likelihood: {e}")

    # 3. BCE Loss
    try:
        bce = binary_cross_entropy_loss([1, 0, 1], [0.9, 0.1, 0.8])
        print(f"  [OK]   binary_cross_entropy:     BCE = {bce:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] binary_cross_entropy:     {e}")
    except Exception as e:
        print(f"  [FAIL] binary_cross_entropy:     {e}")

    # 4. Gaussian NLL
    try:
        nll = gaussian_nll_regression([2.0, 3.0], [2.1, 2.9], sigma=1.0)
        print(f"  [OK]   gaussian_nll_regression:  NLL = {nll:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] gaussian_nll_regression:  {e}")
    except Exception as e:
        print(f"  [FAIL] gaussian_nll_regression:  {e}")
    print("=" * 60)
