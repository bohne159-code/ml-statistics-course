"""
ÜBUNG 02: Wahrscheinlichkeitsrechnung & Bayes-Theorem für ML
=============================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Das Bayes-Theorem ist die theoretische Grundlage moderner KI-Methoden:
1. Bayes-Klassifikator & Naive Bayes: Textklassifikation, Spamfilterung.
2. Bayesian Deep Learning & Bayesian Optimization: Schätzung von Modell-Unsicherheit (Uncertainty Estimation)
   und automatische Hyperparameter-Optimierung.
3. Base-Rate Fallacy (Prävalenzfehler): Warum ein Modell mit 99% Genauigkeit bei unausgeglichenen Daten
   (Imbalanced Data, z.B. Betrugserkennung) trotzdem die meisten Alarme falsch auslöst!

Formel (Satz von Bayes):
    P(A | B) = [ P(B | A) * P(A) ] / P(B)
    Wobei P(B) nach dem Satz der totalen Wahrscheinlichkeit zerlegt werden kann:
    P(B) = P(B | A) * P(A) + P(B | not A) * P(not A)

    In der ML-Sprache:
    Posterior = (Likelihood * Prior) / Evidence

Deine Aufgabe:
--------------
Implementiere die mathematischen Bausteine und baue einen einfachen Naive-Bayes-Spam-Klassifikator.
"""

from typing import Dict, List, Tuple


def bayes_theorem(prior_a: float, likelihood_b_given_a: float, likelihood_b_given_not_a: float) -> float:
    """
    Berechnet P(A | B) mithilfe des Satzes von Bayes und der totalen Wahrscheinlichkeit.
    
    Parameter:
      - prior_a: P(A), die A-priori-Wahrscheinlichkeit für Ereignis A (z.B. 0.01 für 1% Krankheitsrate)
      - likelihood_b_given_a: P(B | A), Wahrscheinlichkeit für Beobachtung B gegeben A (Sensitivität/True Positive Rate)
      - likelihood_b_given_not_a: P(B | not A), Wahrscheinlichkeit für B gegeben NICHT A (False Positive Rate)
      
    Rückgabe:
      - P(A | B), die A-posteriori-Wahrscheinlichkeit (Posterior).
    """
    # TODO: Berechne P(not A)
    # TODO: Berechne die totale Wahrscheinlichkeit P(B) = P(B|A)*P(A) + P(B|not A)*P(not A)
    # TODO: Berechne Posterior P(A|B)
    raise NotImplementedError("Bitte implementieren!")


def posterior_odds_update(prior_odds: float, bayes_factor: float) -> float:
    """
    Berechnet die neuen Quoten (Posterior Odds) nach Erhalt von Evidenz:
    Posterior Odds = Prior Odds * Bayes Factor (Likelihood Ratio)
    Wobei Odds = p / (1 - p)
    """
    # TODO: Berechne die Posterior Odds
    raise NotImplementedError("Bitte implementieren!")


def odds_to_probability(odds: float) -> float:
    """
    Wandelt Odds o in eine Wahrscheinlichkeit p um:
    p = odds / (1 + odds)
    """
    # TODO: Berechne p
    raise NotImplementedError("Bitte implementieren!")


def naive_bayes_predict_spam(
    message_words: List[str],
    word_probs_spam: Dict[str, float],
    word_probs_ham: Dict[str, float],
    prior_spam: float = 0.5
) -> Tuple[str, float]:
    """
    Ein einfacher Naive-Bayes-Textklassifikator (Spam vs. Ham).
    
    Annahme bedingter Unabhängigkeit (Naive Bayes):
    P(Wörter | Spam) = Produkt aller P(word_i | Spam)
    P(Wörter | Ham)  = Produkt aller P(word_i | Ham)
    
    Um numerische Unterläufe (Underflow) bei Multiplikation vieler kleiner Zahlen
    zu vermeiden, addiert man in der Praxis oft Logarithmen:
      log P(Spam | Wörter) ~ log P(Spam) + sum(log P(w_i | Spam))
    
    Für diese einfache Übung berechnen wir die unnormierten Verbund-Wahrscheinlichkeiten:
      unnorm_spam = prior_spam * Produkt(word_probs_spam.get(w, 0.01))
      unnorm_ham  = (1 - prior_spam) * Produkt(word_probs_ham.get(w, 0.01))
      
    Normalisierte Wahrscheinlichkeit:
      p_spam_posterior = unnorm_spam / (unnorm_spam + unnorm_ham)
      
    Rückgabe:
      ("spam" wenn p_spam_posterior >= 0.5 sonst "ham", p_spam_posterior)
    """
    # TODO: Berechne unnorm_spam und unnorm_ham
    # TODO: Normalisiere und gib ("spam"|"ham", p_spam_posterior) zurück
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 02 - Wahrscheinlichkeitsrechnung & Bayes")
    print("=" * 60)
    
    # 1. Bayes Theorem
    try:
        p_fraud = bayes_theorem(0.01, 0.95, 0.02)
        print(f"  [OK]   bayes_theorem:            P(Betrug|Alarm) = {p_fraud:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] bayes_theorem:            {e}")
    except Exception as e:
        print(f"  [FAIL] bayes_theorem:            {e}")

    # 2. Posterior Odds Update
    try:
        odds = posterior_odds_update(1.0, 2.0)
        print(f"  [OK]   posterior_odds_update:    Odds = {odds:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] posterior_odds_update:    {e}")
    except Exception as e:
        print(f"  [FAIL] posterior_odds_update:    {e}")

    # 3. Odds to Probability
    try:
        prob = odds_to_probability(3.0)
        print(f"  [OK]   odds_to_probability:      p = {prob:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] odds_to_probability:      {e}")
    except Exception as e:
        print(f"  [FAIL] odds_to_probability:      {e}")

    # 4. Naive Bayes Spam
    try:
        spam_v = {"win": 0.9, "cash": 0.8}
        ham_v = {"win": 0.05, "cash": 0.05}
        lbl, p_spam = naive_bayes_predict_spam(["win", "cash"], spam_v, ham_v)
        print(f"  [OK]   naive_bayes_predict_spam: Label = '{lbl}' (p = {p_spam:.4f})")
    except NotImplementedError as e:
        print(f"  [TODO] naive_bayes_predict_spam: {e}")
    except Exception as e:
        print(f"  [FAIL] naive_bayes_predict_spam: {e}")
    print("=" * 60)
