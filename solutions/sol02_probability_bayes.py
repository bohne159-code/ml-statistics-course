"""
LÖSUNG 02: Wahrscheinlichkeitsrechnung & Bayes-Theorem für ML
"""

from typing import Dict, List, Tuple


def bayes_theorem(prior_a: float, likelihood_b_given_a: float, likelihood_b_given_not_a: float) -> float:
    prior_not_a = 1.0 - prior_a
    total_prob_b = (likelihood_b_given_a * prior_a) + (likelihood_b_given_not_a * prior_not_a)
    if total_prob_b == 0:
        raise ZeroDivisionError("Evidence P(B) darf nicht 0 sein.")
    posterior_a_given_b = (likelihood_b_given_a * prior_a) / total_prob_b
    return posterior_a_given_b


def posterior_odds_update(prior_odds: float, bayes_factor: float) -> float:
    return prior_odds * bayes_factor


def odds_to_probability(odds: float) -> float:
    return odds / (1.0 + odds)


def naive_bayes_predict_spam(
    message_words: List[str],
    word_probs_spam: Dict[str, float],
    word_probs_ham: Dict[str, float],
    prior_spam: float = 0.5
) -> Tuple[str, float]:
    unnorm_spam = prior_spam
    unnorm_ham = 1.0 - prior_spam

    for word in message_words:
        w_lower = word.lower()
        # Laplace-Smoothing Ersatz: Default 0.01 bei unbekanntem Wort
        unnorm_spam *= word_probs_spam.get(w_lower, 0.01)
        unnorm_ham *= word_probs_ham.get(w_lower, 0.01)

    total = unnorm_spam + unnorm_ham
    if total == 0:
        p_spam = 0.5
    else:
        p_spam = unnorm_spam / total

    label = "spam" if p_spam >= 0.5 else "ham"
    return label, p_spam


if __name__ == "__main__":
    p_fraud = bayes_theorem(0.01, 0.95, 0.02)
    print(f"P(Betrug | Alarm) = {p_fraud:.4f}")

    # Spam Test
    spam_vocab = {"free": 0.8, "money": 0.7, "click": 0.6, "meeting": 0.05}
    ham_vocab = {"free": 0.1, "money": 0.1, "click": 0.1, "meeting": 0.6}
    msg = ["Free", "money", "click"]
    label, prob = naive_bayes_predict_spam(msg, spam_vocab, ham_vocab)
    print(f"Nachricht {msg} -> {label} (p={prob:.4f})")
