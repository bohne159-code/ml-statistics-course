"""
ÜBUNG 07: Statistische Modell-Evaluation & Metriken für ML
==========================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
Ein Modell mit 99% Genauigkeit (Accuracy) kann in der Praxis völlig nutzlos sein, wenn z.B. nur 0.5% der Daten
positiv sind (Kreditkartenbetrug, Krebserkennung).
Ein ML-Engineer muss daher mit statistischen Evaluierungsmetriken sicher umgehen können:
1. Konfusionsmatrix:
   - TP (True Positive), FP (False Positive), TN (True Negative), FN (False Negative)
2. Precision & Recall:
   - Precision = TP / (TP + FP)  -> "Wie viele unserer Alarme sind echte Fälle?"
   - Recall = TP / (TP + FN)     -> "Wie viele der echten Fälle haben wir gefunden?"
   - F1-Score = 2 * (Precision * Recall) / (Precision + Recall) (Harmonisches Mittel!)
3. ROC-Kurve & AUC (Area Under Curve):
   - ROC plottet TPR (Recall) gegen FPR (False Positive Rate = FP / (FP + TN)) über alle Schwellenwerte.
   - AUC hat eine wunderschöne statistische Interpretation:
     Es ist die Wahrscheinlichkeit, dass ein zufällig ausgewählter positiver Datenpunkt
     einen höheren vorhergesagten Score erhält als ein zufällig ausgewählter negativer Datenpunkt!
4. Brier Score:
   - Misst die Kalibrierung von Wahrscheinlichkeiten: Brier = (1/N) * sum((p_i - y_i)^2).
   - Ein Modell, das 80% Konfidenz ausgibt, sollte in 80% der Fälle recht haben.

Deine Aufgabe:
--------------
Implementiere die Metriken und die Berechnung des ROC-AUC-Wertes.
"""

from typing import Dict, List, Tuple


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, int]:
    """
    Berechnet TP, FP, TN, FN aus den binären Ground-Truth- und Vorhersage-Labels (0 oder 1).
    Rückgabe: Dictionary mit Schlüsseln 'TP', 'FP', 'TN', 'FN'.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true und y_pred müssen dieselbe Länge haben.")
    
    # TODO: Zähle TP, FP, TN, FN
    raise NotImplementedError("Bitte implementieren!")


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    """
    Berechnet Precision, Recall und den F1-Score.
    Falls der Nenner 0 ist, soll der Wert als 0.0 zurückgegeben werden.
    """
    # TODO: Berechne precision, recall und f1
    raise NotImplementedError("Bitte implementieren!")


def compute_roc_auc_score(
    y_true: List[int],
    y_scores: List[float]
) -> float:
    """
    Berechnet den ROC-AUC Score (Area Under the Receiver Operating Characteristic Curve).
    
    Elegante statistische Berechnung (Wilcoxon-Mann-Whitney-U-Statistik):
    1. Sammle alle Scores der positiven Beispiele (y == 1) in Liste P.
    2. Sammle alle Scores der negativen Beispiele (y == 0) in Liste N.
    3. Zähle für jedes Paar (p_score, n_score):
       - Wenn p_score > n_score: +1.0
       - Wenn p_score == n_score (Gleichstand): +0.5
       - Wenn p_score < n_score: 0.0
    4. Teile durch (|P| * |N|).
    
    Rückgabe: AUC-Wert zwischen 0.0 und 1.0 (0.5 = Zufallsraten, 1.0 = Perfekt).
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true und y_scores müssen dieselbe Länge haben.")
        
    # TODO: Trenne positive und negative Scores und berechne die paarweise Mann-Whitney-Statistik
    raise NotImplementedError("Bitte implementieren!")


def compute_brier_score(
    y_true: List[int],
    y_probs: List[float]
) -> float:
    """
    Berechnet den Brier-Score (Mean Squared Error auf Wahrscheinlichkeiten):
      Brier = (1 / n) * sum((p_i - y_i)^2)
    Niedriger ist besser (0 = perfekt kalibriert und treffsicher).
    """
    if len(y_true) != len(y_probs):
        raise ValueError("Längen müssen übereinstimmen.")
    # TODO: Berechne den Brier Score
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 07 - Statistische ML-Evaluation & Metriken")
    print("=" * 60)
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_scores = [0.95, 0.80, 0.40, 0.15, 0.70, 0.60, 0.85, 0.20]
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    # 1. Confusion Matrix
    try:
        cm = compute_confusion_matrix(y_true, y_pred)
        print(f"  [OK]   compute_confusion_matrix: {cm}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_confusion_matrix: {e}")
    except Exception as e:
        print(f"  [FAIL] compute_confusion_matrix: {e}")

    # 2. Precision Recall F1
    try:
        p, r, f1 = compute_precision_recall_f1(3, 1, 1)
        print(f"  [OK]   compute_precision_recall: Prec={p:.2f}, Rec={r:.2f}, F1={f1:.2f}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_precision_recall: {e}")
    except Exception as e:
        print(f"  [FAIL] compute_precision_recall: {e}")

    # 3. ROC AUC
    try:
        auc = compute_roc_auc_score(y_true, y_scores)
        print(f"  [OK]   compute_roc_auc_score:    AUC = {auc:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_roc_auc_score:    {e}")
    except Exception as e:
        print(f"  [FAIL] compute_roc_auc_score:    {e}")

    # 4. Brier Score
    try:
        brier = compute_brier_score(y_true, y_scores)
        print(f"  [OK]   compute_brier_score:      Brier = {brier:.4f}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_brier_score:      {e}")
    except Exception as e:
        print(f"  [FAIL] compute_brier_score:      {e}")
    print("=" * 60)
