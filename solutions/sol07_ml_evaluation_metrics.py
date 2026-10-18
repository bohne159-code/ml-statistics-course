"""
LÖSUNG 07: Statistische Modell-Evaluation & Metriken für ML
"""

from typing import Dict, List, Tuple


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true und y_pred müssen dieselbe Länge haben.")
    
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
            
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def compute_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return precision, recall, f1


def compute_roc_auc_score(
    y_true: List[int],
    y_scores: List[float]
) -> float:
    if len(y_true) != len(y_scores):
        raise ValueError("y_true und y_scores müssen dieselbe Länge haben.")
        
    pos_scores = [score for yt, score in zip(y_true, y_scores) if yt == 1]
    neg_scores = [score for yt, score in zip(y_true, y_scores) if yt == 0]
    
    if not pos_scores or not neg_scores:
        raise ValueError("y_true muss sowohl positive als auch negative Klassen enthalten.")
        
    sum_ranks = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                sum_ranks += 1.0
            elif p == n:
                sum_ranks += 0.5
                
    return sum_ranks / (len(pos_scores) * len(neg_scores))


def compute_brier_score(
    y_true: List[int],
    y_probs: List[float]
) -> float:
    if len(y_true) != len(y_probs):
        raise ValueError("Längen müssen übereinstimmen.")
    n = len(y_true)
    return sum((p - y) ** 2 for y, p in zip(y_true, y_probs)) / n


if __name__ == "__main__":
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    y_scores = [0.95, 0.80, 0.40, 0.15, 0.70, 0.60, 0.85, 0.20]
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
    
    cm = compute_confusion_matrix(y_true, y_pred)
    p, r, f1 = compute_precision_recall_f1(cm['TP'], cm['FP'], cm['FN'])
    auc = compute_roc_auc_score(y_true, y_scores)
    brier = compute_brier_score(y_true, y_scores)
    
    print(f"Konfusionsmatrix: {cm}")
    print(f"Precision: {p:.3f} | Recall: {r:.3f} | F1-Score: {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(f"Brier-Score: {brier:.4f}")
