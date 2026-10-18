"""
Generator für das VOLLSTÄNDIGE, ungekürzte Jupyter Notebook (Google Colab).
Enthält alle 10 Module und alle 42 einzelnen Aufgaben mit isolierten Test-Zellen!
"""

import json
import os
import re

MODULE_SECTIONS = [
    {
        "id": 1,
        "title": "Modul 01: Deskriptive Statistik, EDA & Outlier Detection",
        "svg": "01_boxplot_outliers.svg",
        "description": "Lage- und Streuungsmaße, Schiefe, Kovarianz, Pearson-Korrelation und die IQR-Methode zur Ausreißer-Erkennung.",
        "file": "exercises/ex01_descriptive_stats.py",
        "functions": [
            ("calculate_mean", """
d = [10.0, 20.0, 30.0, 40.0, 50.0]
assert math.isclose(calculate_mean(d), 30.0)
assert math.isclose(calculate_mean([5.0]), 5.0)
print("✅ calculate_mean erfolgreich bestanden!")
"""),
            ("calculate_median", """
assert math.isclose(calculate_median([10.0, 20.0, 30.0, 40.0, 50.0]), 30.0)
assert math.isclose(calculate_median([1.0, 2.0, 3.0, 4.0]), 2.5)
assert math.isclose(calculate_median([5.0]), 5.0)
print("✅ calculate_median erfolgreich bestanden!")
"""),
            ("calculate_variance", """
d2 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
assert math.isclose(calculate_variance(d2, sample=True), 32.0 / 7.0, rel_tol=1e-4)
assert math.isclose(calculate_variance(d2, sample=False), 4.0)
print("✅ calculate_variance erfolgreich bestanden!")
"""),
            ("calculate_std_dev", """
d2 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
assert math.isclose(calculate_std_dev(d2, sample=False), 2.0)
assert math.isclose(calculate_std_dev(d2, sample=True), math.sqrt(32.0 / 7.0), rel_tol=1e-4)
print("✅ calculate_std_dev erfolgreich bestanden!")
"""),
            ("calculate_covariance", """
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 10.0]
assert math.isclose(calculate_covariance(x, y, sample=True), 5.0)
assert math.isclose(calculate_covariance(x, y, sample=False), 4.0)
print("✅ calculate_covariance erfolgreich bestanden!")
"""),
            ("calculate_pearson_correlation", """
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 10.0]
assert math.isclose(calculate_pearson_correlation(x, y), 1.0, rel_tol=1e-5)
y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
assert math.isclose(calculate_pearson_correlation(x, y_neg), -1.0, rel_tol=1e-5)
print("✅ calculate_pearson_correlation erfolgreich bestanden!")
"""),
            ("detect_outliers_iqr", """
d_outliers = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 100.0]
outliers, (lb, ub) = detect_outliers_iqr(d_outliers)
assert 100.0 in outliers
assert len(outliers) == 1
assert lb < 1.0 and ub < 100.0
print("✅ detect_outliers_iqr erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 2,
        "title": "Modul 02: Wahrscheinlichkeitsrechnung & Bayes-Theorem",
        "svg": "02_bayes_base_rate.svg",
        "description": "Bedingte Wahrscheinlichkeit, Satz von Bayes, Base Rate Fallacy und Naive-Bayes-Spamfilterung.",
        "file": "exercises/ex02_probability_bayes.py",
        "functions": [
            ("bayes_theorem", """
p = bayes_theorem(0.001, 0.99, 0.05)
expected = (0.99 * 0.001) / ((0.99 * 0.001) + (0.05 * 0.999))
assert math.isclose(p, expected, rel_tol=1e-3)
print("✅ bayes_theorem erfolgreich bestanden!")
"""),
            ("posterior_odds_update", """
assert math.isclose(posterior_odds_update(1.0, 2.0), 2.0)
assert math.isclose(posterior_odds_update(0.5, 3.0), 1.5)
print("✅ posterior_odds_update erfolgreich bestanden!")
"""),
            ("odds_to_probability", """
assert math.isclose(odds_to_probability(3.0), 0.75)
assert math.isclose(odds_to_probability(1.0), 0.5)
print("✅ odds_to_probability erfolgreich bestanden!")
"""),
            ("naive_bayes_predict_spam", """
spam_vocab = {"win": 0.9, "cash": 0.8}
ham_vocab = {"win": 0.05, "cash": 0.05}
label, prob = naive_bayes_predict_spam(["win", "cash"], spam_vocab, ham_vocab, prior_spam=0.5)
assert label == "spam"
assert prob > 0.95
print("✅ naive_bayes_predict_spam erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 3,
        "title": "Modul 03: Verteilungen & Zentraler Grenzwertsatz (CLT)",
        "svg": "03_normal_distribution_clt.svg",
        "description": "Gaußsche Normalverteilung, Bernoulli-PMF, Standardfehler (SEM), CLT-Simulation und 95%-Konfidenzintervalle.",
        "file": "exercises/ex03_distributions_clt.py",
        "functions": [
            ("normal_pdf", """
pdf_val = normal_pdf(0.0, mu=0.0, sigma=1.0)
assert math.isclose(pdf_val, 1.0 / math.sqrt(2 * math.pi), rel_tol=1e-4)
print("✅ normal_pdf erfolgreich bestanden!")
"""),
            ("standard_error_of_mean", """
assert math.isclose(standard_error_of_mean(10.0, 100), 1.0)
assert math.isclose(standard_error_of_mean(4.0, 16), 1.0)
print("✅ standard_error_of_mean erfolgreich bestanden!")
"""),
            ("bernoulli_pmf", """
assert math.isclose(bernoulli_pmf(1, 0.7), 0.7)
assert math.isclose(bernoulli_pmf(0, 0.7), 0.3)
print("✅ bernoulli_pmf erfolgreich bestanden!")
"""),
            ("simulate_clt", """
means = simulate_clt("exponential", sample_size=40, num_experiments=500, seed=123)
avg_mean = sum(means) / len(means)
assert math.isclose(avg_mean, 1.0, rel_tol=0.08)
assert len(means) == 500
print("✅ simulate_clt erfolgreich bestanden!")
"""),
            ("confidence_interval_95", """
low, high = confidence_interval_95(100.0, 2.0)
assert math.isclose(low, 100.0 - 1.96 * 2.0)
assert math.isclose(high, 100.0 + 1.96 * 2.0)
print("✅ confidence_interval_95 erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 4,
        "title": "Modul 04: Hypothesentests & A/B-Testing für ML",
        "svg": "04_hypothesis_testing_pvalue.svg",
        "description": "Z-Test für Conversion Rates, zweiseitiger p-Wert, A/B-Test Entscheidungslogik und Sample Size Sizing.",
        "file": "exercises/ex04_hypothesis_ab_testing.py",
        "functions": [
            ("z_score_proportions", """
z = z_score_proportions(100, 1000, 120, 1000)
assert z > 0
print("✅ z_score_proportions erfolgreich bestanden!")
"""),
            ("approximate_p_value_two_tailed", """
assert math.isclose(approximate_p_value_two_tailed(0.0), 1.0)
assert math.isclose(approximate_p_value_two_tailed(1.96), 0.05, rel_tol=0.05)
print("✅ approximate_p_value_two_tailed erfolgreich bestanden!")
"""),
            ("evaluate_ab_test", """
res = evaluate_ab_test(1000, 20000, 1200, 20000, alpha=0.05)
assert res["is_significant"] is True
assert res["decision"] == "DEPLOY_MODEL_B"
res_null = evaluate_ab_test(1000, 20000, 1005, 20000, alpha=0.05)
assert res_null["is_significant"] is False
assert res_null["decision"] == "KEEP_MODEL_A"
print("✅ evaluate_ab_test erfolgreich bestanden!")
"""),
            ("estimate_required_sample_size", """
n = estimate_required_sample_size(0.05, 0.10)
assert n == 30400
print("✅ estimate_required_sample_size erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 5,
        "title": "Modul 05: Maximum Likelihood Estimation (MLE) & Loss-Funktionen",
        "svg": "05_mle_vs_loss.svg",
        "description": "Analytischer Normalverteilungs-MLE, Bernoulli Log-Likelihood, Herleitung von Binary Cross-Entropy und Gaussian NLL.",
        "file": "exercises/ex05_mle_and_loss_functions.py",
        "functions": [
            ("estimate_normal_mle", """
data = [1.0, 2.0, 3.0, 4.0, 5.0]
mu, sigma = estimate_normal_mle(data)
assert math.isclose(mu, 3.0)
assert math.isclose(sigma, math.sqrt(2.0))
print("✅ estimate_normal_mle erfolgreich bestanden!")
"""),
            ("bernoulli_log_likelihood", """
y_true = [1, 0]
y_prob = [0.8, 0.2]
expected_ll = 2.0 * math.log(0.8)
ll = bernoulli_log_likelihood(y_true, y_prob)
assert math.isclose(ll, expected_ll, rel_tol=1e-4)
print("✅ bernoulli_log_likelihood erfolgreich bestanden!")
"""),
            ("binary_cross_entropy_loss", """
y_true = [1, 0]
y_prob = [0.8, 0.2]
expected_ll = 2.0 * math.log(0.8)
bce = binary_cross_entropy_loss(y_true, y_prob)
assert math.isclose(bce, -expected_ll / 2.0, rel_tol=1e-4)
print("✅ binary_cross_entropy_loss erfolgreich bestanden!")
"""),
            ("gaussian_nll_regression", """
nll = gaussian_nll_regression([2.0], [2.0], sigma=1.0)
assert math.isclose(nll, 0.5 * math.log(2.0 * math.pi), rel_tol=1e-4)
print("✅ gaussian_nll_regression erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 6,
        "title": "Modul 06: Informationstheorie (Entropie & KL-Divergenz)",
        "svg": "06_entropy_kl_divergence.svg",
        "description": "Shannon-Entropie H(P), Cross-Entropy H(P,Q), KL-Divergenz D_KL(P||Q) und 1D-Gauß-KL-Divergenz für VAEs.",
        "file": "exercises/ex06_information_theory.py",
        "functions": [
            ("shannon_entropy", """
assert math.isclose(shannon_entropy([0.5, 0.5], base=2.0), 1.0)
print("✅ shannon_entropy erfolgreich bestanden!")
"""),
            ("cross_entropy", """
p = [0.6, 0.4]
q = [0.3, 0.7]
h_pq = cross_entropy(p, q, base=math.e)
expected = -(0.6 * math.log(0.3) + 0.4 * math.log(0.7))
assert math.isclose(h_pq, expected, rel_tol=1e-4)
print("✅ cross_entropy erfolgreich bestanden!")
"""),
            ("kl_divergence", """
p = [0.6, 0.4]
assert math.isclose(kl_divergence(p, p, base=2.0), 0.0, abs_tol=1e-6)
q = [0.3, 0.7]
h_p = shannon_entropy(p, base=math.e)
d_kl = kl_divergence(p, q, base=math.e)
h_pq = cross_entropy(p, q, base=math.e)
assert math.isclose(h_p + d_kl, h_pq, rel_tol=1e-4)
print("✅ kl_divergence erfolgreich bestanden!")
"""),
            ("kl_divergence_two_gaussians_1d", """
assert math.isclose(kl_divergence_two_gaussians_1d(0.0, 1.0, 0.0, 1.0), 0.0, abs_tol=1e-6)
print("✅ kl_divergence_two_gaussians_1d erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 7,
        "title": "Modul 07: Statistische ML-Evaluation & Metriken (ROC-AUC)",
        "svg": "07_confusion_matrix_roc.svg",
        "description": "Confusion Matrix (TP, FP, TN, FN), Precision, Recall, F1-Score, ROC-AUC via Mann-Whitney-Statistik und Brier Score.",
        "file": "exercises/ex07_ml_evaluation_metrics.py",
        "functions": [
            ("compute_confusion_matrix", """
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 1]
cm = compute_confusion_matrix(y_true, y_pred)
assert cm == {"TP": 2, "FP": 1, "TN": 1, "FN": 1}
print("✅ compute_confusion_matrix erfolgreich bestanden!")
"""),
            ("compute_precision_recall_f1", """
p, r, f1 = compute_precision_recall_f1(2, 1, 1)
assert math.isclose(p, 2.0 / 3.0)
assert math.isclose(r, 2.0 / 3.0)
assert math.isclose(f1, 2.0 / 3.0)
print("✅ compute_precision_recall_f1 erfolgreich bestanden!")
"""),
            ("compute_roc_auc_score", """
auc = compute_roc_auc_score([1, 1, 0, 0], [0.9, 0.7, 0.2, 0.4])
assert math.isclose(auc, 1.0)
print("✅ compute_roc_auc_score erfolgreich bestanden!")
"""),
            ("compute_brier_score", """
assert math.isclose(compute_brier_score([1, 0], [0.9, 0.1]), 0.01)
print("✅ compute_brier_score erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 8,
        "title": "Modul 08: Bootstrapping & Ensemble-Statistik (Bagging)",
        "svg": "08_bootstrapping_resampling.svg",
        "description": "Bootstrap Resampling with replacement, empirische Konfidenzintervalle und mathematische Ensemble-Varianzreduktion.",
        "file": "exercises/ex08_bootstrapping_resampling.py",
        "functions": [
            ("bootstrap_sample", """
data = [10.0, 20.0, 30.0, 40.0, 50.0]
sample = bootstrap_sample(data, seed=42)
assert len(sample) == len(data)
for x in sample:
    assert x in data
print("✅ bootstrap_sample erfolgreich bestanden!")
"""),
            ("bootstrap_confidence_interval", """
data = [10.0, 20.0, 30.0, 40.0, 50.0]
mean_func = lambda x: sum(x) / len(x)
low, high = bootstrap_confidence_interval(data, mean_func, num_bootstraps=500, seed=42)
assert low < 30.0 < high
print("✅ bootstrap_confidence_interval erfolgreich bestanden!")
"""),
            ("calculate_ensemble_variance", """
assert math.isclose(calculate_ensemble_variance(10.0, 0.5, 1), 10.0)
var_inf = calculate_ensemble_variance(10.0, 0.2, 10000)
assert math.isclose(var_inf, 2.0, rel_tol=1e-2)
print("✅ calculate_ensemble_variance erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 9,
        "title": "Modul 09: Multivariate Statistik, Kovarianzmatrix & PCA",
        "svg": "09_pca_covariance_projection.svg",
        "description": "Mittelwertvektor, DxD Kovarianzmatrix, Erklärte Varianz (Explained Variance Ratio) und Hauptkomponenten-Projektion.",
        "file": "exercises/ex09_multivariate_pca.py",
        "functions": [
            ("compute_mean_vector", """
matrix = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
mu = compute_mean_vector(matrix)
assert math.isclose(mu[0], 2.0) and math.isclose(mu[1], 20.0)
print("✅ compute_mean_vector erfolgreich bestanden!")
"""),
            ("compute_covariance_matrix", """
matrix = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
cov = compute_covariance_matrix(matrix, sample=True)
assert math.isclose(cov[0][0], 1.0) and math.isclose(cov[1][1], 100.0)
assert math.isclose(cov[0][1], 10.0) and math.isclose(cov[1][0], 10.0)
print("✅ compute_covariance_matrix erfolgreich bestanden!")
"""),
            ("compute_explained_variance_ratio", """
ratios = compute_explained_variance_ratio([80.0, 20.0])
assert math.isclose(ratios[0], 0.8) and math.isclose(ratios[1], 0.2)
print("✅ compute_explained_variance_ratio erfolgreich bestanden!")
"""),
            ("project_onto_component", """
centered = [[-1.0, 2.0], [1.0, -2.0]]
vec = [0.6, 0.8]
proj = project_onto_component(centered, vec)
assert math.isclose(proj[0], 1.0) and math.isclose(proj[1], -1.0)
print("✅ project_onto_component erfolgreich bestanden!")
""")
        ]
    },
    {
        "id": 10,
        "title": "Modul 10: Data Drift & Monitoring in MLOps (PSI & KS)",
        "svg": "10_data_drift_psi_ks.svg",
        "description": "Empirische Verteilungsfunktion (ECDF), 2-Stichproben Kolmogorov-Smirnov-Statistik und Population Stability Index (PSI).",
        "file": "exercises/ex10_distribution_drift_monitoring.py",
        "functions": [
            ("compute_ecdf", """
data = [1.0, 2.0, 3.0, 4.0]
ecdf = compute_ecdf(data, [0.0, 2.0, 3.5, 5.0])
assert math.isclose(ecdf[0], 0.0) and math.isclose(ecdf[1], 0.5)
assert math.isclose(ecdf[2], 0.75) and math.isclose(ecdf[3], 1.0)
print("✅ compute_ecdf erfolgreich bestanden!")
"""),
            ("compute_ks_statistic", """
data = [1.0, 2.0, 3.0, 4.0]
assert math.isclose(compute_ks_statistic(data, data), 0.0)
separated = [10.0, 20.0, 30.0]
assert math.isclose(compute_ks_statistic(data, separated), 1.0)
print("✅ compute_ks_statistic erfolgreich bestanden!")
"""),
            ("compute_psi", """
p_same = [0.25, 0.25, 0.25, 0.25]
psi_zero, status = compute_psi(p_same, p_same)
assert math.isclose(psi_zero, 0.0, abs_tol=1e-5)
assert status == "NO_DRIFT"
p_exp = [0.8, 0.2]
p_act = [0.1, 0.9]
psi_val, status_drift = compute_psi(p_exp, p_act)
assert psi_val > 0.25 and status_drift == "SIGNIFICANT_DRIFT"
print("✅ compute_psi erfolgreich bestanden!")
""")
        ]
    }
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_function_code(file_path: str, func_name: str) -> str:
    """Extrahiert die Funktionsdefinition inklusive Docstring aus der Datei."""
    full_path = os.path.join(BASE_DIR, file_path)
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex zum Auffinden der Funktion bis zur nächsten 'def ' oder 'if __name__'
    pattern = rf"(def {func_name}\(.*?\n(?:    .*\n|\n)*?)(?=\ndef |\nif __name__|\Z)"
    match = re.search(pattern, content)
    if match:
        code = match.group(1).rstrip()
        return code
    return f"def {func_name}(*args, **kwargs):\n    raise NotImplementedError('Bitte implementieren!')"


notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 Statistik für AI/ML-Engineers – Vollständiger Interaktiver Kurs\n",
                "\n",
                "[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/bohne159-code/ml-statistics-course)\n",
                "\n",
                "Dies ist das **vollständige, ungekürzte Notebook** mit allen **10 Modulen und allen 42 praktischen Aufgaben**!\n",
                "Du kannst jede einzelne Aufgabe direkt hier im Browser auf deinem PC, Tablet (iPad) oder Smartphone programmieren und testen.\n",
                "\n",
                "### 📱 Bedienung in Google Colab:\n",
                "1. Führe zuerst die Imports-Zelle ganz oben aus.\n",
                "2. Wähle die gewünschte Aufgabe, schreibe deine Lösung und tippe auf **▶️ Play**.\n",
                "3. Führe direkt darunter die zugehörige **Test-Zelle** aus – sie prüft deine Funktion isoliert und gibt sofort Feedback!\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Globale Imports für das gesamte Notebook\n",
                "import math\n",
                "import random\n",
                "from typing import List, Tuple, Dict, Callable\n",
                "\n",
                "print('🚀 System initialisiert. Du kannst jetzt mit den Übungen starten!')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

total_funcs = 0

for mod in MODULE_SECTIONS:
    svg_url = f"https://raw.githubusercontent.com/bohne159-code/ml-statistics-course/main/assets/{mod['svg']}"
    
    # Section Markdown Header
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"## {mod['title']}\n",
            f"*{mod['description']}*\n\n",
            f"![Visualisierung]({svg_url})\n\n",
            "---\n"
        ]
    })

    # Add each function + its dedicated test
    for func_name, test_code in mod["functions"]:
        total_funcs += 1
        fn_code = extract_function_code(mod["file"], func_name)
        
        # Header for the single function
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### ✏️ Aufgabe: `{func_name}`\n",
                f"Implementiere die Funktion in der folgenden Zelle und teste sie anschließend darunter."
            ]
        })

        # Function implementation cell
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [fn_code + "\n"]
        })

        # Test verification cell
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"# --- Test für {func_name} ---" + test_code]
        })

out_file = os.path.join(BASE_DIR, "ml_statistics_course.ipynb")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Fertig! Vollständiges Notebook mit {total_funcs} Einzelfunktionen generiert unter:\n{out_file}")
