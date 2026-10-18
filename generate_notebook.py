"""
Skript zur Erstellung eines interaktiven Jupyter Notebooks (Google Colab kompatibel)
für das mobile Lernen am Smartphone!
"""

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 📱 Statistik für AI/ML-Engineers – Mobiles Jupyter Notebook\n",
                "\n",
                "Willkommen zur mobilen Version des Statistikkurses! Dieses Notebook kannst du direkt auf deinem **Smartphone oder Tablet in Google Colab** öffnen und bearbeiten.\n",
                "\n",
                "### 🚀 So funktioniert's am Handy:\n",
                "1. Öffne die jeweilige Code-Zelle.\n",
                "2. Ersetze das `raise NotImplementedError(...)` mit deinem Code.\n",
                "3. Tippe auf den Play-Button (▶️) der Zelle und führe danach direkt die Test-Zelle aus!\n",
                "\n",
                "---"
            ]
        },
        # MODUL 01
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 Modul 01: Deskriptive Statistik & EDA\n",
                "- **Mean:** $\\mu = \\frac{1}{n} \\sum x_i$\n",
                "- **Median:** Zentralwert der sortierten Liste\n",
                "- **Varianz:** $s^2 = \\frac{1}{n-1} \\sum (x_i - \\mu)^2$\n",
                "- **Std-Abweichung:** $s = \\sqrt{s^2}$\n",
                "- **Kovarianz:** $\\text{Cov}(x, y) = \\frac{1}{n-1} \\sum (x_i - \\mu_x)(y_i - \\mu_y)$\n",
                "- **Pearson-Korrelation:** $r = \\frac{\\text{Cov}(x,y)}{s_x s_y}$\n",
                "- **IQR-Ausreißer:** $Q_3 - Q_1$, Grenzen: $[Q_1 - 1.5\\cdot \\text{IQR}, Q_3 + 1.5\\cdot \\text{IQR}]$\n",
                "\n",
                "🎥 [Video: StatQuest Variance & Standard Deviation](https://www.youtube.com/watch?v=SzZ6G434EBs)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import math\n",
                "from typing import List, Tuple\n",
                "\n",
                "def calculate_mean(data: List[float]) -> float:\n",
                "    if not data:\n",
                "        raise ValueError('Datenliste darf nicht leer sein.')\n",
                "    return sum(data) / len(data)\n",
                "\n",
                "def calculate_median(data: List[float]) -> float:\n",
                "    if not data:\n",
                "        raise ValueError('Datenliste darf nicht leer sein.')\n",
                "    s = sorted(data)\n",
                "    n = len(s)\n",
                "    mid = n // 2\n",
                "    if n % 2 == 1:\n",
                "        return float(s[mid])\n",
                "    return (s[mid - 1] + s[mid]) / 2.0\n",
                "\n",
                "def calculate_variance(data: List[float], sample: bool = True) -> float:\n",
                "    if not data:\n",
                "        raise ValueError('Datenliste darf nicht leer sein.')\n",
                "    n = len(data)\n",
                "    if sample and n < 2:\n",
                "        raise ValueError('Stichproben-Varianz erfordert n >= 2.')\n",
                "    m = calculate_mean(data)\n",
                "    div = (n - 1) if sample else n\n",
                "    return sum((x - m) ** 2 for x in data) / div\n",
                "\n",
                "def calculate_std_dev(data: List[float], sample: bool = True) -> float:\n",
                "    return math.sqrt(calculate_variance(data, sample=sample))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# --- TEST FÜR MODUL 1 ---\n",
                "d = [10.0, 20.0, 30.0, 40.0, 50.0]\n",
                "assert calculate_mean(d) == 30.0\n",
                "assert calculate_median(d) == 30.0\n",
                "assert calculate_median([1.0, 2.0, 3.0, 4.0]) == 2.5\n",
                "print('✅ Modul 1 Basistests erfolgreich bestanden!')"
            ]
        },
        # MODUL 02
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎲 Modul 02: Bayes-Theorem & Wahrscheinlichkeitsrechnung\n",
                "Formel: $P(A \\mid B) = \\frac{P(B \\mid A) \\cdot P(A)}{P(B \\mid A)P(A) + P(B \\mid \\neg A)P(\\neg A)}$\n",
                "\n",
                "🎥 [Video: 3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def bayes_theorem(prior_a: float, likelihood_b_given_a: float, likelihood_b_given_not_a: float) -> float:\n",
                "    # TODO: Berechne Posterior P(A | B)\n",
                "    prior_not_a = 1.0 - prior_a\n",
                "    total_b = (likelihood_b_given_a * prior_a) + (likelihood_b_given_not_a * prior_not_a)\n",
                "    return (likelihood_b_given_a * prior_a) / total_b\n",
                "\n",
                "# TEST:\n",
                "p_fraud = bayes_theorem(0.01, 0.95, 0.02)\n",
                "print(f'P(Betrug | Alarm) = {p_fraud:.4f} (ca. 32.4%)')\n",
                "assert round(p_fraud, 3) == 0.324\n",
                "print('✅ Bayes-Theorem Test bestanden!')"
            ]
        },
        # MODUL 03
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔔 Modul 03: Normalverteilung & Zentraler Grenzwertsatz (CLT)\n",
                "Formel Normal-PDF: $f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{1}{2} (\\frac{x-\\mu}{\\sigma})^2}$\n",
                "Standardfehler: $\\text{SEM} = \\frac{\\sigma}{\\sqrt{n}}$\n",
                "\n",
                "🎥 [Video: StatQuest - Central Limit Theorem](https://www.youtube.com/watch?v=YAlJCEDH2uY)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:\n",
                "    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))\n",
                "    exponent = -0.5 * (((x - mu) / sigma) ** 2)\n",
                "    return coeff * math.exp(exponent)\n",
                "\n",
                "def standard_error_of_mean(sample_std: float, n: int) -> float:\n",
                "    return sample_std / math.sqrt(n)\n",
                "\n",
                "# TEST:\n",
                "assert math.isclose(normal_pdf(0.0), 0.39894228, rel_tol=1e-4)\n",
                "assert standard_error_of_mean(10.0, 100) == 1.0\n",
                "print('✅ Modul 3 Tests bestanden!')"
            ]
        },
        # MODUL 04
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧪 Modul 04: Hypothesentests & A/B-Testing für ML\n",
                "Formel Evan Miller Sample Size: $n \\approx \\frac{16 \\cdot p(1-p)}{\\Delta^2}$\n",
                "\n",
                "🎥 [Video: StatQuest - Hypothesis Testing & p-values](https://www.youtube.com/watch?v=vemZtEM63GY)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def z_score_proportions(k_a: int, n_a: int, k_b: int, n_b: int) -> float:\n",
                "    p_a = k_a / n_a\n",
                "    p_b = k_b / n_b\n",
                "    p_pool = (k_a + k_b) / (n_a + n_b)\n",
                "    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))\n",
                "    return (p_b - p_a) / se_pool\n",
                "\n",
                "def estimate_required_sample_size(baseline_p: float, rel_lift: float) -> int:\n",
                "    delta = baseline_p * rel_lift\n",
                "    n = 16.0 * baseline_p * (1.0 - baseline_p) / (delta ** 2)\n",
                "    return math.ceil(n)\n",
                "\n",
                "# TEST:\n",
                "z = z_score_proportions(1000, 20000, 1150, 20000)\n",
                "n_req = estimate_required_sample_size(0.05, 0.10)\n",
                "print(f'Z-Score: {z:.3f}, Benötigte Nutzer: {n_req}')\n",
                "assert z > 3.0 and n_req == 30400\n",
                "print('✅ Modul 4 Tests bestanden!')"
            ]
        },
        # MODUL 05
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 Modul 05: MLE & Loss-Funktionen (MSE / Cross-Entropy)\n",
                "Formel Binary Cross-Entropy: $\\text{BCE} = -\\frac{1}{n} \\sum [y \\ln(p) + (1-y)\\ln(1-p)]$\n",
                "\n",
                "🎥 [Video: StatQuest - Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def binary_cross_entropy_loss(y_true: List[int], y_prob: List[float], eps: float = 1e-15) -> float:\n",
                "    total = 0.0\n",
                "    for y, p in zip(y_true, y_prob):\n",
                "        p_safe = max(eps, min(1.0 - eps, p))\n",
                "        total += y * math.log(p_safe) + (1.0 - y) * math.log(1.0 - p_safe)\n",
                "    return -total / len(y_true)\n",
                "\n",
                "# TEST:\n",
                "bce = binary_cross_entropy_loss([1, 0], [0.8, 0.2])\n",
                "print(f'BCE Loss: {bce:.4f}')\n",
                "assert math.isclose(bce, -math.log(0.8), rel_tol=1e-4)\n",
                "print('✅ Modul 5 Tests bestanden!')"
            ]
        },
        # MODUL 06
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 Modul 06: Informationstheorie (Entropie & KL-Divergenz)\n",
                "Formel Shannon-Entropie: $H(P) = -\\sum P(x) \\log_2 P(x)$\n",
                "Formel KL-Divergenz: $D_{\\text{KL}}(P \\parallel Q) = \\sum P(x) \\log_2 \\frac{P(x)}{Q(x)}$\n",
                "\n",
                "🎥 [Video: 3Blue1Brown - Shannon Entropy](https://www.youtube.com/watch?v=v68zYyaEm-U)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def shannon_entropy(probs: List[float], base: float = 2.0) -> float:\n",
                "    h = 0.0\n",
                "    for p in probs:\n",
                "        if p > 1e-15:\n",
                "            h -= p * (math.log(p) / math.log(base))\n",
                "    return h\n",
                "\n",
                "# TEST:\n",
                "assert math.isclose(shannon_entropy([0.5, 0.5]), 1.0)\n",
                "print('✅ Modul 6 Tests bestanden!')"
            ]
        },
        # MODUL 07
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📈 Modul 07: Statistische ML-Evaluation & ROC-AUC\n",
                "ROC-AUC via Mann-Whitney-Statistik: $P(\\text{Score}_+ > \\text{Score}_-)$\n",
                "\n",
                "🎥 [Video: StatQuest - ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compute_roc_auc_score(y_true: List[int], y_scores: List[float]) -> float:\n",
                "    pos = [s for y, s in zip(y_true, y_scores) if y == 1]\n",
                "    neg = [s for y, s in zip(y_true, y_scores) if y == 0]\n",
                "    ranks = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)\n",
                "    return ranks / (len(pos) * len(neg))\n",
                "\n",
                "# TEST:\n",
                "auc = compute_roc_auc_score([1, 1, 0, 0], [0.9, 0.7, 0.2, 0.4])\n",
                "assert auc == 1.0\n",
                "print('✅ Modul 7 Tests bestanden!')"
            ]
        },
        # MODUL 08, 09, 10
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🌲 Modul 08: Bootstrapping & Ensemble-Varianz\n",
                "Ensemble-Varianz: $\\text{Var}(\\text{Ens}) = \\rho \\sigma^2 + \\frac{1-\\rho}{M} \\sigma^2$\n",
                "\n",
                "🎥 [Video: StatQuest - Bootstrapping](https://www.youtube.com/watch?v=Xz0x-8-cgaQ)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import random\n",
                "def calculate_ensemble_variance(base_var: float, rho: float, m: int) -> float:\n",
                "    return rho * base_var + ((1.0 - rho) / m) * base_var\n",
                "\n",
                "# TEST:\n",
                "v = calculate_ensemble_variance(10.0, 0.2, 50)\n",
                "assert math.isclose(v, 2.16)\n",
                "print('✅ Modul 8 Tests bestanden!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧭 Modul 09: Kovarianzmatrix & PCA\n",
                "🎥 [Video: 3Blue1Brown - Eigenvectors and Eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g)\n",
                "🎥 [Video: StatQuest - PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compute_covariance_matrix(matrix: List[List[float]]) -> List[List[float]]:\n",
                "    n = len(matrix)\n",
                "    d = len(matrix[0])\n",
                "    means = [sum(row[j] for row in matrix) / n for j in range(d)]\n",
                "    cov = [[0.0]*d for _ in range(d)]\n",
                "    for j in range(d):\n",
                "        for k in range(j, d):\n",
                "            val = sum((matrix[i][j] - means[j]) * (matrix[i][k] - means[k]) for i in range(n)) / (n - 1)\n",
                "            cov[j][k] = val\n",
                "            cov[k][j] = val\n",
                "    return cov\n",
                "\n",
                "# TEST:\n",
                "cov = compute_covariance_matrix([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])\n",
                "assert cov[0][0] == 1.0 and cov[1][1] == 100.0\n",
                "print('✅ Modul 9 Tests bestanden!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚨 Modul 10: Data Drift & Monitoring (PSI & KS-Test)\n",
                "Formel: $\\text{PSI} = \\sum (A_b - E_b) \\cdot \\ln(A_b / E_b)$\n",
                "\n",
                "🎥 [Video: StatQuest - Kolmogorov-Smirnov Test](https://www.youtube.com/watch?v=0kSZeWw_3zQ)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compute_psi(expected: List[float], actual: List[float], eps: float = 1e-6) -> float:\n",
                "    return sum((a - e) * math.log(max(eps, a) / max(eps, e)) for e, a in zip(expected, actual))\n",
                "\n",
                "# TEST:\n",
                "psi_val = compute_psi([0.4, 0.4, 0.2], [0.1, 0.3, 0.6])\n",
                "print(f'PSI: {psi_val:.4f}')\n",
                "assert psi_val > 0.25  # Signifikanter Drift!\n",
                "print('✅ Modul 10 Tests bestanden! Alle 10 Module sind mobil verfügbar!')"
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
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(os.path.dirname(__file__), "ml_statistics_course.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Jupyter Notebook erfolgreich erstellt unter: {out_path}")
