# 🧠 Statistik für AI & ML-Engineers: Interaktiver Praxiskurs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bohne159-code/ml-statistics-course/blob/main/ml_statistics_course.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/bohne159-code/ml-statistics-course)

Willkommen zum praxisorientierten Statistikkurs für angehende **AI- und Machine-Learning-Engineers**.
Hier lernst du die mathematischen und statistischen Grundlagen nicht abstrakt auswendig, sondern programmierst die Kernkonzepte in Python nach und siehst unmittelbar, warum sie für modernes Deep Learning, Feature Engineering, Modell-Losses und Produktiv-Deployments unerlässlich sind.

---

## 📂 Kursstruktur

```text
ml_statistics_course/
├── exercises/                           # Deine Arbeitsdateien (hier programmierst du!)
│   ├── ex01_descriptive_stats.py        # 1. Deskriptive Statistik & EDA
│   ├── ex02_probability_bayes.py        # 2. Wahrscheinlichkeitsrechnung & Bayes-Theorem
│   ├── ex03_distributions_clt.py        # 3. Verteilungen & Zentraler Grenzwertsatz (CLT)
│   ├── ex04_hypothesis_ab_testing.py    # 4. Hypothesentests & A/B-Testing für ML
│   ├── ex05_mle_and_loss_functions.py   # 5. MLE & Herleitung von MSE / Cross-Entropy
│   ├── ex06_information_theory.py       # 6. Informationstheorie (Entropie, KL-Divergenz)
│   ├── ex07_ml_evaluation_metrics.py    # 7. Statistische ML-Evaluation & Metriken (ROC-AUC)
│   ├── ex08_bootstrapping_resampling.py # 8. Bootstrapping & Ensemble-Statistik (Bagging)
│   ├── ex09_multivariate_pca.py         # 9. Multivariate Statistik, Kovarianzmatrix & PCA
│   └── ex10_distribution_drift_monitoring.py # 10. Data Drift & MLOps Monitoring (PSI, KS)
├── solutions/                           # Vollständige Musterlösungen zur Selbstkontrolle
├── tests/                               # Automatische Test-Suiten
├── test_runner.py                       # Interaktiver Test-Runner
├── STUDY_GUIDE.md                       # Detailliertes Studienhandbuch mit Theorie & Formeln
└── requirements.txt                     # Optionale Pakete für weiterführende Exploration
```

---

## 📖 Begleitendes Studien-Handbuch (`STUDY_GUIDE.md`)
Bevor du eine Übung löst, wirf einen Blick in das mitgelieferte Handbuch:
👉 **[`STUDY_GUIDE.md`](STUDY_GUIDE.md)**
Dort findest du für jedes einzelne Modul:
- Die exakten mathematischen Formeln
- Ein von Hand durchgerechnetes Zahlenbeispiel
- Hilfreiche Python-Tipps
- Verweise auf die besten freien Buchkapitel und Videos!

---

## ⚡ Schnellstart: Jede Funktion einzeln testen!

Du musst **nicht** alle Funktionen einer Übung auf einmal fertigstellen! Du kannst jede Funktion einzeln implementieren und sofort isoliert testen.

### 1. Einzelne Funktion gezielt testen:
```powershell
# Testet nur 'calculate_mean'
py test_runner.py -f calculate_mean

# Testet alle Funktionen, die 'bayes' im Namen haben
py test_runner.py -f bayes
```

### 2. Alle Funktionen eines Moduls einzeln auflisten:
```powershell
# Zeigt den Status jeder einzelnen Funktion in Modul 1
py test_runner.py -m 1
```

### 3. Gesamten Kursfortschritt ansehen:
```powershell
py test_runner.py
```

### 4. Einzelne Übungsdatei direkt ausführen:
Jede Datei in `exercises/` kann auch direkt ausgeführt werden und testet jede Funktion unabhängig voneinander:
```powershell
py exercises/ex01_descriptive_stats.py
```

---

## 🎯 Warum diese 10 Module für ML Engineers entscheidend sind

| Modul | Mathematischer Kern | Warum du das als ML-Engineer brauchst |
| :--- | :--- | :--- |
| **01. Deskriptive Statistik** | Mean, Median, Varianz, Cov, Korrelation, IQR | Verstehen von Verteilungen, Ausreißer-Bereinigung, Feature-Scaling ($z$-Score), Multikollinearität |
| **02. Wahrscheinlichkeit & Bayes** | Bedingte Wahrscheinlichkeit, Bayes-Theorem, Odds | Base-Rate Fallacy (Imbalanced Data), Naive Bayes Klassifikation, Bayesian Optimization |
| **03. Verteilungen & CLT** | Gauß, Bernoulli, Zentraler Grenzwertsatz, SEM | Gewichtsinitialisierung (He/Xavier), Rauschprozesse in Diffusionsmodellen, Latent Space in VAEs |
| **04. Hypothesentests & A/B-Testing** | $Z$-Test, $p$-Wert, Statistische Power, MDE | Deployment-Entscheidungen im Produktivbetrieb (Ist Modell B wirklich besser als Modell A?) |
| **05. MLE & Loss-Funktionen** | Maximum Likelihood Estimation, Log-Likelihood | Verstehen, warum MSE = MLE unter Gauß-Rauschen ist und Cross-Entropy = MLE unter Bernoulli |
| **06. Informationstheorie** | Shannon-Entropie, Cross-Entropy, KL-Divergenz | Loss von Klassifikatoren & LLMs, Regularisierung in RLHF (PPO Policy Drift), VAE Latent Loss |
| **07. ML-Evaluation & Metriken** | Konfusionsmatrix, Precision, Recall, ROC-AUC, Brier | Beurteilung von Modellen jenseits einfacher Accuracy, Wahrscheinlichkeitskalibrierung |
| **08. Bootstrapping & Resampling** | Bootstrap Sampling, Nicht-param. CI, Bagging-Varianz | Fundament von Random Forests, Konfidenzintervalle für beliebige ML-Metriken ohne Verteilungsannahme |
| **09. Multivariate Statistik & PCA** | Kovarianzmatrix $\Sigma$, Mahalanobis-Distanz, PCA | Arbeit in hochdimensionalen Vektorräumen, Dimensionsreduktion, Maximierung der Merkmalsvarianz |
| **10. Data Drift & MLOps** | ECDF, Kolmogorov-Smirnov-Test, PSI | Monitoring von Modellen in Produktion, Erkennung von Covariate Shift & automatisiertes Retraining |

---

## 💡 Workflow-Empfehlung
1. Öffne die jeweilige Übungsdatei im Editor (z.B. `exercises/ex01_descriptive_stats.py`).
2. Lies die Erklärungen und die Formel im Docstring.
3. Ersetze das `raise NotImplementedError(...)` mit deinem Code.
4. Führe `py test_runner.py --module 1` aus, bis der Test grün wird.
5. Vergleiche deine Lösung bei Bedarf mit `solutions/sol01_descriptive_stats.py`.
