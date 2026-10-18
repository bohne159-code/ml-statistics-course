"""
Granulare Test-Suite für den ML-Statistik-Kurs
Jede Funktion hat einen eigenen, unabhängigen Test!
"""

import math

# =====================================================================
# MODUL 01: Deskriptive Statistik
# =====================================================================

def test_calculate_mean(module):
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert math.isclose(module.calculate_mean(data), 30.0)
    assert math.isclose(module.calculate_mean([5.0]), 5.0)


def test_calculate_median(module):
    assert math.isclose(module.calculate_median([10.0, 20.0, 30.0, 40.0, 50.0]), 30.0)
    assert math.isclose(module.calculate_median([1.0, 2.0, 3.0, 4.0]), 2.5)
    assert math.isclose(module.calculate_median([5.0]), 5.0)


def test_calculate_variance(module):
    d2 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert math.isclose(module.calculate_variance(d2, sample=True), 32.0 / 7.0, rel_tol=1e-4)
    assert math.isclose(module.calculate_variance(d2, sample=False), 4.0)


def test_calculate_std_dev(module):
    d2 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert math.isclose(module.calculate_std_dev(d2, sample=False), 2.0)
    assert math.isclose(module.calculate_std_dev(d2, sample=True), math.sqrt(32.0 / 7.0), rel_tol=1e-4)


def test_calculate_covariance(module):
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    cov_samp = module.calculate_covariance(x, y, sample=True)
    assert math.isclose(cov_samp, 5.0)
    cov_pop = module.calculate_covariance(x, y, sample=False)
    assert math.isclose(cov_pop, 4.0)


def test_calculate_pearson_correlation(module):
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert math.isclose(module.calculate_pearson_correlation(x, y), 1.0, rel_tol=1e-5)
    y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
    assert math.isclose(module.calculate_pearson_correlation(x, y_neg), -1.0, rel_tol=1e-5)


def test_detect_outliers_iqr(module):
    d_outliers = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 100.0]
    outliers, (lb, ub) = module.detect_outliers_iqr(d_outliers)
    assert 100.0 in outliers
    assert len(outliers) == 1
    assert lb < 1.0 and ub < 100.0


# =====================================================================
# MODUL 02: Wahrscheinlichkeit & Bayes
# =====================================================================

def test_bayes_theorem(module):
    p = module.bayes_theorem(0.001, 0.99, 0.05)
    expected = (0.99 * 0.001) / ((0.99 * 0.001) + (0.05 * 0.999))
    assert math.isclose(p, expected, rel_tol=1e-3)


def test_posterior_odds_update(module):
    new_odds = module.posterior_odds_update(1.0, 2.0)
    assert math.isclose(new_odds, 2.0)
    assert math.isclose(module.posterior_odds_update(0.5, 3.0), 1.5)


def test_odds_to_probability(module):
    assert math.isclose(module.odds_to_probability(3.0), 0.75)
    assert math.isclose(module.odds_to_probability(1.0), 0.5)


def test_naive_bayes_predict_spam(module):
    spam_vocab = {"win": 0.9, "cash": 0.8}
    ham_vocab = {"win": 0.05, "cash": 0.05}
    label, prob = module.naive_bayes_predict_spam(["win", "cash"], spam_vocab, ham_vocab, prior_spam=0.5)
    assert label == "spam"
    assert prob > 0.95


# =====================================================================
# MODUL 03: Verteilungen & CLT
# =====================================================================

def test_normal_pdf(module):
    pdf_val = module.normal_pdf(0.0, mu=0.0, sigma=1.0)
    assert math.isclose(pdf_val, 1.0 / math.sqrt(2 * math.pi), rel_tol=1e-4)


def test_standard_error_of_mean(module):
    assert math.isclose(module.standard_error_of_mean(10.0, 100), 1.0)
    assert math.isclose(module.standard_error_of_mean(4.0, 16), 1.0)


def test_bernoulli_pmf(module):
    assert math.isclose(module.bernoulli_pmf(1, 0.7), 0.7)
    assert math.isclose(module.bernoulli_pmf(0, 0.7), 0.3)


def test_simulate_clt(module):
    means = module.simulate_clt("exponential", sample_size=40, num_experiments=500, seed=123)
    avg_mean = sum(means) / len(means)
    assert math.isclose(avg_mean, 1.0, rel_tol=0.08)
    assert len(means) == 500


def test_confidence_interval_95(module):
    low, high = module.confidence_interval_95(100.0, 2.0)
    assert math.isclose(low, 100.0 - 1.96 * 2.0)
    assert math.isclose(high, 100.0 + 1.96 * 2.0)


# =====================================================================
# MODUL 04: Hypothesentests & A/B-Testing
# =====================================================================

def test_z_score_proportions(module):
    z = module.z_score_proportions(100, 1000, 120, 1000)
    assert z > 0


def test_approximate_p_value_two_tailed(module):
    # z=0 -> p=1.0; z=1.96 -> p~0.05
    assert math.isclose(module.approximate_p_value_two_tailed(0.0), 1.0)
    assert math.isclose(module.approximate_p_value_two_tailed(1.96), 0.05, rel_tol=0.05)


def test_evaluate_ab_test(module):
    res = module.evaluate_ab_test(1000, 20000, 1200, 20000, alpha=0.05)
    assert res["is_significant"] is True
    assert res["decision"] == "DEPLOY_MODEL_B"

    res_null = module.evaluate_ab_test(1000, 20000, 1005, 20000, alpha=0.05)
    assert res_null["is_significant"] is False
    assert res_null["decision"] == "KEEP_MODEL_A"


def test_estimate_required_sample_size(module):
    n = module.estimate_required_sample_size(0.05, 0.10)
    assert n == 30400


# =====================================================================
# MODUL 05: MLE & Loss-Funktionen
# =====================================================================

def test_estimate_normal_mle(module):
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mu, sigma = module.estimate_normal_mle(data)
    assert math.isclose(mu, 3.0)
    assert math.isclose(sigma, math.sqrt(2.0))


def test_bernoulli_log_likelihood(module):
    y_true = [1, 0]
    y_prob = [0.8, 0.2]
    expected_ll = 2.0 * math.log(0.8)
    ll = module.bernoulli_log_likelihood(y_true, y_prob)
    assert math.isclose(ll, expected_ll, rel_tol=1e-4)


def test_binary_cross_entropy_loss(module):
    y_true = [1, 0]
    y_prob = [0.8, 0.2]
    expected_ll = 2.0 * math.log(0.8)
    bce = module.binary_cross_entropy_loss(y_true, y_prob)
    assert math.isclose(bce, -expected_ll / 2.0, rel_tol=1e-4)


def test_gaussian_nll_regression(module):
    nll = module.gaussian_nll_regression([2.0], [2.0], sigma=1.0)
    assert math.isclose(nll, 0.5 * math.log(2.0 * math.pi), rel_tol=1e-4)


# =====================================================================
# MODUL 06: Informationstheorie
# =====================================================================

def test_shannon_entropy(module):
    h_fair = module.shannon_entropy([0.5, 0.5], base=2.0)
    assert math.isclose(h_fair, 1.0)


def test_cross_entropy(module):
    p = [0.6, 0.4]
    q = [0.3, 0.7]
    h_pq = module.cross_entropy(p, q, base=math.e)
    expected = -(0.6 * math.log(0.3) + 0.4 * math.log(0.7))
    assert math.isclose(h_pq, expected, rel_tol=1e-4)


def test_kl_divergence(module):
    p = [0.6, 0.4]
    kl_zero = module.kl_divergence(p, p, base=2.0)
    assert math.isclose(kl_zero, 0.0, abs_tol=1e-6)

    q = [0.3, 0.7]
    h_p = module.shannon_entropy(p, base=math.e)
    d_kl = module.kl_divergence(p, q, base=math.e)
    h_pq = module.cross_entropy(p, q, base=math.e)
    assert math.isclose(h_p + d_kl, h_pq, rel_tol=1e-4)


def test_kl_divergence_two_gaussians_1d(module):
    kl_g = module.kl_divergence_two_gaussians_1d(0.0, 1.0, 0.0, 1.0)
    assert math.isclose(kl_g, 0.0, abs_tol=1e-6)


# =====================================================================
# MODUL 07: ML-Evaluation
# =====================================================================

def test_compute_confusion_matrix(module):
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 1]
    cm = module.compute_confusion_matrix(y_true, y_pred)
    assert cm == {"TP": 2, "FP": 1, "TN": 1, "FN": 1}


def test_compute_precision_recall_f1(module):
    p, r, f1 = module.compute_precision_recall_f1(2, 1, 1)
    assert math.isclose(p, 2.0 / 3.0)
    assert math.isclose(r, 2.0 / 3.0)
    assert math.isclose(f1, 2.0 / 3.0)


def test_compute_roc_auc_score(module):
    auc = module.compute_roc_auc_score([1, 1, 0, 0], [0.9, 0.7, 0.2, 0.4])
    assert math.isclose(auc, 1.0)


def test_compute_brier_score(module):
    brier = module.compute_brier_score([1, 0], [0.9, 0.1])
    assert math.isclose(brier, 0.01)


# =====================================================================
# MODUL 08: Bootstrapping & Resampling
# =====================================================================

def test_bootstrap_sample(module):
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    sample = module.bootstrap_sample(data, seed=42)
    assert len(sample) == len(data)
    for x in sample:
        assert x in data


def test_bootstrap_confidence_interval(module):
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    mean_func = lambda x: sum(x) / len(x)
    low, high = module.bootstrap_confidence_interval(data, mean_func, num_bootstraps=500, seed=42)
    assert low < 30.0 < high


def test_calculate_ensemble_variance(module):
    assert math.isclose(module.calculate_ensemble_variance(10.0, 0.5, 1), 10.0)
    var_inf = module.calculate_ensemble_variance(10.0, 0.2, 10000)
    assert math.isclose(var_inf, 2.0, rel_tol=1e-2)


# =====================================================================
# MODUL 09: Multivariate Statistik & PCA
# =====================================================================

def test_compute_mean_vector(module):
    matrix = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    mu = module.compute_mean_vector(matrix)
    assert math.isclose(mu[0], 2.0)
    assert math.isclose(mu[1], 20.0)


def test_compute_covariance_matrix(module):
    matrix = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    cov = module.compute_covariance_matrix(matrix, sample=True)
    assert math.isclose(cov[0][0], 1.0)
    assert math.isclose(cov[1][1], 100.0)
    assert math.isclose(cov[0][1], 10.0)
    assert math.isclose(cov[1][0], 10.0)


def test_compute_explained_variance_ratio(module):
    ratios = module.compute_explained_variance_ratio([80.0, 20.0])
    assert math.isclose(ratios[0], 0.8)
    assert math.isclose(ratios[1], 0.2)


def test_project_onto_component(module):
    centered = [[-1.0, 2.0], [1.0, -2.0]]
    vec = [0.6, 0.8]
    proj = module.project_onto_component(centered, vec)
    assert math.isclose(proj[0], -0.6 + 1.6)
    assert math.isclose(proj[1], 0.6 - 1.6)


# =====================================================================
# MODUL 10: Data Drift & MLOps
# =====================================================================

def test_compute_ecdf(module):
    data = [1.0, 2.0, 3.0, 4.0]
    ecdf = module.compute_ecdf(data, [0.0, 2.0, 3.5, 5.0])
    assert math.isclose(ecdf[0], 0.0)
    assert math.isclose(ecdf[1], 0.5)
    assert math.isclose(ecdf[2], 0.75)
    assert math.isclose(ecdf[3], 1.0)


def test_compute_ks_statistic(module):
    data = [1.0, 2.0, 3.0, 4.0]
    assert math.isclose(module.compute_ks_statistic(data, data), 0.0)
    separated = [10.0, 20.0, 30.0]
    assert math.isclose(module.compute_ks_statistic(data, separated), 1.0)


def test_compute_psi(module):
    p_same = [0.25, 0.25, 0.25, 0.25]
    psi_zero, status = module.compute_psi(p_same, p_same)
    assert math.isclose(psi_zero, 0.0, abs_tol=1e-5)
    assert status == "NO_DRIFT"

    p_exp = [0.8, 0.2]
    p_act = [0.1, 0.9]
    psi_val, status_drift = module.compute_psi(p_exp, p_act)
    assert psi_val > 0.25
    assert status_drift == "SIGNIFICANT_DRIFT"


# =====================================================================
# ZUORDNUNG: MODUL -> EINZELNE FUNKTIONS-TESTS
# =====================================================================

MODULE_FUNCTION_TESTS = {
    1: [
        ("calculate_mean", test_calculate_mean),
        ("calculate_median", test_calculate_median),
        ("calculate_variance", test_calculate_variance),
        ("calculate_std_dev", test_calculate_std_dev),
        ("calculate_covariance", test_calculate_covariance),
        ("calculate_pearson_correlation", test_calculate_pearson_correlation),
        ("detect_outliers_iqr", test_detect_outliers_iqr),
    ],
    2: [
        ("bayes_theorem", test_bayes_theorem),
        ("posterior_odds_update", test_posterior_odds_update),
        ("odds_to_probability", test_odds_to_probability),
        ("naive_bayes_predict_spam", test_naive_bayes_predict_spam),
    ],
    3: [
        ("normal_pdf", test_normal_pdf),
        ("standard_error_of_mean", test_standard_error_of_mean),
        ("bernoulli_pmf", test_bernoulli_pmf),
        ("simulate_clt", test_simulate_clt),
        ("confidence_interval_95", test_confidence_interval_95),
    ],
    4: [
        ("z_score_proportions", test_z_score_proportions),
        ("approximate_p_value_two_tailed", test_approximate_p_value_two_tailed),
        ("evaluate_ab_test", test_evaluate_ab_test),
        ("estimate_required_sample_size", test_estimate_required_sample_size),
    ],
    5: [
        ("estimate_normal_mle", test_estimate_normal_mle),
        ("bernoulli_log_likelihood", test_bernoulli_log_likelihood),
        ("binary_cross_entropy_loss", test_binary_cross_entropy_loss),
        ("gaussian_nll_regression", test_gaussian_nll_regression),
    ],
    6: [
        ("shannon_entropy", test_shannon_entropy),
        ("cross_entropy", test_cross_entropy),
        ("kl_divergence", test_kl_divergence),
        ("kl_divergence_two_gaussians_1d", test_kl_divergence_two_gaussians_1d),
    ],
    7: [
        ("compute_confusion_matrix", test_compute_confusion_matrix),
        ("compute_precision_recall_f1", test_compute_precision_recall_f1),
        ("compute_roc_auc_score", test_compute_roc_auc_score),
        ("compute_brier_score", test_compute_brier_score),
    ],
    8: [
        ("bootstrap_sample", test_bootstrap_sample),
        ("bootstrap_confidence_interval", test_bootstrap_confidence_interval),
        ("calculate_ensemble_variance", test_calculate_ensemble_variance),
    ],
    9: [
        ("compute_mean_vector", test_compute_mean_vector),
        ("compute_covariance_matrix", test_compute_covariance_matrix),
        ("compute_explained_variance_ratio", test_compute_explained_variance_ratio),
        ("project_onto_component", test_project_onto_component),
    ],
    10: [
        ("compute_ecdf", test_compute_ecdf),
        ("compute_ks_statistic", test_compute_ks_statistic),
        ("compute_psi", test_compute_psi),
    ],
}
