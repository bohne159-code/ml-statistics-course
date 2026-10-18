"""
LÖSUNG 09: Multivariate Statistik, Kovarianzmatrix & PCA
"""

from typing import List, Tuple


def compute_mean_vector(matrix: List[List[float]]) -> List[float]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix darf nicht leer sein.")
    n = len(matrix)
    d = len(matrix[0])
    mean_vec = [0.0] * d
    for row in matrix:
        for j in range(d):
            mean_vec[j] += row[j]
    return [val / n for val in mean_vec]


def compute_covariance_matrix(matrix: List[List[float]], sample: bool = True) -> List[List[float]]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix darf nicht leer sein.")
    n = len(matrix)
    d = len(matrix[0])
    if sample and n < 2:
        raise ValueError("Stichproben-Kovarianzmatrix benötigt mindestens 2 Zeilen.")

    means = compute_mean_vector(matrix)
    cov_matrix = [[0.0 for _ in range(d)] for _ in range(d)]
    divisor = (n - 1) if sample else n

    for j in range(d):
        for k in range(j, d):
            cov_sum = sum((matrix[i][j] - means[j]) * (matrix[i][k] - means[k]) for i in range(n))
            val = cov_sum / divisor
            cov_matrix[j][k] = val
            cov_matrix[k][j] = val

    return cov_matrix


def compute_explained_variance_ratio(eigenvalues: List[float]) -> List[float]:
    if not eigenvalues:
        raise ValueError("Eigenwerte dürfen nicht leer sein.")
    total_var = sum(eigenvalues)
    if total_var <= 0:
        raise ValueError("Summe der Eigenwerte muss positiv sein.")
    return [lam / total_var for lam in eigenvalues]


def project_onto_component(
    data_centered: List[List[float]],
    eigenvector: List[float]
) -> List[float]:
    if not data_centered or not eigenvector:
        raise ValueError("Eingaben dürfen nicht leer sein.")
    d = len(eigenvector)
    if len(data_centered[0]) != d:
        raise ValueError("Dimension von Daten und Eigenvektor muss übereinstimmen.")

    projections = []
    for row in data_centered:
        dot = sum(row[j] * eigenvector[j] for j in range(d))
        projections.append(dot)
    return projections


if __name__ == "__main__":
    data = [
        [170.0, 65.0],
        [180.0, 80.0],
        [160.0, 55.0],
        [190.0, 90.0]
    ]
    mu = compute_mean_vector(data)
    cov = compute_covariance_matrix(data)
    print(f"Mittelwertvektor: {mu}")
    print("Kovarianzmatrix:")
    for row in cov:
        print(f"  {[round(x, 2) for x in row]}")
    ratios = compute_explained_variance_ratio([650.0, 10.0])
    print(f"Erklärte Varianz: {ratios}")
