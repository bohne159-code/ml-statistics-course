"""
ÜBUNG 09: Multivariate Statistik, Kovarianzmatrix & PCA
========================================================

Warum braucht das ein AI/ML Engineer?
-------------------------------------
In der realen Welt arbeiten ML-Modelle nicht mit einzelnen Zahlen, sondern mit hochdimensionalen Vektoren
(z.B. 768-dimensionale Text-Embeddings, 2048-dimensionale Bild-Features).
1. Kovarianzmatrix Sigma:
   - Eine D x D Matrix, die auf der Diagonale die Varianzen der einzelnen Merkmale und auf den Nebendiagonalen
     die paarweisen Kovarianzen enthält.
   - Bildet das Herzstück von Gaußschen Prozessen, Kalman-Filtern und multivariaten Normalverteilungen N(mu, Sigma).
2. Mahalanobis-Distanz:
   - Misst den statistischen Abstand eines Punktes zum Verteilungszentrum unter Berücksichtigung von Korrelationen.
   - Der euklidische Abstand ignoriert, dass Merkmale korreliert sein können und unterschiedliche Skalen haben.
   - Wichtigstes Werkzeug für multivariate Ausreißer-Erkennung (Multivariate Outlier Detection).
3. Hauptkomponentenanalyse (PCA - Principal Component Analysis):
   - Eine der wichtigsten Techniken für Dimensionsreduktion und Feature Extraction.
   - Das statistische Ziel von PCA ist es, orthogonale Richtungen zu finden, die die **maximale Varianz**
     der Daten erklären!
   - Mathematisch: Die Hauptkomponenten sind exakt die Eigenvektoren der Kovarianzmatrix,
     und die Eigenwerte lambda_i geben an, wie viel Datenvarianz in dieser Richtung steckt.

Deine Aufgabe:
--------------
Implementiere die Berechnung des Mittelwertvektors, der Kovarianzmatrix, der erklärten Varianz
und die Projektion auf eine Hauptkomponente.
"""

from typing import List, Tuple


def compute_mean_vector(matrix: List[List[float]]) -> List[float]:
    """
    Berechnet den Mittelwertvektor mu für einen Datensatz aus N Datenpunkten mit D Merkmalen.
    matrix hat Dimension (N, D).
    Rückgabe: Liste der Länge D mit den Spaltenmitteln.
    """
    if not matrix or not matrix[0]:
        raise ValueError("Matrix darf nicht leer sein.")
    n = len(matrix)
    d = len(matrix[0])
    # TODO: Berechne den Mittelwert für jede der d Spalten
    raise NotImplementedError("Bitte implementieren!")


def compute_covariance_matrix(matrix: List[List[float]], sample: bool = True) -> List[List[float]]:
    """
    Berechnet die (D x D) Stichproben-Kovarianzmatrix Sigma.
    
    Formel für Element (j, k):
      Sigma_jk = (1 / (N - 1)) * sum_i=1..N (X_ij - mu_j) * (X_ik - mu_k)
      
    Eigenschaften:
      - Symmetrisch: Sigma_jk == Sigma_kj
      - Diagonale Sigma_jj ist die Varianz von Merkmal j.
    """
    if not matrix or not matrix[0]:
        raise ValueError("Matrix darf nicht leer sein.")
    n = len(matrix)
    d = len(matrix[0])
    if sample and n < 2:
        raise ValueError("Stichproben-Kovarianzmatrix benötigt mindestens 2 Zeilen.")

    # TODO: Berechne den Mittelwertvektor und anschließend die d x d Kovarianzmatrix
    raise NotImplementedError("Bitte implementieren!")


def compute_explained_variance_ratio(eigenvalues: List[float]) -> List[float]:
    """
    Berechnet den Anteil der erklärten Varianz (Explained Variance Ratio) für gegebene Eigenwerte.
    In PCA entspricht jeder Eigenwert lambda_i der Varianz entlang der i-ten Hauptkomponente.
    
    Formel:
      ratio_i = lambda_i / sum(eigenvalues)
      
    Rückgabe: Liste der Anteile (summiert sich zu 1.0).
    """
    if not eigenvalues:
        raise ValueError("Eigenwerte dürfen nicht leer sein.")
    total_var = sum(eigenvalues)
    if total_var <= 0:
        raise ValueError("Summe der Eigenwerte muss positiv sein.")

    # TODO: Berechne das Verhältnis für jeden Eigenwert
    raise NotImplementedError("Bitte implementieren!")


def project_onto_component(
    data_centered: List[List[float]],
    eigenvector: List[float]
) -> List[float]:
    """
    Projiziert zentrierte Datenpunkte (N x D) auf einen Hauptkomponenten-Eigenvektor v (Länge D).
    
    Für jeden Datenpunkt x_i:
      z_i = dot_product(x_i, v) = sum_j (x_ij * v_j)
      
    Rückgabe: 1D-Liste der projizierten Skalare z_i (Länge N).
    """
    if not data_centered or not eigenvector:
        raise ValueError("Eingaben dürfen nicht leer sein.")
    d = len(eigenvector)
    if len(data_centered[0]) != d:
        raise ValueError("Dimension von Daten und Eigenvektor muss übereinstimmen.")

    # TODO: Berechne das Skalarprodukt für jede Zeile
    raise NotImplementedError("Bitte implementieren!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Uebung 09 - Multivariate Statistik & PCA")
    print("=" * 60)
    data = [
        [170.0, 65.0],
        [180.0, 80.0],
        [160.0, 55.0],
        [190.0, 90.0]
    ]

    # 1. Mean Vector
    try:
        mu = compute_mean_vector(data)
        print(f"  [OK]   compute_mean_vector:      mu = {mu}")
    except NotImplementedError as e:
        print(f"  [TODO] compute_mean_vector:      {e}")
    except Exception as e:
        print(f"  [FAIL] compute_mean_vector:      {e}")

    # 2. Covariance Matrix
    try:
        cov = compute_covariance_matrix(data)
        print(f"  [OK]   compute_covariance_matrix: {len(cov)}x{len(cov[0])} Matrix")
    except NotImplementedError as e:
        print(f"  [TODO] compute_covariance_matrix:{e}")
    except Exception as e:
        print(f"  [FAIL] compute_covariance_matrix:{e}")

    # 3. Explained Variance Ratio
    try:
        ratios = compute_explained_variance_ratio([650.0, 10.0])
        print(f"  [OK]   compute_explained_var:    PC1 = {ratios[0]*100:.1f}%, PC2 = {ratios[1]*100:.1f}%")
    except NotImplementedError as e:
        print(f"  [TODO] compute_explained_var:    {e}")
    except Exception as e:
        print(f"  [FAIL] compute_explained_var:    {e}")

    # 4. Project onto Component
    try:
        centered = [[-10.0, -5.0], [10.0, 5.0]]
        v = [0.894, 0.447]
        proj = project_onto_component(centered, v)
        print(f"  [OK]   project_onto_component:   Projektion = {proj}")
    except NotImplementedError as e:
        print(f"  [TODO] project_onto_component:   {e}")
    except Exception as e:
        print(f"  [FAIL] project_onto_component:   {e}")
    print("=" * 60)
