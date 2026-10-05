import numpy as np


def calculate_eigenpairs(matrix):
    # We find eigen val and vectors of the square matrix

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    return eigenvalues, eigenvectors


def find_eigenvector_for_one(matrix, tolerance=1e-10):
    
    # Find the eigenvalue closest to 1 and its corresponding eigenvector

    eigenvalues, eigenvectors = calculate_eigenpairs(matrix)

    # Find the eigenvalue closest to 1
    index = np.argmin(np.abs(eigenvalues - 1))

    eigenvalue = eigenvalues[index]
    eigenvector = np.real(eigenvectors[:, index])

    if not np.isclose(eigenvalue, 1.0, atol=tolerance):
        raise ValueError(
            f"No eigenvalue sufficiently close to 1 was found. "
            f"Closest eigenvalue: {eigenvalue}"
        )

    return eigenvalue, eigenvector


def normalize_probability_vector(vector):
    """
    Normalize an eigenvector so that its elements sum to 1.
    """

    vector = np.real(vector)

    # Eigenvectors can have an arbitrary sign.
    # Flip it if the sum is negative.
    if np.sum(vector) < 0:
        vector = -vector

    total = np.sum(vector)

    if np.isclose(total, 0):
        raise ValueError("Cannot normalize a zero-sum vector.")

    return vector / total