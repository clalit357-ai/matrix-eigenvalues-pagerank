import numpy as np

from eigenvalues import (
    find_eigenvector_for_one,
    normalize_probability_vector
)


def build_transition_matrix(nodes, edges):
    """
    Build a column-stochastic transition matrix.

    M[target, source] = probability of moving
    from source to target.
    """

    n = len(nodes)

    # Gives every node an index
    index = {node: i for i, node in enumerate(nodes)}

    # Start with an n x n zero matrix
    M = np.zeros((n, n), dtype=float)

    # Count outgoing links from every node
    outgoing_count = {node: 0 for node in nodes}

    for source, target in edges:
        outgoing_count[source] += 1

    # Fill transition probabilities
    for source, target in edges:

        i = index[target]
        j = index[source]

        M[i, j] = 1 / outgoing_count[source]

    # Handle dangling nodes
    for node in nodes:

        j = index[node]

        if outgoing_count[node] == 0:
            M[:, j] = 1 / n

    return M


def build_google_matrix(transition_matrix, damping_factor=0.85):
    """
    Build the Google matrix.

    G = dM + (1-d)/N
    """

    n = transition_matrix.shape[0]

    teleportation = np.ones((n, n)) / n

    G = (
        damping_factor * transition_matrix
        + (1 - damping_factor) * teleportation
    )

    return G


def pagerank_by_eigenvector(google_matrix):
    """
    Calculate PageRank using the eigenvector corresponding
    to eigenvalue 1.
    """

    eigenvalue, eigenvector = find_eigenvector_for_one(
        google_matrix
    )

    pagerank = normalize_probability_vector(eigenvector)

    return eigenvalue, pagerank


def verify_pagerank(google_matrix, pagerank, tolerance=1e-10):
    """
    Verify that:

        G @ PageRank ≈ PageRank
    """

    result = google_matrix @ pagerank

    error = np.linalg.norm(result - pagerank)

    return error, error < tolerance


if __name__ == "__main__":

    # Example graph
    nodes = ["A", "B", "C", "D"]

    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("C", "A"),
        ("C", "D"),
        ("D", "A")
    ]

    # Step 1: Transition matrix
    M = build_transition_matrix(nodes, edges)

    print("Transition Matrix:")
    print(M)

    # Step 2: Google matrix
    G = build_google_matrix(
        M,
        damping_factor=0.85
    )

    print("\nGoogle Matrix:")
    print(G)

    # Step 3: Eigenvector PageRank
    eigenvalue, pagerank = pagerank_by_eigenvector(G)

    print("\nEigenvalue closest to 1:")
    print(eigenvalue)

    print("\nPageRank:")

    for node, score in zip(nodes, pagerank):
        print(f"{node}: {score:.6f}")

    # Step 4: Check that PageRank sums to 1
    print("\nPageRank sum:")
    print(np.sum(pagerank))

    # Step 5: Verify eigenvector equation
    error, verified = verify_pagerank(
        G,
        pagerank
    )

    print("\nVerification error:")
    print(error)

    print("\nPageRank eigenvector verified?")
    print(verified)