"""
pagerank.py : transition matrix -> Google matrix -> power iteration.

Convention: M and G are COLUMN-stochastic, PageRank is a column vector,
and the fixed point equation is  PR = G @ PR.
"""
import numpy as np


def transition_matrix(A, fix_dangling=True):
    """Adjacency A (A[i,j]=1 if i->j)  ->  column-stochastic M.
    M[i, j] = P(j -> i) = A[j, i] / outdeg(j).
    Dangling pages (outdeg 0) get a uniform column 1/N (surfer jumps anywhere).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    out_deg = A.sum(axis=1)                     # out-links of each page
    M = np.zeros((n, n))
    nz = out_deg > 0
    M[:, nz] = (A[nz, :] / out_deg[nz, None]).T  # normalize rows of A, then transpose
    if fix_dangling:
        M[:, ~nz] = 1.0 / n
    return M


def google_matrix(M, d=0.85):
    """G = d*M + (1-d)/N * E,  E = all-ones matrix. Every column still sums to 1."""
    if not 0 < d < 1:
        raise ValueError("damping factor d must be in (0, 1)")
    n = M.shape[0]
    return d * M + (1 - d) / n * np.ones((n, n))


def power_iteration(G, tol=1e-10, max_iter=1000):
    """Iterate PR <- G @ PR from the uniform vector until the L1 change < tol.
    Returns (pr, history) where history[k] = ||PR_{k+1} - PR_k||_1."""
    n = G.shape[0]
    pr = np.full(n, 1.0 / n)
    history = []
    for _ in range(max_iter):
        new = G @ pr
        err = np.abs(new - pr).sum()
        history.append(err)
        pr = new
        if err < tol:
            break
    else:
        print(f"Warning: no convergence in {max_iter} iterations (err={err:.2e})")
    return pr / pr.sum(), history


def pagerank(A, d=0.85, tol=1e-10, max_iter=1000):
    """Full pipeline from adjacency matrix. Returns (pr, history, G)."""
    M = transition_matrix(A)
    G = google_matrix(M, d)
    pr, history = power_iteration(G, tol, max_iter)
    return pr, history, G


def eigen_check(G, pr):
    """Quick sanity checks (details/eigenvector method belong to eigenvalues.py)."""
    return {
        "sums_to_1": bool(np.isclose(pr.sum(), 1.0)),
        "columns_sum_to_1": bool(np.allclose(G.sum(axis=0), 1.0)),
        "residual ||G@pr - pr||_1": float(np.abs(G @ pr - pr).sum()),
    }
