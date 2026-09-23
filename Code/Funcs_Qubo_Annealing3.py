"""
Funcs_Qubo_Annealing3 — class-free base helpers (QUBO only, numpy).

Canonical QUBO form used throughout this library:

    E(x) = x^T A x + b^T x + c ,   x in {0,1}^n

    A : (n,n) symmetric, diagonal = 0  (dense float64/float32 or scipy CSR)
    b : (n,)  linear coefficients
    c : constant

Delta energy for flipping bit k:

    dE_k = s_k * (b_k + 2*(A x)_k) ,   s_k = 1 - 2*x_k

This replaces evalPBF / Eval_Delta_Energy / modificate_update_deltaE of the
reference library ../../annealing-cop-approximator (dict-based there, and of
arbitrary degree). Here, row k of A IS the monomial list of x_k, which is why
the QUBO case needs no pbf_var_dict at all. Degree > 2 cannot be expressed in
this structure — use the reference library for that.

Packed integer keys (compatible with the gp-qubo-rag-indexer):
    quadratic: key = i*n + j  (i <= j)
    linear:    key = n*n + i
"""

import numpy as np

try:
    from scipy import sparse as _sp
except ImportError:                                     # scipy is optional
    _sp = None

# Backend hook: xp = numpy today, CuPy as a drop-in later via set_backend(cupy).
# Modules reach it through qa3.xp (module attribute, so rebinding takes effect).
xp = np


def set_backend(module):
    """Set the array backend (numpy-compatible, e.g. cupy)."""
    global xp
    xp = module


def is_sparse(A) -> bool:
    return _sp is not None and _sp.issparse(A)


# ── Integer key scheme ───────────────────────────────────────────────────────

def pack_key(i: int, j: int, n: int) -> int:
    """Quadratic key i*n+j (i<=j) — identical to the indexer's scheme."""
    if i > j:
        i, j = j, i
    return i * n + j


def pack_key_linear(i: int, n: int) -> int:
    return n * n + i


def unpack_key(key: int, n: int):
    """Decode a packed key -> (i,) if linear, (i,j) if quadratic."""
    if key >= n * n:
        return (key - n * n,)
    return divmod(key, n)


# ── Energy evaluation ────────────────────────────────────────────────────────

def eval_qubo(A, b, c, X):
    """
    E(x) = x^T A x + b^T x + c.

    X: (n,) or (mc,n), binary (int8/float). Returns a scalar or (mc,).
    """
    X = np.asarray(X, dtype=np.float64)
    G = X @ A                                   # (…,n); works for CSR too
    G = np.asarray(G)
    return (X * G).sum(axis=-1) + X @ np.asarray(b, dtype=np.float64) + c


def eval_delta_energy(A, b, X):
    """
    Full delta-E vector: dE_k = (1-2x_k) * (b_k + 2*(A x)_k).

    X: (n,) or (mc,n). Returns the same shape, float64.
    Replaces the loop over Eval_Delta_Energy in the reference library.
    """
    X = np.asarray(X, dtype=np.float64)
    G = np.asarray(X @ A)
    S = 1.0 - 2.0 * X
    return S * (np.asarray(b, dtype=np.float64) + 2.0 * G)


def delta_energy_from_gradient(b, X, G):
    """dE from a cached G = X @ A (the annealer maintains G incrementally)."""
    S = 1.0 - 2.0 * X
    return S * (b + 2.0 * G)


def canonicalize_matrix(M):
    """
    Arbitrary square matrix M with E(x) = x^T M x  ->  (A, b):
    A = symmetric off-diagonal part, b = diagonal (since x_i^2 = x_i).
    """
    M = np.asarray(M, dtype=np.float64)
    A = 0.5 * (M + M.T)
    b = np.diag(A).copy()
    np.fill_diagonal(A, 0.0)
    return A, b


def csr_from_dense(A, threshold: float = 0.0):
    """Dense -> CSR, optionally with a magnitude cutoff (gram_threshold analogue)."""
    if _sp is None:
        raise ImportError("scipy is required for CSR support")
    A = np.asarray(A)
    if threshold > 0.0:
        A = np.where(np.abs(A) >= threshold, A, 0.0)
    return _sp.csr_matrix(A)
