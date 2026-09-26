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


def delta_e_sigma(A, b, exact: bool = True, samples: int = 8, seed: int = 0):
    """
    sigma = std of dE over uniformly random states AND uniformly random flips.

    This is the natural unit of a QUBO's energy scale. Temperatures expressed
    as multiples of sigma transfer between instances; absolute temperatures do
    not, because they carry the scale of the coefficients.

    Exact, via Walsh coefficients. With x_i = (1+s_i)/2 and chi_U = prod_{i in
    U} s_i, a pseudo-Boolean function E(x) = sum_S a_S prod_{i in S} x_i has

        ehat(U) = sum_{S superset U} a_S * 2^(-|S|)

    Flipping bit k negates chi_U exactly when k is in U, so
    dE_k(s) = -2 * sum_{U containing k} ehat(U) chi_U, and since the chi_U are
    orthonormal under the uniform measure,

        E[dE_k^2] = 4 * sum_{U containing k} ehat(U)^2
        sigma^2   = (1/n) sum_k E[dE_k^2] = (4/n) * sum_U |U| * ehat(U)^2

    The mean of dE over directed edges is zero by symmetry (the reverse of
    every edge carries -dE), so sigma^2 is the raw second moment.

    For a QUBO the sum collapses to degree 2. With Asym = A + A^T:

        ehat_i  = b_i/2 + rowsum(Asym)_i / 4
        ehat_ij = Asym[i,j] / 4
        sigma^2 = (4/n) * [ sum_i ehat_i^2 + sum_{i != j} Asym[i,j]^2 / 16 ]

    Cost is one row sum and one sum of squares over the nonzeros — linear in
    the number of monomials, and it never touches the constant term c.

    exact=False uses the unbiased estimator sigma^2 ~ mean_k dE_k(x)^2 over
    `samples` uniform random states. It exists as an independent control: the
    two agree to ~1/sqrt(samples*n) and disagreeing means the Walsh path has a
    bug. Test_Sigma_Schedule.py checks both against brute force for n <= 10.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    if n == 0:
        return 0.0

    if not exact:
        rng = np.random.default_rng(seed)
        X = (rng.random((samples, n)) < 0.5).astype(np.int8)
        dE = eval_delta_energy(A, b, X)
        return float(np.sqrt((dE.astype(np.float64) ** 2).mean()))

    if is_sparse(A):
        Asym = (A + A.T).tocsr()
        row = np.asarray(Asym.sum(axis=1)).ravel()
        sq_off = float(Asym.multiply(Asym).sum())       # diagonal is zero
    else:
        Ad = np.asarray(A, dtype=np.float64)
        Asym = Ad + Ad.T
        row = Asym.sum(axis=1)
        sq_off = float((Asym ** 2).sum() - (np.diag(Asym) ** 2).sum())

    ehat_lin = 0.5 * b + 0.25 * row
    sigma2 = (4.0 / n) * (float((ehat_lin ** 2).sum()) + sq_off / 16.0)
    return float(np.sqrt(max(sigma2, 0.0)))
