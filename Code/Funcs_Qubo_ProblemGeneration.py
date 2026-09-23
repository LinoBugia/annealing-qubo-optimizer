"""
Funcs_Qubo_ProblemGeneration — problem generators and converters (QUBO only).

Array versions of the generators in the reference library
(../../annealing-cop-approximator/Code/Funcs_pbfGenerators.py and
Funcs_Annealing2.createPoly). Every generator returns the canonical form
(A, b, c) — see Funcs_Qubo_Annealing3.

The converters qubo_from_pbf / pbf_from_qubo bridge to the reference library
(tuple keys) and to the gp-qubo-rag-indexer (packed integer keys).
"""

import random
from itertools import combinations

import numpy as np

from Funcs_Qubo_Annealing3 import canonicalize_matrix, unpack_key


# ── Random QUBOs (mirror of createPoly, degree 2) ────────────────────────────

def create_random_qubo(variables: int, density: float = 1.0, seed=42):
    """
    Random QUBO with the same coefficient distribution as the reference's
    createPoly(variables, 2, density, seed): random()*uniform(1,256) +
    random(), positive values only. The same seed gives exactly the same
    coefficients as the reference library (same draw order: all linear terms
    first, then all quadratic ones).
    """
    random.seed(seed)
    b = np.zeros(variables)
    A = np.zeros((variables, variables))
    varList = list(range(variables))
    for (i,) in combinations(varList, 1):
        if random.random() < density:
            b[i] = random.random() * random.uniform(1, 256) + random.random()
    for (i, j) in combinations(varList, 2):
        if random.random() < density:
            v = random.random() * random.uniform(1, 256) + random.random()
            A[i, j] = A[j, i] = 0.5 * v          # x^T A x counts (i,j) twice
    return A, b, 0.0


def create_random_qubo_signed(variables: int, density: float = 1.0, seed=42):
    """Mirror of createPoly_negative (random signs, spin-glass-like)."""
    random.seed(seed)
    b = np.zeros(variables)
    A = np.zeros((variables, variables))
    varList = list(range(variables))
    for i in range(1, 3):
        for m in combinations(varList, i):
            if random.random() < density:
                sign = 1 if random.random() < 0.5 else -1
                v = sign * (0.5 * random.random() * random.uniform(1, 256) + random.random())
                if i == 1:
                    b[m[0]] = v
                else:
                    A[m[0], m[1]] = A[m[1], m[0]] = 0.5 * v
    return A, b, 0.0


# ── Number partitioning (mirror of GenerateNumberPartitioningpbf) ────────────

def create_number_partitioning_qubo(numbers):
    numbers = np.asarray(numbers, dtype=np.float64)
    n = len(numbers)
    total = numbers.sum()
    c = (total ** 2) / 2.0
    b = -4.0 * numbers * (total - numbers)
    M = 8.0 * np.outer(numbers, numbers)        # pbf[(i,j)] = 8*n_i*n_j, i<j
    A = 0.5 * M                                 # x^T A x counts both orders
    np.fill_diagonal(A, 0.0)
    return A, b, c


# ── Graph binary clustering (mirror of GenerateGraphBinaryClustering_pbf) ────

def normalize_coords(coords):
    """Centre, then scale to mean L2 norm 1 (as in the reference / indexer)."""
    x = np.asarray(coords, dtype=np.float64)
    x = x - x.mean(axis=0)
    mag = np.linalg.norm(x, axis=1).sum()
    if mag > 0:
        x = x * len(x) / mag
    return x


def create_graph_binary_clustering_qubo(coords, normalize: bool = True):
    """
    Gram-based bisection QUBO:
        reference pbf form: linear Q_row[i]*Q_col[i], quadratic -2*Q_ij
        (all pairs incl. i=j and both orders)
    ->  b_i = Q_row[i]*Q_col[i] - 2*Q_ii ,  A = -(Q + Q^T) with zero diagonal.
    """
    x = normalize_coords(coords) if normalize else np.asarray(coords, dtype=np.float64)
    Q = x @ x.T
    Q_row = Q.sum(axis=0)
    Q_col = Q.sum(axis=1)
    b = Q_row * Q_col - 2.0 * np.diag(Q)
    A = -(Q + Q.T)
    np.fill_diagonal(A, 0.0)
    return A, b, 0.0


def lloyd_bisect(coords, iterations: int = 10, seed=None, start=None):
    """
    k=2 Lloyd as a warm-start generator (as in the gp-qubo-rag-indexer):
    random binary start, empty sides are repaired, NO balance repair (that
    would destroy the start values).
    Returns: (assign (n,), iterations, converged)
    """
    x = np.asarray(coords, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    assign = np.asarray(start, dtype=np.int8) if start is not None \
        else rng.integers(0, 2, size=n).astype(np.int8)
    it_used, converged = 0, False
    for it in range(iterations):
        it_used = it + 1
        for side in (0, 1):                      # repair an empty side
            if not (assign == side).any():
                assign[rng.integers(0, n)] = side
        c0 = x[assign == 0].mean(axis=0)
        c1 = x[assign == 1].mean(axis=0)
        d0 = ((x - c0) ** 2).sum(axis=1)
        d1 = ((x - c1) ** 2).sum(axis=1)
        new = (d1 < d0).astype(np.int8)
        if (new == assign).all():
            converged = True
            break
        assign = new
    return assign, it_used, converged


# ── Start states ─────────────────────────────────────────────────────────────

def random_start_states(n: int, count: int = 1, seed=42):
    """`count` random binary vectors (count,n) int8 — replaces
    getInitialVarAssignement of the reference library."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(count, n)).astype(np.int8)


# ── Converters: pbf (dict) <-> (A, b, c) ─────────────────────────────────────

def qubo_from_pbf(pbf: dict, n: int):
    """
    pbf dict -> (A, b, c). Accepts BOTH key formats:
      - tuple keys of the reference library:  (), (i,), (i,j)
      - packed integer keys of the indexer:   i*n+j (i<=j), n*n+i
    Quadratic keys with i==j are booked as linear (since x_i^2 = x_i).
    Degree > 2 raises ValueError (use the reference library for that).
    """
    A = np.zeros((n, n))
    b = np.zeros(n)
    c = 0.0
    for key, val in pbf.items():
        if isinstance(key, tuple):
            mon = key
        else:
            mon = unpack_key(int(key), n)
        if len(mon) == 0:
            c += val
        elif len(mon) == 1:
            b[mon[0]] += val
        elif len(mon) == 2:
            i, j = mon
            if i == j:
                b[i] += val
            else:
                A[i, j] += 0.5 * val
                A[j, i] += 0.5 * val
        else:
            raise ValueError("qubo_from_pbf: degree > 2 (%r) — use the reference library" % (mon,))
    return A, b, c


def qubo_from_matrix(M):
    """E(x) = x^T M x for an arbitrary square M -> canonical (A, b)."""
    return canonicalize_matrix(M)


def pbf_from_qubo(A, b, c=0.0, int_keys: bool = False, tol: float = 0.0):
    """
    (A, b, c) -> pbf dict for the reference library (tuple keys) or for the
    indexer (int_keys=True, packed scheme). Quadratic coefficient (i<j) = 2*A_ij.
    """
    A = np.asarray(A)
    b = np.asarray(b)
    n = len(b)
    pbf = {}
    if c != 0.0:
        if int_keys:
            raise ValueError("a constant cannot be represented in the int-key scheme")
        pbf[()] = float(c)
    for i in range(n):
        if abs(b[i]) > tol:
            pbf[n * n + i if int_keys else (i,)] = float(b[i])
    ii, jj = np.nonzero(np.triu(np.abs(A) > tol, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        pbf[i * n + j if int_keys else (i, j)] = float(2.0 * A[i, j])
    return pbf
