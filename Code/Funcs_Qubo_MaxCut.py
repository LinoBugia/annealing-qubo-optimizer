"""
Funcs_Qubo_MaxCut — read Gset / Max-Cut instances and convert them to QUBO.

Unlike the MPS import (Funcs_Qubo_MpsImport) this needs NO penalty term:
Max-Cut is natively unconstrained. That is exactly why this problem class
suits the DA better — there is a single energy scale rather than two
annealing against each other.

    cut(x) = sum_{(i,j) in E} w_ij * [x_i != x_j]
           = sum_{(i,j) in E} w_ij * (x_i + x_j - 2 x_i x_j)

    minimise E(x) = -cut(x)

In canonical form E = x^T A x + b^T x + c, with W the weighted adjacency
matrix (symmetric, zero diagonal):

    A = W            since x^T W x = 2 * sum_{i<j} w_ij x_i x_j
    b = -W @ 1       (negative weighted degree per vertex)
    c = 0

    cut value = -E(x)        exactly, with no correction term

File format (Gset, Stanford):
    line 1:  n m
    then:    i j w     with i,j in 1..n  (1-indexed!)
"""

import os

import numpy as np

try:
    from scipy import sparse as sp
except ImportError:
    sp = None


def read_gset(path):
    """
    Gset file -> (W as CSR, n, m). Edges are symmetrised, duplicate entries
    are summed, and self-loops are dropped (they only shift the level and
    carry no meaning in Max-Cut).
    """
    if sp is None:
        raise RuntimeError("scipy is required for the Max-Cut import")
    with open(path, "r") as fh:
        head = fh.readline().split()
        n, m = int(head[0]), int(head[1])
        data = np.loadtxt(fh, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    i = data[:, 0].astype(np.int64) - 1              # 1-indexed -> 0-indexed
    j = data[:, 1].astype(np.int64) - 1
    w = data[:, 2].astype(np.float64)
    keep = i != j
    i, j, w = i[keep], j[keep], w[keep]
    W = sp.csr_matrix((np.concatenate([w, w]),
                       (np.concatenate([i, j]), np.concatenate([j, i]))),
                      shape=(n, n))
    W.sum_duplicates()
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W, n, m


def maxcut_to_qubo(W):
    """W (CSR, symmetric) -> canonical (A, b, c). The cut equals -E."""
    n = W.shape[0]
    b = -np.asarray(W.sum(axis=1)).ravel()
    return W, b, 0.0


def cut_value(x, W):
    """
    Cut value computed directly from the partition, independently of the
    QUBO, so the reported result does not rest on the conversion being right.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    # sum_{i<j} w_ij [x_i != x_j] = 0.5 * sum_{ij} w_ij (x_i + x_j - 2 x_i x_j)
    deg = np.asarray(W.sum(axis=1)).ravel()
    return float(x @ deg - x @ (W @ x))


def load_gset(path):
    """
    File -> (A, b, c, info).

    `info` also carries the graph's component structure, because for Max-Cut
    it changes what the instance actually is. A variable of degree 0 cannot
    affect the cut at all, and two components are two independent problems —
    but a single DA chain accepts one flip per step across the whole vector,
    so every component competes for the same budget. G70 is the extreme case:
    10 000 variables, of which 1 354 appear in no edge, and 1 598 separate
    components. Its nominal size is not its effective one.

    Reported: `components`, `isolated` (degree-0 variables), `largest`
    (biggest component) and `effective_n` = n - isolated.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    W, n, m = read_gset(path)
    A, b, c = maxcut_to_qubo(W)
    deg = np.diff(W.indptr)
    iso = int((deg == 0).sum())
    try:
        from scipy.sparse.csgraph import connected_components
        ncomp, lab = connected_components(W, directed=False)
        largest = int(np.bincount(lab).max())
    except ImportError:                                 # scipy.csgraph optional
        ncomp, largest = -1, -1
    return A, b, c, dict(W=W, n=n, m=m, nnz=int(W.nnz),
                         avg_degree=float(W.nnz) / n,
                         weighted=bool(np.any(W.data != 1.0)),
                         min_degree=int(deg.min()), max_degree=int(deg.max()),
                         components=int(ncomp), isolated=iso,
                         largest=largest, effective_n=n - iso)
