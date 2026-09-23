"""
Funcs_Qubo_MpsImport — read MPS instances (MIPLIB) and convert them to QUBO.

IMPORTANT — scope. A MIPLIB instance is a MIP, not a QUBO. This converter
accepts EXCLUSIVELY pure set-packing instances:

    max  sum_i c_i x_i        (equivalently min -sum_i c_i x_i)
    s.t. sum_{i in S_r} x_i <= 1    for every row r
         x in {0,1}^n

Only this case converts LOSSLESSLY — the constraint becomes a purely
quadratic penalty term, with no slack variables and no encoding:

    E(x) = -sum_i c_i x_i  +  P * sum_{r} sum_{i<j in S_r} x_i x_j

Everything else (>= rows, RHS != 1, coefficients != 1, integer variables with
a bound > 1) is rejected with a clear error rather than silently converted
into something wrong. Set covering (>=) needs slack variables and general
linear constraints additionally need a binary expansion — deliberately NOT
implemented here.

The heart of the conversion: with the constraint matrix M (m x n, 0/1
entries), (M^T M)_ij is the number of rows containing i and j TOGETHER —
exactly the number of violated pairs when both are set to 1. Hence

    A = (P/2) * (M^T M  without its diagonal)  ,  b = -c  ,  c_const = 0

in the library's canonical form (x^T A x counts every pair twice, hence the
P/2). One sparse matmul, no pair enumeration in Python.

Penalty: a 0->1 flip gains at most max(c_i) and costs at least P as soon as
it violates a pair. P > max(c_i) makes any violation unprofitable; the
default is P = 2 * max(c_i).
"""

import gzip
import os

import numpy as np

try:
    from scipy import sparse as sp
except ImportError:                                     # scipy is mandatory here
    sp = None


class MpsError(ValueError):
    """Instance falls outside the supported scope."""


def _open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="latin-1")
    return open(path, "r", encoding="latin-1")


def read_mps(path):
    """
    Read an MPS file (.gz accepted) into a raw structure — no interpretation.

    Returns a dict:
      name        instance name
      obj_row     name of the N row
      row_type    {row name: 'L'|'G'|'E'}
      row_index   {row name: 0..m-1}   (non-N rows only, in file order)
      col_index   {column name: 0..n-1}
      entries     (rows, cols, vals) as lists — constraint matrix in COO
      obj         {column index: coefficient}
      rhs         {row index: value}
      bounds      {column index: [(type, value), ...]}
      integer     set of column indices between INTORG/INTEND
    """
    name, obj_row = "", None
    row_type, row_index, col_index = {}, {}, {}
    rr, cc, vv = [], [], []
    obj, rhs, bounds, integer = {}, {}, {}, set()
    section = None
    int_mode = False

    with _open_maybe_gz(path) as fh:
        for raw in fh:
            if not raw.strip() or raw.startswith("*"):
                continue
            if not raw[0].isspace():                    # section header
                tok = raw.split()
                head = tok[0].upper()
                if head == "NAME":
                    name = tok[1] if len(tok) > 1 else ""
                    section = "NAME"
                elif head in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS",
                              "OBJSENSE", "ENDATA"):
                    section = head
                    if head == "ENDATA":
                        break
                else:
                    section = head                      # unknown -> ignore
                continue

            tok = raw.split()
            if section == "ROWS":
                typ, rname = tok[0].upper(), tok[1]
                if typ == "N":
                    if obj_row is None:                 # first N row is the objective
                        obj_row = rname
                    row_type[rname] = "N"
                else:
                    row_type[rname] = typ
                    row_index[rname] = len(row_index)

            elif section == "COLUMNS":
                if len(tok) > 2 and tok[1] == "'MARKER'":
                    up = raw.upper()
                    if "INTORG" in up:
                        int_mode = True
                    elif "INTEND" in up:
                        int_mode = False
                    continue
                cname = tok[0]
                if cname not in col_index:
                    col_index[cname] = len(col_index)
                ci = col_index[cname]
                if int_mode:
                    integer.add(ci)
                for k in range(1, len(tok) - 1, 2):     # pairs (row, value)
                    rname, val = tok[k], float(tok[k + 1])
                    if rname == obj_row:
                        obj[ci] = obj.get(ci, 0.0) + val
                    elif rname in row_index:
                        rr.append(row_index[rname])
                        cc.append(ci)
                        vv.append(val)

            elif section == "RHS":
                start = 1 if (len(tok) % 2 == 1) else 0   # vector name optional
                for k in range(start, len(tok) - 1, 2):
                    rname, val = tok[k], float(tok[k + 1])
                    if rname in row_index:
                        rhs[row_index[rname]] = val

            elif section == "BOUNDS":
                btype = tok[0].upper()
                cname = tok[2] if len(tok) > 2 else None
                if cname is None or cname not in col_index:
                    continue
                ci = col_index[cname]
                val = float(tok[3]) if len(tok) > 3 else None
                bounds.setdefault(ci, []).append((btype, val))

            elif section == "RANGES":
                raise MpsError("RANGES section is not supported — the instance "
                               "is not pure set packing.")

    return dict(name=name, obj_row=obj_row, row_type=row_type,
                row_index=row_index, col_index=col_index,
                entries=(rr, cc, vv), obj=obj, rhs=rhs,
                bounds=bounds, integer=integer)


def _check_set_packing(mps):
    """Check the scope and return (M, c). Raises MpsError."""
    n = len(mps["col_index"])
    m = len(mps["row_index"])
    if n == 0 or m == 0:
        raise MpsError("empty instance")

    bad = sorted({t for t, _ in
                  ((typ, r) for r, typ in mps["row_type"].items() if typ != "N")}
                 - {"L"})
    if bad:
        raise MpsError(
            "only '<=' rows (L) are set packing; found: %s. "
            "'G' would be set covering (needs slack variables), 'E' set "
            "partitioning." % ", ".join(bad))

    rr, cc, vv = mps["entries"]
    vals = np.asarray(vv, dtype=np.float64)
    if vals.size == 0:
        raise MpsError("no constraint entries")
    if not np.all(vals == 1.0):
        raise MpsError("set packing requires coefficients of 1; found e.g. %r"
                       % np.unique(vals)[:5].tolist())

    rhs = np.zeros(m, dtype=np.float64)
    for ri, val in mps["rhs"].items():
        rhs[ri] = val
    if not np.all(rhs == 1.0):
        raise MpsError("set packing requires an RHS of 1; found e.g. %r"
                       % np.unique(rhs)[:5].tolist())

    # Binarity: UP 1 (or BV) on every variable, and integer
    ub = np.full(n, np.inf)
    for ci, blist in mps["bounds"].items():
        for btype, val in blist:
            if btype == "BV":
                ub[ci] = 1.0
            elif btype == "UP":
                ub[ci] = val
            elif btype in ("FX",):
                ub[ci] = val
    nonbin = int(np.sum(ub != 1.0))
    if nonbin:
        raise MpsError("%d variables are not restricted to {0,1} "
                       "(upper bound != 1)" % nonbin)
    if len(mps["integer"]) not in (0, n):
        raise MpsError("mixed integer/continuous — not supported")

    M = sp.csr_matrix((vals, (np.asarray(rr), np.asarray(cc))), shape=(m, n))
    M.sum_duplicates()

    c = np.zeros(n, dtype=np.float64)
    for ci, val in mps["obj"].items():
        c[ci] = -val            # MPS minimises -sum c_i x_i -> c = gain per x_i
    return M, c


def set_packing_to_qubo(mps, penalty=None):
    """
    Pure set-packing MPS -> canonical (A, b, c_const), with A as CSR.

    penalty: None -> 2 * max(|c_i|). Larger makes violations more expensive
             but spreads the dE scale, which makes the cooling harder to hit.

    Also returned (dict): M (constraint matrix), c (objective gains), the
    penalty, and the pair statistics.
    """
    if sp is None:
        raise RuntimeError("scipy is required for the MPS import")
    M, c = _check_set_packing(mps)
    P = float(penalty) if penalty is not None else 2.0 * float(np.abs(c).max())

    # (M^T M)_ij = number of rows holding i AND j — exactly the violated pairs.
    MtM = (M.T @ M).tocsr()
    MtM.setdiag(0.0)
    MtM.eliminate_zeros()

    A = (0.5 * P) * MtM                 # x^T A x counts every pair twice
    b = -c                              # minimise -sum c_i x_i
    return A, b, 0.0, dict(M=M, c=c, penalty=P, n=M.shape[1], m=M.shape[0],
                           pairs=int(MtM.nnz // 2))


def evaluate_set_packing(x, info):
    """
    Evaluate a state AGAINST THE ORIGINAL PROBLEM, not against the QUBO.

    Returns a dict: selected, violated_rows, feasible, objective
    (objective is the MIPLIB objective value, i.e. -sum c_i x_i when
    minimising).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    row_sum = info["M"] @ x
    violated = int(np.sum(row_sum > 1.0 + 1e-9))
    selected = float(info["c"] @ x)
    return dict(selected=selected,
                violated_rows=violated,
                feasible=violated == 0,
                objective=-selected)


def repair_set_packing(x, info, order=None):
    """
    Make a state feasible by greedily switching variables off in violated
    rows, then fill up any free variables again.

    The annealer may visit infeasible states on the way — but the result of
    the optimisation has to be measured against the original problem. Without
    repair, an infeasible state with many variables set would look wrongly
    "better" than the best feasible one.
    """
    x = np.asarray(x, dtype=np.int8).copy()
    M = info["M"].tocsr()
    Mc = M.tocsc()
    if order is None:                                   # drop the cheapest first
        order = np.argsort(-(Mc.getnnz(axis=0)))

    while True:
        row_sum = M @ x
        bad = np.flatnonzero(row_sum > 1)
        if bad.size == 0:
            break
        # in every violated row, flip off the "most expensive" set variable
        for r in bad:
            lo, hi = M.indptr[r], M.indptr[r + 1]
            members = M.indices[lo:hi]
            on = members[x[members] == 1]
            if on.size <= 1:
                continue
            rank = {v: k for k, v in enumerate(order)}
            worst = max(on.tolist(), key=lambda v: rank[v])
            x[worst] = 0

    # fill up: any free variable that does not break a row
    for i in order[::-1]:
        if x[i]:
            continue
        x[i] = 1
        rows = Mc.indices[Mc.indptr[i]:Mc.indptr[i + 1]]
        if np.any((M @ x)[rows] > 1):
            x[i] = 0
    return x


def load_set_packing(path, penalty=None):
    """Convenience: file -> (A, b, c_const, info)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return set_packing_to_qubo(read_mps(path), penalty=penalty)
