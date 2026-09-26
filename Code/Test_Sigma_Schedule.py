"""
Test_Sigma_Schedule — correctness of delta_e_sigma and the "sigma" schedule.

sigma = std(dE) over uniformly random states and uniformly random flips. It is
computed in closed form from the Walsh coefficients, which is linear in the
number of monomials. Since that derivation is easy to get subtly wrong, it is
checked here against a brute force that enumerates EVERY directed edge of the
hypercube — all n * 2^n of them.

That enumeration only runs for n <= 10 (10 * 1024 = 10 240 edges, milliseconds).
It is a correctness oracle, never a code path any real instance touches: for an
11 811-variable instance the exact formula is one row sum and one sum of squares
over the nonzeros, and the timing check at the bottom shows what that costs.

Usage:  python3 Test_Sigma_Schedule.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Funcs_Qubo_Annealing3 as qa3
from Funcs_Qubo_TempSchedules import generate_cooling_schedule

try:
    from scipy import sparse as sp
except ImportError:
    sp = None


def sigma_bruteforce(A, b):
    """
    std(dE) over all n * 2^n directed edges, by enumeration. n <= 10 only.

    The mean over directed edges is exactly zero — every edge's reverse
    carries -dE — so this returns the raw second moment's square root, which
    is what delta_e_sigma computes.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    if n > 10:
        raise ValueError("brute force is for n <= 10, got %d" % n)
    Ad = A.toarray() if (sp is not None and sp.issparse(A)) else np.asarray(A)
    states = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
    dE = (1.0 - 2.0 * states) * (b + 2.0 * (states @ Ad))      # (2^n, n)
    assert abs(dE.mean()) < 1e-9, "mean over directed edges must vanish"
    return float(np.sqrt((dE ** 2).mean()))


def random_qubo(n, rng, density=0.6, scale=1.0):
    """Symmetric A with zero diagonal, plus a linear term."""
    M = rng.normal(0.0, scale, (n, n))
    M *= (rng.random((n, n)) < density)
    A = np.triu(M, 1)
    A = A + A.T
    return A, rng.normal(0.0, scale, n)


def main():
    rng = np.random.default_rng(7)
    ok = True

    print("1. exact Walsh sigma vs brute force over all n*2^n directed edges")
    print("   n   edges    exact         brute force   |diff|")
    for n in range(2, 11):
        for trial in range(3):
            scale = [1.0, 1e-3, 1e4][trial]           # scale must not matter
            A, b = random_qubo(n, rng, scale=scale)
            ex = qa3.delta_e_sigma(A, b)
            bf = sigma_bruteforce(A, b)
            d = abs(ex - bf)
            rel = d / max(bf, 1e-300)
            if trial == 0:
                print("  %2d  %6d   %-12.6g  %-12.6g  %.2e"
                      % (n, n * 2 ** n, ex, bf, d))
            if rel > 1e-9:
                print("     FAIL at n=%d scale=%g: rel=%.3e" % (n, scale, rel))
                ok = False

    print("\n2. same, sparse CSR input (the path real instances take)")
    if sp is None:
        print("   scipy missing, skipped")
    else:
        for n in (6, 9, 10):
            A, b = random_qubo(n, rng)
            ex = qa3.delta_e_sigma(sp.csr_matrix(A), b)
            bf = sigma_bruteforce(A, b)
            print("   n=%-3d dense %.9f  csr %.9f  |diff| %.2e"
                  % (n, bf, ex, abs(ex - bf)))
            if abs(ex - bf) > 1e-9:
                ok = False

    print("\n3. edge cases")
    A0 = np.zeros((5, 5))
    cases = [("A=0, b=0", A0, np.zeros(5)),
             ("A=0, b!=0", A0, np.arange(5, dtype=float)),
             ("A!=0, b=0", random_qubo(5, rng)[0], np.zeros(5))]
    for name, A, b in cases:
        ex, bf = qa3.delta_e_sigma(A, b), sigma_bruteforce(A, b)
        print("   %-12s exact %.9f  brute %.9f" % (name, ex, bf))
        if abs(ex - bf) > 1e-9:
            ok = False

    print("\n4. sampling estimator (the independent control) vs exact")
    for n, s in ((200, 4), (200, 64), (2000, 64)):
        A, b = random_qubo(n, rng, density=0.1)
        ex = qa3.delta_e_sigma(A, b)
        est = qa3.delta_e_sigma(A, b, exact=False, samples=s, seed=1)
        print("   n=%-5d samples=%-3d exact %.5f  sampled %.5f  rel %.4f  "
              "(expected ~%.4f)"
              % (n, s, ex, est, abs(est - ex) / ex, 1.0 / np.sqrt(s * n)))
        if abs(est - ex) / ex > 10.0 / np.sqrt(s * n):
            print("     FAIL: estimator off by far more than its own noise")
            ok = False

    print("\n5. cost of the exact path at real instance size")
    if sp is not None:
        for n, deg in ((11811, 210), (20000, 4)):
            rows = np.repeat(np.arange(n), deg)
            cols = rng.integers(0, n, n * deg)
            vals = rng.normal(0.0, 1.0, n * deg)
            M = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            M = (M + M.T) * 0.5
            M.setdiag(0.0)
            M.eliminate_zeros()
            bb = rng.normal(0.0, 1.0, n)
            t0 = time.perf_counter()
            s = qa3.delta_e_sigma(M, bb)
            dt = time.perf_counter() - t0
            print("   n=%-6d nnz=%-9d sigma=%.4f   %.1f ms"
                  % (n, M.nnz, s, 1e3 * dt))

    print("\n6. sigma schedule shape")
    sig = 2.0
    for form in ("geometric", "staircase", "linear"):
        T = generate_cooling_schedule(["sigma", 0.5, 0.05, form, sig], 1000)
        lev = len(np.unique(np.round(T, 12)))
        print("   %-10s T[0]=%.4f (%.2f sigma)  T[-1]=%.4f (%.3f sigma)  "
              "levels=%d  monotone=%s"
              % (form, T[0], T[0] / sig, T[-1], T[-1] / sig, lev,
                 bool(np.all(np.diff(T) <= 1e-12))))
        if abs(T[0] - 0.5 * sig) > 1e-9 or abs(T[-1] - 0.05 * sig) > 1e-9:
            print("     FAIL: endpoints not pinned to c_start/c_end * sigma")
            ok = False
        if not np.all(np.diff(T) <= 1e-12):
            ok = False

    print("\n%s" % ("ALL OK" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
