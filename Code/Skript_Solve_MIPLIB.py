"""
Skript_Solve_MIPLIB — load a MIPLIB set-packing instance, solve it, measure honestly.

Flow:
  1. fetch the instance (locally or from miplib.zib.de) and convert to QUBO
  2. greedy baseline (variables by ascending conflict degree)
  3. optional: temperature probe — short runs at constant T
  4. main run starting from the greedy solution
  5. evaluate AGAINST THE ORIGINAL PROBLEM (feasibility + objective), not
     against the QUBO

Why the temperature is so small: the DA tests ALL n flips per step and
accepts at most one, so the per-flip acceptance has to be p/n, not p. For a
median uphill barrier dE and a target acceptance p:

    T ~ dE / ln(n / p)

At dE=1, n=11811, p=0.3 that gives T ~ 0.095 — not T ~ 1.

Usage:
    python Skript_Solve_MIPLIB.py --instance cdc7-4-3-2 --probe
    python Skript_Solve_MIPLIB.py --instance cdc7-4-3-2 --steps 200000 --mc 16
"""

import argparse
import contextlib
import io
import os
import sys
import time
import shutil
import subprocess
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Funcs_Qubo_Annealing3 as qa3
from Funcs_Qubo_MpsImport import (evaluate_set_packing, load_set_packing,
                                  repair_set_packing)
from Funcs_Qubo_Optimizers import qubo_min_solver

MIPLIB_URL = "https://miplib.zib.de/WebData/instances/%s.mps.gz"
INSTANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Instances")

# Best-known objective values (MIPLIB). The instance is OPEN — there is no
# proven optimum, so beating it is not ruled out, but the value comes from a
# specialised solver.
BEST_KNOWN = {"cdc7-4-3-2": -307.0}


def fetch(instance):
    """Keep the instance locally; download only if it is not there yet."""
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    path = os.path.join(INSTANCE_DIR, instance + ".mps.gz")
    if not os.path.exists(path):
        url = MIPLIB_URL % instance
        print("downloading %s ..." % url)
        if shutil.which("curl"):        # urllib fails here on the certificate chain
            subprocess.check_call(["curl", "-sSL", "-o", path, url])
        else:
            urllib.request.urlretrieve(url, path)
    print("instance: %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
    return path


def greedy_packing(A, info):
    """
    Greedy: variables by ascending number of conflict partners, taking
    whatever fits. Yields a MAXIMAL (not maximum) feasible packing — and
    therefore a strict local optimum in the single-flip neighbourhood: from
    here EVERY individual flip is uphill.
    """
    n = info["n"]
    M = info["M"].tocsr()
    Mc = M.tocsc()
    order = np.argsort(np.diff(A.indptr), kind="stable")
    x = np.zeros(n, dtype=np.int8)
    rowcnt = np.zeros(M.shape[0], dtype=np.int32)
    for i in order:
        rows = Mc.indices[Mc.indptr[i]:Mc.indptr[i + 1]]
        if np.all(rowcnt[rows] == 0):
            x[i] = 1
            rowcnt[rows] += 1
    return x


def run(A, b, cc, info, x0, cooling, steps, mc, seed=11, recompute_every=4096):
    """One run; returns (best_state, selected_count, feasible, seconds)."""
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        Xm, Mins, _, _ = qubo_min_solver(
            A, b, cc, steps=steps, num_MC=mc, cooling_param=list(cooling),
            initial_varAssignements_pre=x0, save_addinfo=True,
            seed_rand=seed, recompute_every=recompute_every)
    dt = time.perf_counter() - t0
    k = int(np.argmin(Mins))
    x = np.asarray(Xm[k], dtype=np.int8)
    ev = evaluate_set_packing(x, info)
    if not ev["feasible"]:                      # with P>max(c) this should never trigger
        x = repair_set_packing(x, info)
        ev = evaluate_set_packing(x, info)
    return x, int(ev["selected"]), ev["feasible"], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="cdc7-4-3-2")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--mc", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.18)
    ap.add_argument("--penalty", type=float, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="short runs over a T grid instead of the main run")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    path = fetch(args.instance)
    t0 = time.perf_counter()
    A, b, cc, info = load_set_packing(path, penalty=args.penalty)
    n = info["n"]
    mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6
    print("QUBO: n=%d | conflict pairs=%d | A.nnz=%d (%.1f MB CSR, dense would "
          "be %.1f GB) | penalty=%g | %.1f s"
          % (n, info["pairs"], A.nnz, mb, n * n * 8 / 1e9, info["penalty"],
             time.perf_counter() - t0))
    print("mc*n = %d  (efficiency plateau per the benchmark: 1e5 .. 2.5e5)"
          % (args.mc * n))

    xg = greedy_packing(A, info)
    evg = evaluate_set_packing(xg, info)
    bk = BEST_KNOWN.get(args.instance)
    print("\ngreedy: %d selected, feasible=%s%s"
          % (evg["selected"], evg["feasible"],
             "" if bk is None else "  | best known %d, gap %d"
             % (-bk, -bk - evg["selected"])))

    dE = qa3.eval_delta_energy(A, b, xg)
    up = dE[dE > 0]
    print("dE at the greedy start: all %d flips uphill, q50=%.0f -> "
          "T ~ q50/ln(n/0.3) = %.3f"
          % (up.size, np.quantile(up, .5), np.quantile(up, .5) / np.log(n / 0.3)))

    if args.probe:
        print("\nprobe (4000 steps, num_MC=%d, start=greedy):" % args.mc)
        print("  T       selected  feasible   s")
        for T in (0.03, 0.06, 0.10, 0.15, 0.18, 0.22, 0.30, 0.50):
            _, sel, feas, dt = run(A, b, cc, info, xg, ["constant", T],
                                   4000, args.mc, args.seed)
            print("  %-6.2f  %5d    %-8s  %5.1f" % (T, sel, feas, dt))
        return

    print("\nmain run: %d steps, num_MC=%d, constant T=%g"
          % (args.steps, args.mc, args.temp))
    x, sel, feas, dt = run(A, b, cc, info, xg, ["constant", args.temp],
                           args.steps, args.mc, args.seed)
    print("result: %d selected, feasible=%s, %.1f s (%.2f ms/step)"
          % (sel, feas, dt, 1e3 * dt / args.steps))
    if bk is not None:
        print("objective %d vs best known %d -> gap %d (%.1f%%)"
              % (-sel, int(bk), int(-bk) - sel, 100.0 * (int(-bk) - sel) / -bk))
    out = os.path.join(INSTANCE_DIR, "%s_solution_%d.npy" % (args.instance, sel))
    np.save(out, x)
    print("solution vector -> %s" % out)


if __name__ == "__main__":
    main()
