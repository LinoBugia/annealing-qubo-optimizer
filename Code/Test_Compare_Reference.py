"""
Test_Compare_Reference — verification harness against the reference library
../../annealing-cop-approximator (dict-based PBFs).

NOTE: this is the only part of the project that needs the reference library.
Nothing in Funcs_Qubo_* imports it; if it is absent, this script reports that
and exits instead of failing with a traceback.

Part A (machine precision):
  A1  E(x): eval_qubo  vs  evalPBF                at random states
  A2  dE vector: eval_delta_energy  vs  Eval_Delta_Energy (loop)
  A3  incremental dE after a flip sequence  vs  modificate_update_deltaE
  A4  converter round-trip pbf -> (A,b,c) -> pbf (tuple and int keys)

Part B (statistics + runtime):
  same problem, same cooling, old DA vs new batched DA: distribution of
  minima and runtime per trial.

Usage:  python Test_Compare_Reference.py [n] [seed]
"""

import random
import sys
import time

import numpy as np

sys.path.insert(0, "../../annealing-cop-approximator/Code")

from Funcs_Qubo_Annealing3 import eval_qubo, eval_delta_energy
from Funcs_Qubo_ProblemGeneration import (qubo_from_pbf, pbf_from_qubo,
                                          random_start_states)
from Funcs_Qubo_Optimizers import qubo_min_solver

try:
    from Funcs_Annealing2 import (createPoly, createPoly_negative,
                                  createPolyDict, evalPBF,
                                  Eval_Delta_Energy, modificate_update_deltaE)
    from Funcs_Optimizers import pbf_min_solver
    HAVE_REF = True
except ImportError as e:
    print("reference library not importable (%s) — internal checks only." % e)
    HAVE_REF = False


def _fail(name, diff):
    print("  FAIL %s  max|diff| = %.3e" % (name, diff))
    return 1


def _ok(name, diff):
    print("  ok   %-42s max|diff| = %.3e" % (name, diff))
    return 0


def part_A(n=80, seed=7, n_states=25, n_flips=200):
    print("Part A — machine precision (n=%d, seed=%d)" % (n, seed))
    fails = 0
    pbf = createPoly(n, 2, density=0.6, seed=seed)
    pbf_var_dict = createPolyDict(pbf, n)
    A, b, c = qubo_from_pbf(pbf, n)
    rng = np.random.default_rng(seed)

    # A1: energies
    X = rng.integers(0, 2, size=(n_states, n)).astype(np.int8)
    E_new = eval_qubo(A, b, c, X)
    E_ref = np.array([evalPBF(pbf, dict(enumerate(x.tolist()))) for x in X])
    diff = np.abs(E_new - E_ref).max()
    fails += _ok("A1 E(x) vs evalPBF", diff) if diff < 1e-9 * max(1, np.abs(E_ref).max()) \
        else _fail("A1 E(x)", diff)

    # A2: full dE vector
    x = X[0]
    va = dict(enumerate(x.tolist()))
    dE_new = eval_delta_energy(A, b, x)
    dE_ref = np.array([Eval_Delta_Energy(pbf, pbf_var_dict[i], va, i)
                       for i in range(n)])
    diff = np.abs(dE_new - dE_ref).max()
    fails += _ok("A2 dE vector vs Eval_Delta_Energy", diff) if diff < 1e-9 * max(1, np.abs(dE_ref).max()) \
        else _fail("A2 dE vector", diff)

    # A3: incremental dE over a flip sequence
    #     new: G update + row formula    reference: modificate_update_deltaE
    random.seed(seed)
    flips = [random.randint(0, n - 1) for _ in range(n_flips)]
    xf = x.astype(np.float64).copy()
    G = xf @ A
    dE_inc = (1.0 - 2.0 * xf) * (b + 2.0 * G)
    dE_ref_l = dE_ref.tolist()
    va_ref = dict(enumerate(x.tolist()))
    for k in flips:
        s = 1.0 - 2.0 * xf[k]
        xf[k] = 1.0 - xf[k]
        G += s * A[k, :]
        dE_inc = (1.0 - 2.0 * xf) * (b + 2.0 * G)
        dE_ref_l, va_ref = modificate_update_deltaE(va_ref, dE_ref_l, [k],
                                                    pbf, pbf_var_dict)
    diff = np.abs(dE_inc - np.asarray(dE_ref_l)).max()
    scale = max(1, np.abs(dE_ref_l).max())
    fails += _ok("A3 dE after %d flips vs modificate_update" % n_flips, diff) \
        if diff < 1e-8 * scale else _fail("A3 dE update", diff)
    assert all(int(xf[i]) == va_ref[i] for i in range(n)), "A3 state drift"

    # A4: converter round-trips
    for int_keys in (False, True):
        pbf2 = pbf_from_qubo(A, b, 0.0, int_keys=int_keys)
        A2, b2, c2 = qubo_from_pbf(pbf2, n)
        diff = max(np.abs(A2 - A).max(), np.abs(b2 - b).max())
        name = "A4 round-trip %s" % ("int keys" if int_keys else "tuple keys")
        fails += _ok(name, diff) if diff < 1e-12 else _fail(name, diff)
    return fails


def part_B(n=300, seed=11, steps=2000, num_MC=8, cooling_c=10.0, offset=1.0):
    print("Part B — statistics + runtime (n=%d, steps=%d, num_MC=%d)"
          % (n, steps, num_MC))
    # signed problem (spin-glass-like) — a non-trivial landscape
    pbf = createPoly_negative(n, 2, density=0.3, seed=seed)
    pbf_var_dict = createPolyDict(pbf, n)
    A, b, c = qubo_from_pbf(pbf, n)
    x0 = random_start_states(n, 1, seed)[0]
    cooling = ["logarithmic", cooling_c, 0]
    seeds = [1000 + i for i in range(num_MC)]

    t0 = time.time()
    _, Mins_ref, _, _ = pbf_min_solver(
        pbf, pbf_var_dict, "digitalAnnealing", steps, num_MC, list(cooling),
        seeds, seed, save_addinfo=True, offset_increase_rate=offset,
        initial_varAssignement_pre=x0.tolist(), random_start=False)
    t_ref = time.time() - t0

    t0 = time.time()
    _, Mins_new, _, _ = qubo_min_solver(
        A, b, c, steps=steps, num_MC=num_MC, cooling_param=list(cooling),
        seed_rand=seed, initial_varAssignements_pre=x0,
        offset_increase_rate=offset, save_addinfo=True)
    t_new = time.time() - t0

    print("  reference: min=%.6g  mean=%.6g  |  new: min=%.6g  mean=%.6g"
          % (np.min(Mins_ref), np.mean(Mins_ref),
             np.min(Mins_new), np.mean(Mins_new)))
    print("  total runtime: reference %.2fs  |  new %.2fs  (%.1fx)"
          % (t_ref, t_new, t_ref / max(t_new, 1e-9)))
    print("  per trial&step:  reference %.3g ms  |  new %.3g ms"
          % (1e3 * t_ref / (num_MC * steps), 1e3 * t_new / (num_MC * steps)))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    if not HAVE_REF:
        sys.exit(1)
    fails = part_A(n=n, seed=seed)
    if fails:
        print("Part A: %d FAILURES — Part B skipped" % fails)
        sys.exit(1)
    print("Part A: all exact to machine precision.")
    part_B(seed=seed)
