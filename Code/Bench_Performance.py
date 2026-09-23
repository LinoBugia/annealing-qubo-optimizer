"""
Bench_Performance — flat runtime and limit measurement of the batched DA kernel.

The question: what problem size does this handle, on what hardware, in what
time — and where is the wall? NOT convergence, NOT solution quality.

Only the kernel (digital_annealing_batch) is timed; problem construction sits
outside the clock. Each measurement picks its step count adaptively so large n
do not run forever.

Blocks:
  S1  step time vs n (dense, mc=8)             -> scaling exponent
  S2  step time vs num_MC                      -> what an extra trial costs
  S3  dense vs CSR across density              -> when sparse pays off
  S3b CSR at fixed degree, very large n        -> how far n can actually go
  S4  float64 vs float32                       -> the memory lever
  S5  problem type at fixed n                  -> does the type matter?
  S6  recompute_every                          -> cost of exact reconstruction
  S7  end-to-end qubo_min_solver including setup
  S8  anchor against the dict-based reference (SA / DA)
  S9  breakdown of one step into its numpy operations

Usage:  python Bench_Performance.py [--quick] [--max-n N] [--out FILE.csv]
"""

import argparse
import csv
import gc
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Funcs_Qubo_Annealing3 as qa3
from Funcs_Qubo_Annealers import digital_annealing_batch
from Funcs_Qubo_ProblemGeneration import (create_graph_binary_clustering_qubo,
                                          create_number_partitioning_qubo,
                                          random_start_states)
from Funcs_Qubo_Randomizers import BulkRandomizer
from Funcs_Qubo_TempSchedules import generate_cooling_schedule

try:
    from scipy import sparse as sp
except ImportError:
    sp = None

ROWS = []                       # collected measurement rows -> CSV
TARGET_SEC = 0.6                # target DIFFERENCE of the measurement window
PILOT_STEPS = 16


# ── Helpers ──────────────────────────────────────────────────────────────────

def rss_gb():
    """Peak RSS of the process in GB (macOS reports bytes, Linux KB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def rss_now_gb():
    """Current RSS in GB (peak RSS never falls, so it is useless per point)."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out.strip()) / 1e6
    except Exception:
        return float("nan")


def hw_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__}
    try:
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        info["ram_gb"] = round(int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"])) / 1e9)
        info["cores"] = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]))
    except Exception:
        info["cpu"] = platform.processor() or "?"
    return info


_HW = None


def hw():
    global _HW
    if _HW is None:
        _HW = hw_info()
    return _HW


MEM_CAP_GB = 1.0                # hard cap for a single matrix


def fits_dense(n, dtype=np.float64, frac=0.25):
    """
    May a dense (n,n) matrix be allocated?

    Two bounds, the smaller one wins:
      - MEM_CAP_GB: hard cap (1 GB by default), raise it with --mem-cap-gb
      - frac of total RAM

    The cap is deliberately low. A sweep up to n=32768 takes 8.6 GB in one
    block; together with whatever else runs on the machine that pushes it into
    memory pressure, where it freezes rather than merely slowing down — all
    the more so when little disk is free for swap. Large n belong in the CSR
    path (S3), not the dense one.
    """
    need = n * n * np.dtype(dtype).itemsize
    return need < min(MEM_CAP_GB * 1e9, frac * hw().get("ram_gb", 16) * 1e9)


def fast_signed_qubo(n, seed=42, dtype=np.float64):
    """
    Dense signed QUBO in canonical form, built with numpy (the library's
    combinations generator is O(n^2) in Python and unusable above n ~ 4000).
    For runtime only size and dtype matter, not the coefficient distribution —
    symmetry is still established exactly so that dE and the acceptance rate
    stay realistic.
    """
    rng = np.random.default_rng(seed)
    # draw directly in the target dtype: uniform() always returns float64,
    # and a later astype(float32) doubles peak memory — which blows up exactly
    # the sizes this benchmark is about.
    A = rng.random(size=(n, n), dtype=dtype)
    A -= dtype(0.5)
    A *= dtype(128.0)
    CH = 1024                                   # mirror block-wise (memory)
    for i0 in range(0, n, CH):
        i1 = min(i0 + CH, n)
        if i1 < n:
            A[i1:, i0:i1] = A[i0:i1, i1:].T
        blk = np.triu(A[i0:i1, i0:i1], 1)
        A[i0:i1, i0:i1] = blk + blk.T
    np.fill_diagonal(A, 0.0)
    b = ((rng.random(n) - 0.5) * 128.0).astype(dtype, copy=False)
    return A, b, 0.0


def sparse_signed_qubo(n, density, seed=42):
    """Symmetric CSR QUBO at a target density (nonzero fraction of n^2)."""
    if sp is None:
        raise RuntimeError("scipy is missing")
    rng = np.random.default_rng(seed)
    S = sp.random(n, n, density=density / 2.0, format="csr",
                  random_state=np.random.RandomState(seed),
                  data_rvs=lambda k: rng.uniform(-64.0, 64.0, size=k))
    S = (S + S.T).tocsr()
    S.setdiag(0.0)
    S.eliminate_zeros()
    b = rng.uniform(-64.0, 64.0, size=n)
    return S.astype(np.float64), b, 0.0


def degree_signed_qubo(n, degree=64, seed=42):
    """
    Symmetric CSR QUBO at a fixed average DEGREE rather than a fixed density.

    Built straight from an edge list: O(n*degree). sp.random() is unusable
    here — it parametrises over n^2 and hangs above n ~ 10^5 (n^2 = 10^10
    candidate positions).
    """
    if sp is None:
        raise RuntimeError("scipy is missing")
    rng = np.random.default_rng(seed)
    k = max(1, degree // 2)                     # half the edges, then mirror
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = rng.integers(0, n, size=n * k, dtype=np.int32)
    keep = rows != cols                         # the diagonal is 0 by contract
    rows, cols = rows[keep], cols[keep]
    vals = (rng.random(rows.size) - 0.5) * 128.0
    S = sp.csr_matrix(
        (np.concatenate([vals, vals]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n))
    S.sum_duplicates()
    S.setdiag(0.0)
    S.eliminate_zeros()
    b = (rng.random(n) - 0.5) * 128.0
    return S, b, 0.0


def cooling_for(A, b, x0, steps, quantile=0.5):
    """
    Logarithmic schedule on the problem's dE scale. The point is NOT good
    convergence but a realistic mix of accepted and rejected flips: at a far
    too cold T the scan would have systematically different branch costs.
    """
    dE = qa3.eval_delta_energy(A, b, x0)
    up = dE[dE > 0]
    c = float(np.quantile(up, quantile)) if len(up) else 1.0
    return generate_cooling_schedule(["logarithmic", max(c, 1e-12)], steps)


def time_kernel(A, b, c, mc, n, seed=7, recompute_every=1024,
                mem_budget_mb=64.0, max_steps=40000, min_steps=64, reps=3,
                fixed_steps=None):
    """
    Measure the pure step time of the kernel as a two-point slope:

        t(s)  = setup + s * per_step
        t(2s) = setup + 2s * per_step   ->   per_step = (t(2s) - t(s)) / s

    This cancels the kernel's one-off O(n^2) setup (G = X@A, E, dE, RNG block
    allocation) exactly, instead of smearing it into the step time — at large
    n that setup outweighs hundreds of steps.

    The measurement window is calibrated by doubling until the DIFFERENCE
    t(2s)-t(s) reaches at least TARGET_SEC; otherwise the slope drowns in
    noise (at small s the one-off 64 MB RNG block dominates). The window is
    also at least recompute_every long, so the periodic exact reconstruction
    is amortised INTO the step time rather than being skipped (at large n it
    would otherwise be systematically understated).

    The measurement is repeated `reps` times and the MINIMUM is kept. On a
    big.LITTLE CPU (M series: 4 P + 6 E cores) the process migrates between
    clusters, which adds up to 1.7x spread upward — the minimum is the least
    disturbed run, not a lucky outlier.

    Returns: (seconds_per_step, steps_used, setup_seconds)
    """
    X0 = random_start_states(n, mc, seed)
    if recompute_every:
        min_steps = max(min_steps, recompute_every)

    def run(steps):
        # cooling_for does an O(n^2) matvec — deliberately BEFORE the clock
        Ts = cooling_for(A, b, X0[0], steps)
        r = BulkRandomizer(seed, mem_budget_mb)
        gc.collect()
        t0 = time.perf_counter()
        digital_annealing_batch(A, b, c, X0, Ts, 0.0, r, save_addinfo=True,
                                recompute_every=recompute_every)
        return time.perf_counter() - t0

    run(PILOT_STEPS)                                    # warm caches and BLAS
    if fixed_steps:
        # Variants must be compared at the SAME step count: otherwise each
        # one amortises its periodic costs over a different window and the
        # comparison ends up measuring the window.
        steps = fixed_steps
        ds = [run(2 * steps) - run(steps) for _ in range(reps)]
    else:
        steps = min_steps
        while True:
            d = run(2 * steps) - run(steps)
            if d >= TARGET_SEC or 2 * steps >= max_steps:
                break
            grow = max(2.0, min(8.0, TARGET_SEC / max(d, 1e-6)))
            steps = min(max_steps // 2, int(steps * grow))
        ds = [d] + [run(2 * steps) - run(steps) for _ in range(reps - 1)]
    per_step = max(min(ds) / steps, 1e-12)
    setup = max(run(1) - per_step, 0.0)                 # time to the 1st step
    return per_step, steps, setup


def record(block, **kw):
    ROWS.append(dict(block=block, **kw))


def emit(block, n, mc, per_step, steps, setup, **extra):
    """Record one measurement and print it on a single line. Metrics:
       us/step, and ns per evaluated flip (n*mc proposals per step)."""
    ns_flip = per_step * 1e9 / (n * mc)
    record(block, n=n, mc=mc, steps=steps, sec_per_step=per_step,
           us_per_step=per_step * 1e6, ns_per_flip=ns_flip,
           setup_sec=setup, **extra)
    tail = "  ".join("%s=%s" % (k, v) for k, v in extra.items())
    print("    n=%-6d mc=%-4d %9.1f us/step  %7.2f ns/flip  "
          "setup %6.3fs  (%d steps)  %s"
          % (n, mc, per_step * 1e6, ns_flip, setup, steps, tail))


# ── S1: step time vs n ───────────────────────────────────────────────────────

def s1_size_sweep(ns, mc=8):
    print("\nS1 — step time vs n (dense float64, mc=%d)" % mc)
    for n in ns:
        if not fits_dense(n):
            print("    n=%-6d skipped — A would need %.1f GB"
                  % (n, n * n * 8 / 1e9))
            continue
        t_build = time.perf_counter()
        A, b, c = fast_signed_qubo(n)
        t_build = time.perf_counter() - t_build
        mem = A.nbytes / 1e9
        per_step, steps, setup = time_kernel(A, b, c, mc, n)
        emit("S1_size", n, mc, per_step, steps, setup,
             A_gb=round(mem, 4), build_sec=round(t_build, 2),
             rss_gb=round(rss_now_gb(), 2))
        del A, b
        gc.collect()


# ── S2: step time vs num_MC ──────────────────────────────────────────────────

def s2_mc_sweep(ns, mcs):
    print("\nS2 — step time vs num_MC (dense float64)")
    for n in ns:
        A, b, c = fast_signed_qubo(n)
        base = None
        for mc in mcs:
            per_step, steps, setup = time_kernel(A, b, c, mc, n)
            if base is None:
                base = per_step
            emit("S2_mc", n, mc, per_step, steps, setup,
                 factor_vs_mc1=round(per_step / base, 3))
        del A, b
        gc.collect()


# ── S3: dense vs CSR ─────────────────────────────────────────────────────────

def s3_sparse(ns, densities, mc=8):
    if sp is None:
        print("\nS3 — skipped (scipy missing)")
        return
    print("\nS3 — dense vs CSR (mc=%d)" % mc)
    for n in ns:
        A, b, c = fast_signed_qubo(n)
        per_step, steps, setup = time_kernel(A, b, c, mc, n)
        emit("S3_sparse", n, mc, per_step, steps, setup,
             fmt="dense", density=1.0, A_gb=round(A.nbytes / 1e9, 4))
        dense_ref = per_step
        del A, b
        gc.collect()
        for d in densities:
            S, bs, cs = sparse_signed_qubo(n, d)
            gb = (S.data.nbytes + S.indices.nbytes + S.indptr.nbytes) / 1e9
            per_step, steps, setup = time_kernel(S, bs, cs, mc, n)
            emit("S3_sparse", n, mc, per_step, steps, setup,
                 fmt="csr", density=d, A_gb=round(gb, 4), nnz=int(S.nnz),
                 vs_dense=round(per_step / dense_ref, 3))
            del S, bs
            gc.collect()


# ── S3b: how far can n actually go (CSR, bounded degree) ─────────────────────

def s3b_large_sparse(ns, degree=64, mc=8):
    """
    The actual limit question. The dense path is capped by n^2 memory, but a
    large real QUBO is not dense — it has a bounded DEGREE (neighbours per
    variable), not a constant density. With nnz = n*degree memory grows only
    LINEARLY in n, and the step is O(mc*n) anyway. This pushes n as far as it
    will go.
    """
    if sp is None:
        print("\nS3b — skipped (scipy missing)")
        return
    print("\nS3b — CSR at fixed degree %d (mc=%d)" % (degree, mc))
    for n in ns:
        need = n * degree * 12.0                    # data(8) + indices(4..8)
        if need > MEM_CAP_GB * 1e9:
            print("    n=%-7d skipped — nnz would need ~%.1f GB" % (n, need / 1e9))
            continue
        t0 = time.perf_counter()
        S, b, c = degree_signed_qubo(n, degree)
        t_build = time.perf_counter() - t0
        gb = (S.data.nbytes + S.indices.nbytes + S.indptr.nbytes) / 1e9
        per_step, steps, setup = time_kernel(S, b, c, mc, n, reps=1)
        emit("S3b_large", n, mc, per_step, steps, setup, fmt="csr",
             degree=degree, nnz=int(S.nnz), A_gb=round(gb, 4),
             dense_would_be_gb=round(n * n * 8 / 1e9, 1),
             build_sec=round(t_build, 1), rss_gb=round(rss_now_gb(), 2))
        del S, b
        gc.collect()


# ── S4: dtype ────────────────────────────────────────────────────────────────

def s4_dtype(ns, mc=8):
    print("\nS4 — float64 vs float32 (dense, mc=%d)" % mc)
    for n in ns:
        for dt in (np.float64, np.float32):
            if not fits_dense(n, dt):
                print("    n=%-6d %-8s skipped — A would need %.1f GB"
                      % (n, np.dtype(dt).name, n * n * np.dtype(dt).itemsize / 1e9))
                continue
            A, b, c = fast_signed_qubo(n, dtype=dt)
            per_step, steps, setup = time_kernel(A, b, c, mc, n)
            emit("S4_dtype", n, mc, per_step, steps, setup,
                 dtype=np.dtype(dt).name, A_gb=round(A.nbytes / 1e9, 4))
            del A, b
            gc.collect()


# ── S5: problem type ─────────────────────────────────────────────────────────

def s5_problem_type(n=2048, mc=8):
    print("\nS5 — problem type at n=%d, mc=%d" % (n, mc))
    rng = np.random.default_rng(3)
    probs = {
        "random_signed": lambda: fast_signed_qubo(n),
        "number_partitioning": lambda: create_number_partitioning_qubo(
            rng.integers(1, 1 << 20, size=n).astype(float)),
        "gram_clustering": lambda: create_graph_binary_clustering_qubo(
            rng.normal(size=(n, 16))),
    }
    for name, gen in probs.items():
        A, b, c = gen()
        A = np.ascontiguousarray(A, dtype=np.float64)
        per_step, steps, setup = time_kernel(A, b, c, mc, n)
        emit("S5_problem", n, mc, per_step, steps, setup, problem=name)
        del A, b
        gc.collect()


# ── S6: recompute_every ──────────────────────────────────────────────────────

def s6_recompute(n=4096, mc=8, every=(128, 512, 1024, 4096, 0)):
    print("\nS6 — cost of recompute_every (n=%d, mc=%d)" % (n, mc))
    A, b, c = fast_signed_qubo(n)
    for e in every:
        per_step, steps, setup = time_kernel(A, b, c, mc, n, recompute_every=e,
                                             fixed_steps=4096, min_steps=1)
        emit("S6_recompute", n, mc, per_step, steps, setup,
             recompute_every=e if e else "off")
    del A, b
    gc.collect()


# ── S7: end to end ───────────────────────────────────────────────────────────

def s7_end_to_end(cases, steps=1000, num_MC=8):
    """Full qubo_min_solver call incl. setup (dE init, cooling, tracking)."""
    print("\nS7 — end-to-end qubo_min_solver (steps=%d, num_MC=%d)"
          % (steps, num_MC))
    import contextlib
    import io

    from Funcs_Qubo_Optimizers import qubo_min_solver
    for n in cases:
        A, b, c = fast_signed_qubo(n)
        x0 = random_start_states(n, 1, 42)[0]
        gc.collect()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            qubo_min_solver(A, b, c, steps=steps, num_MC=num_MC,
                            cooling_param=["logarithmic", 50.0],
                            initial_varAssignements_pre=x0,
                            offset_increase_rate="auto_gp", save_addinfo=True)
        t = time.perf_counter() - t0
        record("S7_e2e", n=n, mc=num_MC, steps=steps, total_sec=t,
               sec_per_step=t / steps, us_per_step=1e6 * t / steps,
               ns_per_flip=t * 1e9 / (steps * n * num_MC))
        print("    n=%-6d  total %7.2fs  (%d trials x %d steps)  "
              "%.1f us/step" % (n, t, num_MC, steps, 1e6 * t / steps))
        del A, b
        gc.collect()


# ── S8: anchor against the dict-based reference ──────────────────────────────

def s8_reference(ns, steps=300, seed=11, density=0.3):
    """A small anchor: dict-based SA and DA from the reference library.
    Only a few small n — the reference is O(n) per step in Python."""
    print("\nS8 — anchor: dict reference SA / DA (steps=%d, 1 trial)" % steps)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "annealing-cop-approximator", "Code"))
    try:
        from Funcs_Annealers import DigitalAnnealing, simulatedAnnealing
        from Funcs_Annealing2 import (createPolyDict, evalPBF,
                                      Eval_Delta_Energy)
    except ImportError as e:
        print("    reference not importable (%s) — skipped" % e)
        return
    from Funcs_Qubo_ProblemGeneration import pbf_from_qubo

    for n in ns:
        rng = np.random.default_rng(seed)
        A = rng.uniform(-64.0, 64.0, size=(n, n))
        A = np.triu(A, 1)
        mask = rng.random((n, n)) < density
        A = A * np.triu(mask, 1)
        A = A + A.T
        b = rng.uniform(-64.0, 64.0, size=n) * (rng.random(n) < density)
        c = 0.0

        pbf = pbf_from_qubo(A, b, c)
        pvd = createPolyDict(pbf, n)
        x0 = random_start_states(n, 1, seed)[0]
        va = dict(enumerate(x0.tolist()))
        Ts = cooling_for(A, b, x0, steps)
        E0 = evalPBF(pbf, va)
        dE0 = [Eval_Delta_Energy(pbf, pvd[i], va, i) for i in range(n)]

        t0 = time.perf_counter()
        DigitalAnnealing(E0, list(dE0), pbf, pvd, steps, dict(va), list(Ts),
                         True, seed, 0.0)
        t_da = (time.perf_counter() - t0) / steps

        sa_steps = max(steps, 4 * n)            # SA does 1 proposal per step
        Ts_sa = cooling_for(A, b, x0, sa_steps)
        t0 = time.perf_counter()
        simulatedAnnealing(pbf, pvd, sa_steps, dict(va), list(Ts_sa), True, seed)
        t_sa = (time.perf_counter() - t0) / sa_steps

        A64 = np.ascontiguousarray(A)
        per_step, st, setup = time_kernel(A64, b, c, 1, n)

        record("S8_ref", n=n, mc=1, steps=steps, monomials=len(pbf),
               us_per_step_ref_da=t_da * 1e6,
               ns_per_flip_ref_da=t_da * 1e9 / n,
               us_per_step_ref_sa=t_sa * 1e6,
               ns_per_flip_ref_sa=t_sa * 1e9,
               us_per_step=per_step * 1e6,
               ns_per_flip=per_step * 1e9 / n,
               speedup_da=t_da / per_step)
        print("    n=%-5d monomials=%-6d | ref-DA %8.1f us/step (%6.0f ns/flip) | "
              "ref-SA %7.1f us/flip | new %7.1f us/step (%5.1f ns/flip) | %.0fx"
              % (n, len(pbf), t_da * 1e6, t_da * 1e9 / n, t_sa * 1e6,
                 per_step * 1e6, per_step * 1e9 / n, t_da / per_step))
        del A, A64, pbf, pvd
        gc.collect()


# ── S9: breakdown of one step ────────────────────────────────────────────────

def s9_step_profile(ns=(1024, 4096, 16384), mc=8, iters=200):
    """
    Split the step into its numpy operations and time each one. This answers
    what faster hardware could actually buy: a step is O(mc*n) of memory
    traffic plus mc*n random numbers — if the RNG dominates, only an
    RNG-strong backend (GPU) helps, not more clock and not more BLAS threads.
    """
    print("\nS9 — breakdown of one step (mc=%d)" % mc)
    for n in ns:
        if not fits_dense(n):
            continue
        A, b, c = fast_signed_qubo(n)
        X = random_start_states(n, mc, 7)
        Xf = X.astype(np.float64)
        G = Xf @ A
        dE = (1.0 - 2.0 * Xf) * (b + 2.0 * G)
        offs = np.zeros(mc)
        rnd = BulkRandomizer(7, 64.0)
        T = float(np.quantile(dE[dE > 0], 0.5))
        rows = np.arange(mc)
        kk = np.random.default_rng(1).integers(0, n, mc)
        sgn = np.ones(mc)

        def timeit(fn):
            fn()                                    # warm up
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) / iters

        eps = rnd.exponentials(mc, n)
        acc = dE < (offs[:, None] + T * eps)
        parts = {
            "rng_exp": lambda: rnd.exponentials(mc, n),
            "scan_cmp": lambda: dE < (offs[:, None] + T * eps),
            "count_sum": lambda: acc.sum(axis=1),
            "select_cumsum": lambda: (acc.cumsum(axis=1) > 0).argmax(axis=1),
            "G_update": lambda: G.__setitem__(
                rows, G[rows] + sgn[:, None] * A[kk, :]),
            "dE_recompute": lambda: dE.__setitem__(
                rows, (1.0 - 2.0 * Xf[rows]) * (b + 2.0 * G[rows])),
        }
        times = {k: timeit(f) for k, f in parts.items()}
        tot = sum(times.values())
        row = {("us_" + k): v * 1e6 for k, v in times.items()}
        row.update({("pct_" + k): 100.0 * v / tot for k, v in times.items()})
        record("S9_profile", n=n, mc=mc, sum_us=tot * 1e6, **row)
        print("    n=%-6d total %7.1f us  |  %s" % (
            n, tot * 1e6,
            "  ".join("%s %.0f%%" % (k, 100 * v / tot)
                      for k, v in sorted(times.items(), key=lambda x: -x[1]))))
        del A, b, G, dE, Xf
        gc.collect()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global MEM_CAP_GB
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-n", type=int, default=8192)
    ap.add_argument("--out", default="bench_performance.csv")
    ap.add_argument("--only", default="", help="blocks to run, e.g. S1,S2")
    ap.add_argument("--mem-cap-gb", type=float, default=1.0,
                    help="hard cap for ONE dense matrix (GB)")
    args = ap.parse_args()
    MEM_CAP_GB = args.mem_cap_gb

    info = hw_info()
    print("Bench_Performance — %s" % info.get("cpu", "?"))
    print("  %s | %s cores | %s GB RAM | numpy %s"
          % (info["platform"], info.get("cores", "?"), info.get("ram_gb", "?"),
             info["numpy"]))

    ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ns = [n for n in ns if n <= args.max_n]
    if args.quick:
        ns = [n for n in ns if n <= 2048]

    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    run = lambda tag: (not only) or tag in only

    t_all = time.perf_counter()
    if run("S1"):
        s1_size_sweep(ns)
    if run("S2"):
        s2_mc_sweep([n for n in (512, 4096) if n <= args.max_n],
                    [1, 2, 4, 8, 16, 32, 64, 128])
    if run("S3"):
        s3_sparse([n for n in (4096, 16384, 65536) if n <= max(args.max_n, 65536)],
                  [0.001, 0.01, 0.05])
    if run("S3B"):
        s3b_large_sparse([16384, 65536, 262144])
    if run("S4"):
        s4_dtype([n for n in (2048, 4096, 8192) if n <= args.max_n])
    if run("S5"):
        s5_problem_type()
    if run("S6"):
        s6_recompute()
    if run("S7"):
        s7_end_to_end([n for n in (512, 2048, 8192) if n <= args.max_n])
    if run("S8"):
        s8_reference([100, 200, 500, 1000, 2000])
    if run("S9"):
        s9_step_profile([n for n in (1024, 4096, 8192) if n <= args.max_n])

    print("\ntotal %.1f min, peak RSS %.2f GB" %
          ((time.perf_counter() - t_all) / 60.0, rss_gb()))

    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(ROWS)
    print("CSV -> %s (%d rows)" % (args.out, len(ROWS)))


if __name__ == "__main__":
    main()
