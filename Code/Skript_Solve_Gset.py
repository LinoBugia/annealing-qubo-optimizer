"""
Skript_Solve_Gset — Max-Cut on the Gset instances (Stanford / Yinyu Ye).

Two things here are NOT set by hand but derived from measurements:

  num_MC   from the efficiency plateau in the benchmark (Bench_Performance,
           S2): mc*n between 1e5 and 2.5e5 costs ~12 ns per evaluated flip.
           At n=800 that means mc=128, at n=20000 mc=8 — small instances
           want MANY trials, not few.

  T        from the scan correction:  T = q50(dE+) / ln(n / p).
           The DA tests all n flips per step, so the per-flip acceptance has
           to be p/n. On G1 the formula predicts T~0.63 and the measured
           optimum was 0.4-0.6 — it therefore holds well outside the gp case
           it was originally calibrated for.

The schedule is the da_gp_floor CURVE SHAPE (decaying onto a floor rather
than to zero), but with temperatures from the formula above instead of from
the gp calibration.

Usage:
    python Skript_Solve_Gset.py --steps 100000
    python Skript_Solve_Gset.py --instances G1,G22 --steps 300000
"""

import argparse
import os
import sys
import time
import shutil
import subprocess
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Funcs_Qubo_Annealing3 as qa3
from Funcs_Qubo_Annealers import digital_annealing_batch
from Funcs_Qubo_MaxCut import cut_value, load_gset
from Funcs_Qubo_TempSchedules import generate_cooling_schedule
from Funcs_Qubo_Randomizers import BulkRandomizer

GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/%s"
GSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Instances", "Gset")

# Best-known cut values. Source: the compilation in
# github.com/0816keisuke/max-cut-problem-benchmark (traced there to the
# Stanford Gset page and the Toshiba SBM benchmark); G81 from
# arXiv:2505.18508, where it is reported as proven optimal.
# NOTE: best-known values drift — sources give BOTH 10299 and 10116 for G55.
# The higher one is used here, i.e. the less flattering bar for us.
BEST_KNOWN = {"G1": 11624, "G11": 564, "G14": 3064, "G22": 13359,
              "G32": 1410, "G55": 10299, "G60": 14188, "G70": 9591,
              "G81": 14060}

DEFAULT_SET = ["G1", "G11", "G14", "G22", "G32", "G55", "G60", "G70", "G81"]


def download(url, path):
    """Try curl first: urllib fails on some setups on the certificate chain
    (TLS proxy / missing CA bundle)."""
    if shutil.which("curl"):
        rc = subprocess.call(["curl", "-sSL", "-o", path, url])
        if rc == 0 and os.path.getsize(path) > 0:
            return path
    urllib.request.urlretrieve(url, path)
    return path


def fetch(name):
    os.makedirs(GSET_DIR, exist_ok=True)
    path = os.path.join(GSET_DIR, name)
    if not os.path.exists(path):
        download(GSET_URL % name, path)
    return path


def pick_mc(n, target=1.5e5, lo=4, hi=256):
    """Pick num_MC so that mc*n lands on the measured efficiency plateau."""
    return int(np.clip(round(target / n), lo, hi))


def scan_temperature(A, b, x, n, p=0.3):
    """T = q50(dE+) / ln(n/p) — scan correction for 'all n flips per step'."""
    dE = qa3.eval_delta_energy(A, b, x)
    up = dE[dE > 0]
    q50 = float(np.quantile(up, 0.5)) if up.size else 1.0
    return q50 / np.log(n / p)


def floor_curve(T_hot, T_frz, S, d=2.22):
    t = np.arange(1, S + 1, dtype=float)
    return T_frz + (T_hot - T_frz) * np.log(2.0) / np.log(1.0 + t ** d)


def solve(name, steps, seed=17, offset=0.0, mc=None, schedule="scan",
          c_start=None, c_end=None, form="geometric"):
    path = fetch(name)
    A, b, c, info = load_gset(path)
    n = info["n"]
    mc = mc or pick_mc(n)
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n).astype(np.int8)          # Max-Cut: 50% is natural
    T_scan = scan_temperature(A, b, x0, n)
    sigma = qa3.delta_e_sigma(A, b)
    if schedule == "sigma":
        # Temperatures as multiples of sigma = std(dE), so one schedule fits
        # every instance. Valid here BECAUSE Max-Cut has a homogeneous dE
        # spectrum (spread q50/min is 1..5); on a gapped one sigma is
        # dominated by directions the chain never takes.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            Ts = generate_cooling_schedule(
                ["sigma", c_start, c_end, form, sigma], steps)
    else:
        Ts = floor_curve(1.5 * T_scan, 0.3 * T_scan, steps)

    X0 = rng.integers(0, 2, size=(mc, n)).astype(np.int8)   # own start per trial
    r = BulkRandomizer(seed, 64.0)
    t0 = time.perf_counter()
    out = digital_annealing_batch(A, b, c, X0, Ts, offset, r,
                                  save_addinfo=True, track_offsets=offset > 0.0,
                                  recompute_every=4096)
    dt = time.perf_counter() - t0
    cuts = np.array([cut_value(x, info["W"]) for x in out["X_min"]])
    best = int(cuts.max())
    bk = BEST_KNOWN.get(name)

    # Can the E_Offset do anything here at all? It only grows on steps where
    # NOTHING was admissible, and it resets on every accepted flip. The share
    # of steps entering with a nonzero offset is therefore both the size of
    # the opportunity and the ceiling on the effect: measured 11.7 % on a
    # set-packing instance where the offset was worth +3, and 0.5 % on an
    # independent-set instance where it is a dead parameter.
    blocked = np.nan
    off_max = np.nan
    if offset > 0.0:
        o = out["Offsets"]
        blocked = 100.0 * float((o > 0).sum()) / o.size
        off_max = float(o.max())

    return dict(name=name, n=n, m=info["m"], deg=info["avg_degree"], mc=mc,
                components=info["components"], isolated=info["isolated"],
                effective_n=info["effective_n"],
                T_scan=T_scan, best=best, median=int(np.median(cuts)),
                bk=bk, gap=None if bk is None else bk - best,
                pct=None if bk is None else 100.0 * best / bk,
                sec=dt, ms_step=1e3 * dt / steps, offset=offset,
                sigma=sigma, blocked=blocked, off_max=off_max, cuts=cuts,
                schedule=schedule, T_hi=float(Ts[0]), T_lo=float(Ts[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=",".join(DEFAULT_SET))
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--offset-sweep", default=None,
                    help="comma-separated E_Offset rates; each instance is "
                         "run at every rate and the table is grouped by rate")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--schedule", default="scan", choices=("scan", "sigma"))
    ap.add_argument("--c-start", default=None,
                    help="c_start, or a comma-separated list -> matrix mode. "
                         "Unset uses the calibrated default from "
                         "Funcs_Qubo_TempSchedules.")
    ap.add_argument("--c-end", default=None,
                    help="c_end, or a comma-separated list -> matrix mode")
    ap.add_argument("--form", default="geometric",
                    choices=("geometric", "staircase", "linear"))
    ap.add_argument("--schedule-sweep", action="store_true",
                    help="compare the scan schedule against a grid of sigma "
                         "schedules on each instance")
    ap.add_argument("--sigma-grid", default=None,
                    help="c_start/c_end pairs, e.g. 0.5/0.05,0.25/0.02")
    args = ap.parse_args()

    # None -> let the schedule module apply its own calibrated defaults, so
    # the driver cannot silently override them (it did, once).
    cs_list = ([None] if args.c_start is None
               else [float(v) for v in str(args.c_start).split(",")])
    ce_list = ([None] if args.c_end is None
               else [float(v) for v in str(args.c_end).split(",")])
    if args.schedule == "sigma" and (len(cs_list) > 1 or len(ce_list) > 1):
        cs_list = [v for v in cs_list if v is not None]
        ce_list = [v for v in ce_list if v is not None]
        # Grid over the sigma window. The measured constant-T optimum is
        # 0.116*sigma on G1 and 0.134*sigma on G22, so a window that brackets
        # ~0.12 is the region worth resolving; anything starting far above it
        # spends the run where erring hot collapses.
        names = [x.strip() for x in args.instances.split(",") if x.strip()]
        print("Gset / Max-Cut — sigma window matrix, %d steps, %s, seed %d"
              % (args.steps, args.form, args.seed))
        print("  measured T_opt/sigma: G1 0.116, G22 0.134\n")
        for name in names:
            grid = {}
            for cs in cs_list:
                for ce in ce_list:
                    if ce >= cs:
                        continue
                    r = solve(name, args.steps, args.seed, args.offset,
                              schedule="sigma", c_start=cs, c_end=ce,
                              form=args.form)
                    grid[(cs, ce)] = r
            bk = BEST_KNOWN.get(name)
            print("%s   (best known %s)" % (name, bk or "-"))
            print("   c_start \\ c_end " + "".join("%10.3g" % ce for ce in ce_list))
            for cs in cs_list:
                cells = []
                for ce in ce_list:
                    r = grid.get((cs, ce))
                    cells.append("%10d" % r["best"] if r else "%10s" % "-")
                print("   %-15.3g" % cs + "".join(cells))
            bestcell = max(grid.items(), key=lambda kv: kv[1]["best"])
            (bcs, bce), br = bestcell
            print("   best: c_start=%.3g c_end=%.3g -> %d (%.2f%%), "
                  "T %.4f..%.4f\n"
                  % (bcs, bce, br["best"], br["pct"] or 0.0,
                     br["T_hi"], br["T_lo"]), flush=True)
        return

    if args.offset_sweep is not None:
        rates = [float(v) for v in args.offset_sweep.split(",")]
        names = [s.strip() for s in args.instances.split(",") if s.strip()]
        print("Gset / Max-Cut — E_Offset sweep, %d steps, seed %d\n"
              % (args.steps, args.seed))
        print("  'blocked' = share of steps entering with a nonzero offset, "
              "i.e. following a\n  step where NOTHING was admissible. It is "
              "the ceiling on what the offset can do.\n")
        print("%-5s %8s %8s %9s %8s %8s %8s %8s"
              % ("inst", "rate", "cut", "bestknown", "%", "blocked",
                 "off_max", "s"))
        base = {}
        for name in names:
            for rate in rates:
                r = solve(name, args.steps, args.seed, rate)
                if rate == rates[0]:
                    base[name] = r["best"]
                delta = r["best"] - base[name]
                print("%-5s %8.4g %8d %9s %8s %7s%% %8s %8.0f%s"
                      % (name, rate, r["best"], r["bk"] or "-",
                         "%.2f" % r["pct"] if r["pct"] else "-",
                         "-" if np.isnan(r["blocked"]) else "%.2f" % r["blocked"],
                         "-" if np.isnan(r["off_max"]) else "%.3g" % r["off_max"],
                         r["sec"], "" if rate == rates[0] else "  %+d" % delta),
                      flush=True)
            print()
        return

    if args.schedule == "sigma":
        head = ("T in sigma=std(dE) units: %s..%s sigma, %s"
                % (cs_list[0] if cs_list[0] is not None else "default",
                   ce_list[0] if ce_list[0] is not None else "default",
                   args.form))
    else:
        head = "T from the scan correction"
    print("Gset / Max-Cut — %d steps, %s, mc on the plateau\n"
          % (args.steps, head))
    print("%-5s %6s %7s %5s %5s %7s %8s %9s %7s %6s %7s"
          % ("inst", "n", "edges", "deg", "mc", "T_scan", "cut",
             "bestknown", "gap", "%", "s"))
    rows = []
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        r = solve(name, args.steps, args.seed, args.offset,
                  schedule=args.schedule, c_start=cs_list[0],
                  c_end=ce_list[0], form=args.form)
        rows.append(r)
        print("%-5s %6d %7d %5.1f %5d %7.3f %8d %9s %7s %6s %7.0f%s"
              % (r["name"], r["n"], r["m"], r["deg"], r["mc"], r["T_scan"],
                 r["best"], r["bk"] if r["bk"] else "-",
                 r["gap"] if r["gap"] is not None else "-",
                 "%.2f" % r["pct"] if r["pct"] else "-", r["sec"],
                 "" if r["components"] <= 1 else
                 "   %d components, %d isolated -> effective n=%d"
                 % (r["components"], r["isolated"], r["effective_n"])),
              flush=True)
    done = [r for r in rows if r["pct"] is not None]
    if done:
        print("\nmean %.2f%% of best known | worst %s (%.2f%%) | "
              "best %s (%.2f%%)"
              % (np.mean([r["pct"] for r in done]),
                 min(done, key=lambda r: r["pct"])["name"],
                 min(r["pct"] for r in done),
                 max(done, key=lambda r: r["pct"])["name"],
                 max(r["pct"] for r in done)))
        hit = [r["name"] for r in done if r["gap"] is not None and r["gap"] <= 0]
        print("best known matched or beaten: %s" % (hit or "none"))


if __name__ == "__main__":
    main()
