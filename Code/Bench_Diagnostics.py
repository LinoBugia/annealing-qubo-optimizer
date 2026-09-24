"""
Bench_Diagnostics — one instrumented run that answers "what is this chain
actually doing", so a mechanism can be diagnosed before it is built.

Every metric here earned its place by settling a question that reasoning alone
got wrong:

  admissible set size A   a temperature derived from the MEDIAN uphill move
                          instead of the MINIMUM left ~590 flips admissible
                          per step; the run degenerated into a random walk and
                          six temperatures returned the starting solution
  fully rejected share    the E_Offset only grows on fully rejected steps —
                          0.5% of steps on one instance, 12% on another, so
                          the same parameter is dead on one and decisive on
                          the other
  forced-undo rate        after an uphill escape the undo is downhill by
                          exactly the same amount (dE flips sign exactly, A
                          has a zero diagonal) and is unconditionally
                          admissible. Whether that matters is a measurement,
                          not an argument
  Hamming vs flips        26 382 accepted flips produced 341 bits of net
                          displacement — the chain was cycling, not exploring,
                          which is why three escape mechanisms did nothing
  distinct touched        675 of 11 811 variables in 30 000 steps: the search
                          never even looks at 94% of the problem
  feasible share          with a penalty of 2*max(c) the feasible manifold is
                          absorbing (100% feasible, zero violations ever), so
                          the chain can never cross to another region
  pairwise trial Hamming  two good solutions shared 13% of their set bits —
                          there is no backbone to fix, and an elite pool would
                          be genuinely diverse

The loop below mirrors digital_annealing_batch step for step, including the
order of the RNG draws, so the numbers describe the real algorithm and not a
simplified stand-in. It is deliberately a separate copy: the kernel stays free
of counters.
"""

import numpy as np

import Funcs_Qubo_Annealing3 as qa3
from Funcs_Qubo_Randomizers import BulkRandomizer


def delta_e_spectrum(A, b, x, p=0.3):
    """
    The uphill dE spectrum at a state, and the temperature each statistic
    implies. Which one is right depends on the shape:

      spread-out spectrum (real-valued coefficients) -> use the median
      gapped spectrum (a lowest barrier well below the rest) -> use the
      minimum, because only those moves are ever taken

    `spread` = q50/min is what tells the two apart, NOT the share of moves
    sitting at the minimum. On cdc7 only 2% of uphill moves are at the
    minimum, yet the minimum is still right: those 232 moves are the only
    gateway, everything above them never happens. spread is 15 there and
    about 1-2 on Max-Cut.
    """
    dE = qa3.eval_delta_energy(A, b, x)
    up = dE[dE > 0]
    if up.size == 0:
        return {"n_uphill": 0}
    n = len(b)
    lo = float(up.min())
    scale = np.log(n / p)
    return {"n_uphill": int(up.size),
            "n_total": n,
            "min": lo,
            "q25": float(np.quantile(up, .25)),
            "q50": float(np.quantile(up, .50)),
            "count_at_min": int((up == up.min()).sum()),
            "share_at_min": float((up == up.min()).mean()),
            "T_from_min": lo / scale,
            "T_from_q50": float(np.quantile(up, .50)) / scale,
            "spread": float(np.quantile(up, .50)) / lo if lo > 0 else float("inf")}


def diagnose(A, b, c, X0, T, steps, offset_increase_rate=0.0, seed=11,
             window=50, buckets=1, violations=None, recompute_every=4096):
    """
    Run `steps` steps at constant temperature `T` and record what the chain
    does. Returns a dict; pass it to report().

    X0         : (mc, n) start states, one row per trial
    violations : optional callable (X, E) -> (mc,) count of violated
                 constraints. It gets E as well so a penalty encoding can
                 use the cheap identity instead of a matmul per step —
                 for set packing with penalty P, E = P*V - |S|, so
                 V = (E + X.sum(1)) / P.
    buckets    : split the run into this many equal phases and report each
                 separately — an effect that only appears late (or only
                 early) is invisible in a single aggregate.
    """
    sparse = qa3.is_sparse(A)
    b = np.asarray(b, dtype=np.float64)
    X = np.asarray(X0, dtype=np.int8).copy()
    X_start = X.copy()
    mc, n = X.shape
    rnd = BulkRandomizer(seed=seed)

    Xf = X.astype(np.float64)
    G = np.asarray(Xf @ A)
    E = (Xf * G).sum(axis=1) + Xf @ b + c
    dE = (1.0 - 2.0 * Xf) * (b + 2.0 * G)
    offs = np.zeros(mc)
    rows_all = np.arange(mc)

    prev = np.full(mc, -1, dtype=np.int64)
    hist = np.full((mc, window), -1, dtype=np.int64)
    hp = np.zeros(mc, dtype=np.int64)
    touched = np.zeros((mc, n), dtype=bool)
    flips = np.zeros(mc, dtype=np.int64)

    per = max(1, steps // max(buckets, 1))
    out_buckets = []
    acc_n = imm = win = forced = dead = 0
    A_sum = 0.0; down_sum = 0.0; p_undo = 0.0; A_all = []
    feas_n = 0; feas_tot = 0; viol_sum = 0.0; viol_max = 0.0

    for step in range(steps):
        eps = rnd.exponentials(mc, n)
        acc = dE < (offs[:, None] + T * eps)
        count = acc.sum(axis=1)
        has = count > 0
        down = ((dE < 0) & acc).sum(axis=1)
        dead += int((~has).sum())

        # was the previous flip still available, and how likely is it to be
        # picked? selection is uniform over the admissible set, so 1/A
        idx = np.where(prev >= 0, prev, 0)
        avail = np.zeros(mc, dtype=bool)
        m = (prev >= 0) & has
        if m.any():
            avail[m] = acc[rows_all[m], idx[m]]
        p_undo += float((avail / np.maximum(count, 1)).sum())
        forced += int((avail & (down == 1) & (dE[rows_all, idx] < 0)).sum())

        if has.any():
            A_sum += float(count[has].sum())
            down_sum += float(down[has].sum())
            A_all.extend(count[has].tolist())
            u = rnd.uniforms(mc)
            r = np.minimum((u * count).astype(np.int64),
                           np.maximum(count - 1, 0))
            k = (acc.cumsum(axis=1) > r[:, None]).argmax(axis=1)
            rows, kk = rows_all[has], k[has]
            acc_n += rows.size
            imm += int((kk == prev[rows]).sum())
            win += int((hist[rows] == kk[:, None]).any(axis=1).sum())

            sgn = 1.0 - 2.0 * X[rows, kk].astype(np.float64)
            E[rows] += dE[rows, kk]
            X[rows, kk] = 1 - X[rows, kk]
            if sparse:
                for r_i, k_i, s_i in zip(rows.tolist(), kk.tolist(),
                                         sgn.tolist()):
                    lo, hi = A.indptr[k_i], A.indptr[k_i + 1]
                    G[r_i, A.indices[lo:hi]] += s_i * A.data[lo:hi]
            else:
                G[rows] += sgn[:, None] * A[kk, :]
            dE[rows] = (1.0 - 2.0 * X[rows].astype(np.float64)) * (b + 2.0 * G[rows])
            offs[rows] = 0.0
            prev[rows] = kk
            touched[rows, kk] = True
            flips[rows] += 1
            hist[rows, hp[rows] % window] = kk
            hp[rows] += 1
        offs[~has] += offset_increase_rate

        if violations is not None:
            v = np.asarray(violations(X, E), dtype=np.float64)
            feas_n += int((v < 0.5).sum()); feas_tot += mc
            viol_sum += float(v.sum()); viol_max = max(viol_max, float(v.max()))

        if recompute_every and (step + 1) % recompute_every == 0:
            Xf = X.astype(np.float64)
            G = np.asarray(Xf @ A)
            E = (Xf * G).sum(axis=1) + Xf @ b + c
            dE = (1.0 - 2.0 * Xf) * (b + 2.0 * G)

        if (step + 1) % per == 0 and len(out_buckets) < buckets:
            a = max(acc_n, 1)
            out_buckets.append({
                "step": step + 1,
                "flips": float(flips.mean()),
                "hamming": float((X != X_start).sum(axis=1).mean()),
                "distinct": float(touched.sum(axis=1).mean()),
                "A_mean": A_sum / a,
                "down_mean": down_sum / a,
                "accepted_pct": 100.0 * acc_n / (per * mc),
                "rejected_pct": 100.0 * dead / (per * mc),
                "immediate_pct": 100.0 * imm / a,
                "window_pct": 100.0 * win / a,
                "forced_pct": 100.0 * forced / a,
                "expected_undo_pct": 100.0 * p_undo / a,
                "feasible_pct": (100.0 * feas_n / feas_tot) if feas_tot else None,
                "viol_mean": (viol_sum / feas_tot) if feas_tot else None,
            })
            acc_n = imm = win = forced = dead = 0
            A_sum = down_sum = p_undo = 0.0
            feas_n = feas_tot = 0; viol_sum = 0.0

    D = [int((X[i] != X[j]).sum())
         for i in range(mc) for j in range(i + 1, mc)]
    Ah = np.array(A_all) if A_all else np.array([0])
    F = float(flips.mean())
    # a free random walk on n bits reaches this after F flips
    walk = (n / 2.0) * (1.0 - np.exp(-2.0 * F / n)) if n else 0.0
    return {
        "n": n, "mc": mc, "T": T, "steps": steps, "window": window,
        "buckets": out_buckets,
        "A_median": float(np.median(Ah)),
        "A_p10": float(np.percentile(Ah, 10)),
        "A_p90": float(np.percentile(Ah, 90)),
        "flips": F,
        "hamming": float((X != X_start).sum(axis=1).mean()),
        "hamming_random_walk": float(walk),
        "distinct": float(touched.sum(axis=1).mean()),
        "distinct_pct": 100.0 * float(touched.sum(axis=1).mean()) / n,
        "pair_hamming": float(np.mean(D)) if D else 0.0,
        "selected_mean": float(X.sum(axis=1).mean()),
        "viol_max": viol_max if violations is not None else None,
    }


def report(d):
    """Format a diagnose() result. Read it top down: what the step looks
    like, whether the chain moves, whether it stays feasible."""
    L = []
    L.append("n=%d  mc=%d  T=%.4f  %d steps (%.2f sweeps)"
             % (d["n"], d["mc"], d["T"], d["steps"], d["steps"] / d["n"]))
    L.append("")
    hdr = ("  step      flips  Hamming  distinct   meanA  down  acc%  rej%"
           "  imm%  wnd%  frc%")
    if d["buckets"] and d["buckets"][0]["feasible_pct"] is not None:
        hdr += "  feas%"
    L.append(hdr)
    for b in d["buckets"]:
        line = ("  %6d  %9.0f  %7.0f  %8.0f  %6.2f %5.2f %5.1f %5.2f %5.2f"
                " %5.2f %5.2f"
                % (b["step"], b["flips"], b["hamming"], b["distinct"],
                   b["A_mean"], b["down_mean"], b["accepted_pct"],
                   b["rejected_pct"], b["immediate_pct"], b["window_pct"],
                   b["forced_pct"]))
        if b["feasible_pct"] is not None:
            line += " %6.2f" % b["feasible_pct"]
        L.append(line)
    L.append("")
    L.append("admissible set A      median %.0f   p10 %.0f   p90 %.0f"
             % (d["A_median"], d["A_p10"], d["A_p90"]))
    L.append("  A is the real control variable: the selection is uniform over")
    L.append("  it, so the chance of picking any particular move is 1/A.")
    L.append("")
    churn = d["hamming"] / max(d["flips"], 1.0)
    L.append("displacement          %.0f bits from %.0f flips  (%.4f per flip)"
             % (d["hamming"], d["flips"], churn))
    L.append("  a free random walk would have reached %.0f"
             % d["hamming_random_walk"])
    L.append("distinct touched      %.0f of %d  (%.1f%% of the problem)"
             % (d["distinct"], d["n"], d["distinct_pct"]))
    L.append("pairwise trial dist   %.0f   (mean %.0f bits set per trial)"
             % (d["pair_hamming"], d["selected_mean"]))
    if d["viol_max"] is not None:
        L.append("violations            max ever %.0f" % d["viol_max"])
    L.append("")
    L.append("reading it:")
    if churn < 0.05:
        L.append("  * cycling, not exploring — escape mechanisms act inside")
        L.append("    the region the chain already occupies and will not help")
    else:
        L.append("  * the chain covers new ground; displacement tracks flips")
    if d["buckets"]:
        last = d["buckets"][-1]
        if last["rejected_pct"] < 2.0:
            L.append("  * the chain is never stuck (%.2f%% fully rejected), so"
                     % last["rejected_pct"])
            L.append("    E_Offset is effectively a dead parameter here")
        if last["forced_pct"] > 20.0:
            L.append("  * the undo is often the only downhill move (%.1f%%) —"
                     % last["forced_pct"])
            L.append("    but blocking it forces a SECOND removal, which may")
            L.append("    be worse than the undo. Measure before building")
        if last["feasible_pct"] is not None and last["feasible_pct"] > 99.9:
            L.append("  * the feasible manifold is absorbing — the chain never")
            L.append("    crosses infeasible ground, so it cannot reach a")
            L.append("    region that requires passing through one")
    return "\n".join(L)
