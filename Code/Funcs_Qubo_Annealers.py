"""
Funcs_Qubo_Annealers — batched Digital Annealing kernel (QUBO only, numpy).

DA semantics identical to the reference (DigitalAnnealing in Funcs_Annealers.py):

    E_Offset = 0
    per step:
        T = Ts[step]
        test all n flips: dE_i <= 0 OR Metropolis(dE_i - E_Offset, T)
        pick ONE of the admissible ones uniformly and flip it; E_Offset = 0
        none admissible -> E_Offset += offset_increase_rate

Deliberate differences from the reference, each with a measured reason:
  - all num_MC trials run simultaneously as an (mc,n) batch (one flip per
    step and trial is preserved — that is the DA semantics!), with the
    acceptance scan vectorised
  - Metropolis via the standard_exponential trick (see Funcs_Qubo_Randomizers)
  - dE from the gradient G = X @ A instead of iterating monomials:
        dE = (1-2X) * (b + 2G),  flipping k: G += s_k * A[k,:]
  - G and E are periodically recomputed exactly by matvec (float64 drift)
  - min tracking is seeded WITH the initial state (the reference's empty
    Min_varAssignement bug is not reproduced — with a warm start it would be
    the normal case rather than an edge case)

The RNG streams naturally differ from the reference; the comparison is made
over E/dE to machine precision and over trajectory statistics
(Test_Compare_Reference.py), not over identical paths.
"""

import Funcs_Qubo_Annealing3 as qa3
from Funcs_Qubo_Randomizers import BulkRandomizer


def digital_annealing_batch(A, b, c, X0, Ts, offset_increase_rate: float,
                            randomizer: BulkRandomizer,
                            save_addinfo: bool = True,
                            track_offsets: bool = False,
                            recompute_every: int = 1024):
    """
    Run all Monte-Carlo trials of one batch simultaneously.

    A, b, c : QUBO in canonical form (A dense or scipy CSR)
    X0      : (mc, n) binary start states (one row per trial)
    Ts      : (steps,) cooling schedule
    offset_increase_rate : E_Offset increment on a fully rejected step
    track_offsets        : record the E_Offset trajectories (for the visualiser)
    recompute_every      : recompute G and E exactly every k steps (numerical
                           hardening)

    Returns a dict:
      "Trajectories" (mc, steps+1) float64 — E over time, incl. start energy
      "E_min"        (mc,)
      "X_min"        (mc, n) int8      (only if save_addinfo)
      "X_final"      (mc, n) int8      — the state the chains ended on, which
                     is NOT the best one; needed to continue a chain across
                     phases (e.g. a soft/hard penalty cycle) where X_min from
                     one phase is measured on a different energy scale
      "Offsets"      (mc, steps)       (only if track_offsets)
    """
    xp = qa3.xp
    sparse = qa3.is_sparse(A)
    b = xp.asarray(b, dtype=xp.float64)
    Ts = xp.asarray(Ts, dtype=xp.float64)
    steps = len(Ts)

    X = xp.asarray(X0, dtype=xp.int8).copy()
    mc, n = X.shape
    Xf = X.astype(xp.float64)
    G = xp.asarray(Xf @ A)                              # (mc,n) gradient
    E = (Xf * G).sum(axis=1) + Xf @ b + c               # (mc,)
    dE = (1.0 - 2.0 * Xf) * (b + 2.0 * G)

    offs = xp.zeros(mc, dtype=xp.float64)
    vals = xp.empty((mc, steps + 1), dtype=xp.float64)
    vals[:, 0] = E
    offsets_traj = xp.empty((mc, steps), dtype=xp.float64) if track_offsets else None

    # seed min tracking WITH the initial state
    E_min = E.copy()
    X_min = X.copy() if save_addinfo else None

    if sparse:
        indptr, indices, data = A.indptr, A.indices, A.data

    rows_all = xp.arange(mc)
    for step in range(steps):
        T = Ts[step]
        if track_offsets:
            offsets_traj[:, step] = offs

        # acceptance scan: dE - offset < T*eps  (dE<=0 is included automatically);
        # reference edge case T<=0: only downhill flips are admissible.
        if T > 0.0:
            eps = randomizer.exponentials(mc, n)
            acc = dE < (offs[:, None] + T * eps)
        else:
            acc = dE <= 0.0

        count = acc.sum(axis=1)
        has = count > 0
        if has.any():
            # pick the r-th admissible flip uniformly (per trial)
            u = randomizer.uniforms(mc)
            r = xp.minimum((u * count).astype(xp.int64),
                           xp.maximum(count - 1, 0))
            cs = acc.cumsum(axis=1)
            k = (cs > r[:, None]).argmax(axis=1)

            rows = rows_all[has]
            kk = k[has]
            sgn = 1.0 - 2.0 * X[rows, kk].astype(xp.float64)
            E[rows] += dE[rows, kk]
            X[rows, kk] = 1 - X[rows, kk]
            if sparse:
                for r_i, k_i, s_i in zip(rows.tolist(), kk.tolist(), sgn.tolist()):
                    lo, hi = indptr[k_i], indptr[k_i + 1]
                    G[r_i, indices[lo:hi]] += s_i * data[lo:hi]
            else:
                G[rows] += sgn[:, None] * A[kk, :]
            dE[rows] = (1.0 - 2.0 * X[rows].astype(xp.float64)) * (b + 2.0 * G[rows])
            offs[rows] = 0.0
        offs[~has] += offset_increase_rate

        vals[:, step + 1] = E
        if save_addinfo:
            imp = E <= E_min
            if imp.any():
                E_min[imp] = E[imp]
                X_min[imp] = X[imp]
        else:
            xp.minimum(E_min, E, out=E_min)

        # numerical hardening: reconstruct G and E exactly, periodically
        if recompute_every and (step + 1) % recompute_every == 0:
            Xf = X.astype(xp.float64)
            G = xp.asarray(Xf @ A)
            E = (Xf * G).sum(axis=1) + Xf @ b + c
            dE = (1.0 - 2.0 * Xf) * (b + 2.0 * G)

    out = {"Trajectories": vals, "E_min": E_min, "X_final": X}
    if save_addinfo:
        out["X_min"] = X_min
    if track_offsets:
        out["Offsets"] = offsets_traj
    return out
