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

# ── sigma-calibrated offset kick ─────────────────────────────────────────────
#
# The plain E_Offset grows by a fixed rate per empty step. At low T the chain
# then escapes almost exclusively through directions with dE_k < g_min + rate,
# i.e. through the SMALLEST gap, because the offset creeps up from zero and the
# smallest barrier is cleared first. The width of the escape is whatever the
# rate happens to make it.
#
# The kick sets that width explicitly instead, in units of sigma = std(dE), and
# mixes several widths. It is a separate, optional mechanism: the existing
# offset_increase_rate is untouched and the two can be used together or apart.
#
# One thing the kernel gets for free: an empty step IS a strict local minimum.
# Acceptance is `dE < offs + T*eps` with offs >= 0 and T*eps > 0, so every
# dE_k <= 0 is unconditionally admissible. If nothing was admissible, every
# dE_k must have been positive. No separate check is needed.
KICK_DEFAULTS = dict(
    w=3,                    # consecutive empty steps before the kick may fire
    c_low=0.1,              # narrow escape:  E_off = g_min + c_low  * sigma
    c_wide=1.0,             # wide escape:    E_off = g_min + c_wide * sigma
    c_b1=0.5, c_b2=1.5,     # band: g_min + c_b1*sigma <= dE_k <= g_min + c_b2*sigma
    p=(0.2, 0.4, 0.3, 0.1),  # none, low, wide, band  (must sum to 1)
    phases=None,            # [(fraction, mode), ...] — overrides p, see below
    window=(0.0, 1.0),      # fraction of the run over which the kick is active
    check_every=16,         # test the trigger only every k steps — see below
)

# `phases` is the plan-driven alternative to drawing a mode from `p`. It is a
# list of (fraction_of_the_run, mode) and is resolved ONCE, before the loop,
# into a per-step mode array — so the whole policy is fixed from outside and
# the step itself only reads an integer. It is also cheaper than `p`: the mode
# no longer varies per trial, so one of the two RNG draws disappears.
#
#     phases=[(0.5, "none"), (0.25, "wide"), (0.25, "low")]
#
# Alternation needs no extra machinery — it is just more phases:
#
#     phases=[(0.4, "none"), (0.1, "wide"), (0.1, "none"), (0.1, "wide"), ...]
#
# The three that matter, and what they are for:
#   none   no offset, thermal escape only. The right opening phase: early on
#          the chain is far from any minimum and blocked steps are rare.
#   wide   E_off = g_min + c_wide*sigma, a broad escape. Works well when no
#          penalty terms are present; alternate it with none.
#   low    E_off = g_min + c_low*sigma, a narrow one just above the smallest
#          gap.
#
# Why check_every exists: testing the trigger on EVERY step costs two RNG
# draws plus a handful of (mc,) operations, and those are dominated by numpy
# call overhead rather than by arithmetic. On a large instance that is noise
# against a 1.7 ms step; on an 800-variable one, where a step costs ~0.1 ms,
# it is a double-digit percentage. Checking every k steps amortises it away.
#
# The cost is that a kick fires up to k-1 steps late. The trigger is "w
# consecutive blocked steps", and a chain that is stuck stays stuck, so a late
# kick is the same kick — only the deadline moves. `empty` is still counted
# every step (one fused expression) because a *missed* blocked step would
# corrupt the count itself.


def _kick_step(dE, offs, empty, fire, cfg, mode_src, u_pick, xp):
    """
    Apply the kick to the trials in `fire`. Returns the forced-flip indices.

    low/wide only RAISE the offset and let the normal DA rule run: everything
    with dE_k <= E_off is certain, the rest keeps its Metropolis chance, and
    the pick stays uniform over the admissible set. band is the one mode that
    breaks that — it picks directly inside the band, which is the only way to
    strictly prefer the higher directions over the smallest gap.
    """
    rows = xp.flatnonzero(fire)
    sig = cfg["sigma"]
    g = dE[rows].min(axis=1)
    if xp.isscalar(mode_src) or getattr(mode_src, "ndim", 1) == 0:
        mode = xp.full(rows.size, int(mode_src), dtype=xp.int64)   # phase plan
    else:
        cum = xp.cumsum(xp.asarray(cfg["p"], dtype=xp.float64))
        mode = xp.minimum(xp.searchsorted(cum, mode_src[rows], side="right"), 3)

    forced = xp.full(rows.size, -1, dtype=xp.int64)
    new_off = xp.zeros(rows.size, dtype=xp.float64)
    new_off = xp.where(mode == 1, g + cfg["c_low"] * sig, new_off)
    new_off = xp.where(mode == 2, g + cfg["c_wide"] * sig, new_off)

    bnd = mode == 3
    if bnd.any():
        br = rows[bnd]
        gb = g[bnd]
        d = dE[br]
        inb = (d >= (gb + cfg["c_b1"] * sig)[:, None]) & \
              (d <= (gb + cfg["c_b2"] * sig)[:, None])
        cnt = inb.sum(axis=1)
        hit = cnt > 0
        r = xp.minimum((u_pick[br] * cnt).astype(xp.int64),
                       xp.maximum(cnt - 1, 0))
        pick = (inb.cumsum(axis=1) > r[:, None]).argmax(axis=1)
        idx = xp.flatnonzero(bnd)
        forced[idx[hit]] = pick[hit]
        # empty band -> fall back to wide, as specified
        nb = idx[~hit]
        new_off[nb] = g[nb] + cfg["c_wide"] * sig

    offs[rows] = new_off            # mode 0 ("none") leaves it at 0
    empty[rows] = 0                 # every mode resets the counter
    out = xp.full(empty.size, -1, dtype=xp.int64)
    out[rows] = forced
    return out


def digital_annealing_batch(A, b, c, X0, Ts, offset_increase_rate: float,
                            randomizer: BulkRandomizer,
                            save_addinfo: bool = True,
                            track_offsets: bool = False,
                            recompute_every: int = 1024,
                            kick: dict = None):
    """
    Run all Monte-Carlo trials of one batch simultaneously.

    A, b, c : QUBO in canonical form (A dense or scipy CSR)
    X0      : (mc, n) binary start states (one row per trial)
    Ts      : (steps,) cooling schedule
    offset_increase_rate : E_Offset increment on a fully rejected step
    track_offsets        : record the E_Offset trajectories (for the visualiser)
    recompute_every      : recompute G and E exactly every k steps (numerical
                           hardening)
    kick                 : optional sigma-calibrated offset kick (see
                           KICK_DEFAULTS). None = off, and the step loop is
                           then byte-identical to before. Requires "sigma";
                           every other key falls back to its default.

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

    kcfg = None
    if kick is not None:
        kcfg = dict(KICK_DEFAULTS)
        kcfg.update(kick)
        if "sigma" not in kcfg or not (kcfg["sigma"] > 0.0):
            raise ValueError("kick: 'sigma' must be given and positive "
                             "(qa3.delta_e_sigma(A, b))")
        p = xp.asarray(kcfg["p"], dtype=xp.float64)
        if p.size != 4 or abs(float(p.sum()) - 1.0) > 1e-9 or (p < 0).any():
            raise ValueError("kick: 'p' must be 4 non-negative probabilities "
                             "(none, low, wide, band) summing to 1")
        kcfg["p"] = p
        w0, w1 = kcfg["window"]
        k_lo, k_hi = int(w0 * steps), int(w1 * steps)
        k_every = max(1, int(kcfg["check_every"]))
        empty = xp.zeros(mc, dtype=xp.int64)    # consecutive empty steps
        mode_of_step = None
        if kcfg["phases"]:
            names = ("none", "low", "wide", "band")
            fr = [float(f) for f, _ in kcfg["phases"]]
            if any(f < 0 for f in fr) or sum(fr) <= 0:
                raise ValueError("kick: phase fractions must be non-negative "
                                 "and not all zero")
            mode_of_step = xp.zeros(steps, dtype=xp.int64)
            edge = 0
            tot = sum(fr)
            for i, (f, nm) in enumerate(kcfg["phases"]):
                if nm not in names:
                    raise ValueError("kick: unknown phase mode %r (%s)"
                                     % (nm, ", ".join(names)))
                end = steps if i == len(kcfg["phases"]) - 1 else \
                    edge + int(round(steps * float(f) / tot))
                mode_of_step[edge:end] = names.index(nm)
                edge = min(end, steps)

    rows_all = xp.arange(mc)
    for step in range(steps):
        T = Ts[step]

        forced = None
        if kcfg is not None and step % k_every == 0 and k_lo <= step < k_hi:
            # Drawn here rather than per step: the two uniforms plus the
            # trigger test are numpy-call-overhead dominated, so on a small
            # instance they would show up in the step time. Both are drawn
            # unconditionally at a check point so the RNG stream does not
            # depend on whether a kick fires — otherwise two runs with the
            # same seed would diverge for reasons invisible in the output.
            # With a phase plan the mode is a scalar read from the plan, so
            # only the band pick needs randomness; without one it is drawn per
            # trial. Both are drawn at every check point regardless of whether
            # a kick fires, so the stream never depends on the outcome.
            if mode_of_step is None:
                mode_src = randomizer.uniforms(mc)
                u_pick = randomizer.uniforms(mc)
            else:
                mode_src = mode_of_step[step]
                u_pick = randomizer.uniforms(mc)
            fire = empty >= kcfg["w"]
            if fire.any():
                forced = _kick_step(dE, offs, empty, fire, kcfg,
                                    mode_src, u_pick, xp)

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
        if forced is not None:
            has = has | (forced >= 0)       # band picks directly, bypassing acc
        if has.any():
            # pick the r-th admissible flip uniformly (per trial)
            u = randomizer.uniforms(mc)
            r = xp.minimum((u * count).astype(xp.int64),
                           xp.maximum(count - 1, 0))
            cs = acc.cumsum(axis=1)
            k = (cs > r[:, None]).argmax(axis=1)
            if forced is not None:
                k = xp.where(forced >= 0, forced, k)

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
        if kcfg is not None:
            empty = (empty + 1) * (~has)    # one fused op, not two masked ones

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
