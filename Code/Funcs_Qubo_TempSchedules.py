"""
Funcs_Qubo_TempSchedules — cooling schedule generation (numerically robust).

Mirror of Generate_Cooling_Schedule from the reference library
(../../annealing-cop-approximator/Code/Funcs_Annealing2.py), QUBO only.
The calibrated "da_gp" schedule is carried over unchanged: scan correction
T = dE/ln(n/p), d fixed at 2.22, gamma cap on the landscape depth. That
calibration is tuned for ONE problem type (warm-started Gram bisection) — do
not change its semantics without measuring.

cooling_param contract for "da_gp" (identical to the reference):
    [0] "da_gp"
    [1] n_steps
    [2] delta_E_vec   (None -> the solver fills it at the REAL start point)
    [3] start_assign  (warm-start / Lloyd assignment, for imbalance correction)
    [4] E_warm        (energy at the warm start, "E_Lloyd")
    [5] E_start       (energy of a random state, for the depth D)
    [6] p_hot   [7] p_cold   [8] gamma
    [9] c cache  [10] d cache
"""

import numpy as np


def generate_cooling_schedule(cooling_param: list, steps: int,
                              imbalance_correction: bool = False,
                              hold_steps: int = 0):
    """
    Build the (steps,) temperature array from cooling_param.

    Types: constant, linear, rising, logarithmic, logarithmic_step,
           exponential, geometric, hyperbolic, sigma, da_gp.
    imbalance_correction: centre the per-side medians of the dE vector before
    taking quantiles (currently disabled in the reference — toggled here).

    hold_steps: a PLATEAU at the start — the initial temperature T(1) is held
    constant for `hold_steps` steps before the cooling curve begins. The
    plateau is ADDED ON TOP: the result is (steps + hold - 1) long, the
    cooling curve keeps its full `steps` support points and therefore exactly
    the same end temperature as without a plateau. 0 or 1 means off (at t=1
    the curve sits at T_start anyway). Purpose: without it, T already falls
    from step 2 and the annealer barely leaves the warm-start basin; the
    plateau gives it a real exploration phase. Cost: the run gets `hold` steps
    longer. The da_gp CALIBRATION itself is untouched.
    """
    h = max(1, int(hold_steps))
    steps_eff = steps                     # full cooling curve, plateau prepended
    n_out = steps + h - 1
    T = []
    for step in range(1, n_out + 1):
        t = max(1, step - h + 1)          # Zeitachse MIT Plateau
        kind = cooling_param[0]
        if kind == "constant":
            T.append(cooling_param[1])
        elif kind == "linear":
            T_start, T_end = cooling_param[1], cooling_param[2]
            T.append(T_start + t * (T_end - T_start) / steps_eff)
        elif kind == "rising":
            T.append(cooling_param[1] * t)
        elif kind == "logarithmic":
            c = cooling_param[1]
            T.append(c / (np.log(1 + pow(t, 2.22))))
        elif kind == "logarithmic_step":
            c = cooling_param[1]
            for _ in range(cooling_param[2]):
                T.append(c / (np.log(1 + pow(t, 2.22))))
        elif kind == "exponential":
            c = cooling_param[1]
            T.append(np.exp(t / c) - 1)
        elif kind == "geometric":
            T0, alpha = cooling_param[1], cooling_param[2]
            T.append(T0 * alpha ** (t - 1))
        elif kind == "sigma":
            # Temperatures in units of sigma = std(dE) of THIS instance, so
            # the same schedule transfers between problems. An absolute T does
            # not: it carries the scale of the coefficients, and getting it
            # wrong is the single most expensive mistake this engine allows.
            #
            # The cap at 0.5*sigma is deliberate. A uniformly random start
            # state IS the equilibrium at T = infinity, so a hotter phase
            # cannot reach anything a random restart does not already give,
            # and above sigma the DA scan has no advantage over a random walk.
            if step == 1:
                _calibrate_sigma(cooling_param)
            T_hi, T_lo, form, K = (cooling_param[6], cooling_param[7],
                                   cooling_param[8], cooling_param[9])
            u = (t - 1) / max(steps_eff - 1, 1)             # 0 .. 1
            if form == "linear":
                T.append(T_hi + (T_lo - T_hi) * u)
            elif form == "staircase":
                # geometric levels held constant, so each level can be checked
                # for equilibrium and its length tuned to the relaxation time
                lev = min(int(u * K), K - 1)
                T.append(T_hi * (T_lo / T_hi) ** (lev / max(K - 1, 1)))
            else:                                            # geometric
                T.append(T_hi * (T_lo / T_hi) ** u)
        elif kind == "hyperbolic":
            T0 = cooling_param[1]
            T.append(T0 / t)
        elif kind == "da_gp":
            if step == 1:
                _calibrate_da_gp(cooling_param, imbalance_correction)
            c, d = cooling_param[9], cooling_param[10]
            T.append(c / (np.log(1 + pow(t, d))))
        elif kind == "da_gp_floor":
            # Like da_gp, but the curve approaches the freezing bound
            # instead of dropping below it (see _calibrate_da_gp, slot 11):
            #     T(t) = T_freeze + (T_hot - T_freeze) * ln2/ln(1+t^d)
            # t=1   -> exactly T_hot      (ln(1+1) = ln2)
            # t=S   -> just above T_freeze, never under it
            # Both boundary conditions hold exactly; d now only controls the
            # curvature in between, no longer the end temperature.
            if step == 1:
                _calibrate_da_gp(cooling_param, imbalance_correction)
            d = cooling_param[10]
            T_hot_c, T_frz = cooling_param[11], cooling_param[12]
            T.append(T_frz + (T_hot_c - T_frz)
                     * np.log(2.0) / np.log(1 + pow(t, d)))
        else:
            raise ValueError("unknown cooling schedule: %r" % (kind,))
    T = np.asarray(T[:n_out], dtype=np.float64)
    if h > 1:
        print("   plateau: %d steps at T_start=%.4g, then the full %d "
              "cooling steps -> %d steps total, T_end=%.4g"
              % (h, T[0], steps_eff, len(T), T[-1]))
    return T


def _calibrate_sigma(cooling_param: list):
    """
    Fill the "sigma" contract in place and report it.

        [0] "sigma"
        [1] c_start   (default 0.3)   T_start = c_start * sigma
        [2] c_end     (default 0.11)  T_end   = c_end   * sigma
        [3] form      "geometric" (default) | "staircase" | "linear"
        [4] sigma     std(dE) of this instance — the SOLVER fills this, the
                      same way it fills slot 2 of "da_gp" with the dE vector
        [5] K         number of levels for "staircase" (default 10)
        [6] T_start cache  [7] T_end cache  [8] form cache  [9] K cache

    DOMAIN OF VALIDITY — measured, and narrower than it looks. sigma is the
    right unit only when the dE spectrum is HOMOGENEOUS:

        instance   n       T_opt   sigma     T_opt/sigma   spread(q50/min)
        G1         800     0.80      6.92    0.116          5
        G22       2000     0.60      4.47    0.134          3
        cdc7-4-3-2 11811   0.18    420.00    0.0004        15

    G1 and G22 agree to 16 %, and c_start=0.5 / c_end=0.05 brackets their
    measured optimum. cdc7 is off by a factor of 300: the defaults would put
    T_end at 0.05*420 = 21 against a measured optimum of 0.18, i.e. **117x too
    hot** — the exact failure that cost six wasted runs on another instance.

    Why: cdc7 is a penalty encoding with 210 conflicts per variable, so sigma
    is dominated by constraint-violating directions the chain never takes. It
    lives on the feasible manifold where dE is +-1, while sigma measures the
    whole spectrum including the half that is never used.

    So sigma does NOT dissolve the spread question, it reproduces it. Use it
    where spread = q50(dE+)/min(dE+) is small (1..5, Max-Cut and friends); on
    a gapped spectrum keep the scan correction T = dE_min/ln(n/p) instead.
    A sigma taken over only the reachable part of the spectrum would be the
    principled fix and has not been built.

    WHERE THE DEFAULTS COME FROM — measured on G1 and G22, 10 000 steps,
    seed 17, a 5x4 and a 3x4 window grid (43 runs). Mean cut per c_end,
    pooled over every c_start in the grid:

        c_end    0.01    0.02    0.05    0.10    0.12    0.15    0.20
        G1      11595   11613   11607   11621   11621   11607   11588
        G22     13183   13210   13233   13238   13234   13207   13132

    Unimodal on both, peaking at c_end ~ 0.10..0.12 — which is exactly the
    measured constant-T optimum, T_opt/sigma = 0.116 on G1 and 0.134 on G22.
    So the rule is: **cool down TO the optimal temperature and stop there, do
    not cool through it.** The originally guessed 0.05 was a factor of two too
    cold, and 0.01 costs ~30 cut on G1 and ~55 on G22.

    c_start is far less critical: 0.15 through 0.5 all perform within noise.
    Only 0.1 collapses, and that is the row that STARTS below T_opt — the
    lower edge sits exactly where it should.

    Best results from that grid: G1 11 624, which MATCHES the best known
    value, and G22 13 256 (99.23 %), against 11 591 and 13 154 documented in
    README.md at ten times the step count.

    Calibrated on two Max-Cut instances. Two points, one problem class, one
    seed per cell — treat 0.3/0.11 as a measured starting point, not a
    constant of nature.

    Why a geometric sweep at all: the DA speed-up over a single-flip random walk depends
    only on T/sigma. Measured for n <= 7, half the gain is left at 0.37 sigma,
    90 % at 0.16 sigma, and it saturates (factor n) below 0.07 sigma. A
    geometric curve from 0.5 to 0.05 sigma spends about half its budget above
    0.16 sigma and 30 % below 0.1 sigma; a linear one would spend 75 % and
    11 %, i.e. most of the run where the scan buys nothing.
    """
    c_start = cooling_param[1] if len(cooling_param) > 1 else None
    c_end = cooling_param[2] if len(cooling_param) > 2 else None
    form = cooling_param[3] if len(cooling_param) > 3 else None
    sigma = cooling_param[4] if len(cooling_param) > 4 else None
    K = cooling_param[5] if len(cooling_param) > 5 else None

    c_start = 0.3 if c_start is None else float(c_start)
    c_end = 0.11 if c_end is None else float(c_end)
    form = "geometric" if form is None else str(form)
    K = 10 if K is None else max(1, int(K))
    if form not in ("geometric", "staircase", "linear"):
        raise ValueError("sigma schedule: unknown form %r "
                         "(geometric | staircase | linear)" % (form,))
    if sigma is None:
        raise ValueError(
            "sigma schedule: slot 4 (sigma) is empty. qubo_min_solver fills "
            "it from the problem; when calling generate_cooling_schedule "
            "directly, pass it as [\"sigma\", c_start, c_end, form, sigma].")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma schedule: sigma must be finite and positive, "
                         "got %r — a constant PBF has no energy scale" % sigma)

    while len(cooling_param) < 10:
        cooling_param.append(None)
    cooling_param[6] = c_start * sigma
    cooling_param[7] = c_end * sigma
    cooling_param[8] = form
    cooling_param[9] = K
    print("   sigma schedule: sigma=%.6g -> T_start=%.6g (%.3g sigma), "
          "T_end=%.6g (%.3g sigma), form=%s%s"
          % (sigma, cooling_param[6], c_start, cooling_param[7], c_end, form,
             ", %d levels" % K if form == "staircase" else ""))


def _calibrate_da_gp(cooling_param: list, imbalance_correction: bool):
    """One-off da_gp calibration — carried over from the reference library."""
    n_steps = cooling_param[1]
    delta_E_vec = np.asarray(cooling_param[2], dtype=float)
    start_assign = np.asarray(cooling_param[3], dtype=int)
    E_warm = cooling_param[4]
    E_start = cooling_param[5]

    if len(cooling_param) > 8:
        p_hot, p_cold, gamma = cooling_param[6], cooling_param[7], cooling_param[8]
    else:
        p_hot = 0.3     # acceptance of a median uphill flip at t=1
        p_cold = 0.01   # residual acceptance of a small (q25) flip at the end
        gamma = 0.05    # cap: T(1) <= gamma * landscape depth

    # Imbalance correction: an unbalanced split produces a SYSTEMATIC error
    # gradient per side through the balance terms — centre the per-side
    # medians on their common mean.
    dE = delta_E_vec
    bias = 0.0
    split_diff = 0
    if imbalance_correction:
        if len(start_assign) == len(dE) and 0 < start_assign.sum() < len(dE):
            split_diff = int(2 * start_assign.sum() - len(dE))   # n1 - n0
            m0 = np.median(dE[start_assign == 0])
            m1 = np.median(dE[start_assign == 1])
            mid = 0.5 * (m0 + m1)
            bias = 0.5 * abs(m1 - m0)
            dE = dE.copy()
            dE[start_assign == 0] += mid - m0
            dE[start_assign == 1] += mid - m1

    up = dE[dE > 0]
    if len(up) >= 4:
        q25, q50 = np.quantile(up, [0.25, 0.5])
    else:
        # fallback: ~n/2 flips bridge the depth D
        D = max(E_start - E_warm, 1e-12)
        q50 = 2.0 * D / max(len(delta_E_vec), 1)
        q25 = 0.5 * q50

    # Scan correction: the DA tests ALL n flips per step and accepts as soon
    # as ONE gets through. For a per-STEP acceptance of p, the per-flip
    # acceptance must be ~ p/n:  exp(-dE/T) = p/n  ->  T = dE / ln(n/p).
    n_vars = max(len(delta_E_vec), 2)
    T_hot = q50 / np.log(n_vars / p_hot)
    T_freeze = max(q25, 1e-12) / np.log(n_vars / p_cold)

    # d is FIXED: solving it from two points would give d < 1 at small spread
    # (square-root shape, far too flat). Steep log cooling is the proven form.
    d = 2.22
    c = min(T_hot * np.log(2.0),
            T_freeze * np.log(1.0 + n_steps ** d))
    D = max(E_start - E_warm, 0.0)
    if D > 0:
        c = min(c, gamma * D * np.log(2.0))     # never melt the warm start

    # Curve start: T_hot, but never above the gamma cap and never below the
    # freezing bound (otherwise the curve would not be monotonically falling)
    T_hot_capped = c / np.log(2.0)
    T_frz_target = min(T_freeze, T_hot_capped)

    while len(cooling_param) < 13:
        cooling_param.append(0.0)
    cooling_param[9] = c
    cooling_param[10] = d
    cooling_param[11] = T_hot_capped      # start value for da_gp_floor
    cooling_param[12] = T_frz_target      # target value (freezing bound)
    print("da_gp: q50=%.4g q25=%.4g (diff=%d, bias=%.4g) -> "
          "c=%.4g d=%.3g  (T1=%.4g, T_end=%.4g | freezing bound %.4g)"
          % (q50, q25, split_diff, bias, c, d,
             T_hot_capped, c / np.log(1.0 + n_steps ** d), T_freeze))


def auto_offset_rate(delta_E_vec, start_assign=None, k_escape: float = 25.0):
    """
    "auto_gp" offset: E_Offset rate on the measured delta-E scale. The
    barrier B ~ q50 of the uphill flips, escape after ~k_escape rejected
    steps -> B/k_escape. The same imbalance bias as in the cooling is removed
    first (active here, as in the reference optimizer).
    """
    dE = np.asarray(delta_E_vec, dtype=float)
    if start_assign is not None:
        la = np.asarray(start_assign, dtype=int)
        if len(la) == len(dE) and 0 < la.sum() < len(la):
            m0 = np.median(dE[la == 0])
            m1 = np.median(dE[la == 1])
            mid = 0.5 * (m0 + m1)
            dE = dE.copy()
            dE[la == 0] += mid - m0
            dE[la == 1] += mid - m1
    up = dE[dE > 0]
    rate = float(np.quantile(up, 0.5)) / k_escape if len(up) else 0.0
    print("auto_gp offset: q50(dE+)/%g -> offset_increase_rate = %.6g"
          % (k_escape, rate))
    return rate
