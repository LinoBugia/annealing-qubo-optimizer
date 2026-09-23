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
           exponential, geometric, hyperbolic, da_gp.
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
