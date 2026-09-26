"""
Funcs_Qubo_Optimizers — qubo_min_solver + visualisation + CSV persistence.

Mirror of pbf_min_solver from the reference library
(../../annealing-cop-approximator/Code/Funcs_Optimizers.py), QUBO only.

New relative to the reference:
  - initial_varAssignements_pre accepts a LIST of start vectors; num_MC
    Monte-Carlo trials are run for EACH of them (all trials of one start
    group run as a single (num_MC, n) batch).
  - E_Offset trajectories are genuinely tracked during the run (the
    visualiser no longer reconstructs them from deltaE==0).
  - Min tracking is seeded with the initial state, so it never returns empty.

The visualiser and the Runs/ CSV layout are carried over from the reference.
"""

import contextlib
import csv
import datetime
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from Funcs_Qubo_Annealers import digital_annealing_batch
from Funcs_Qubo_Annealing3 import (eval_qubo, eval_delta_energy,
                                   delta_e_sigma as qa3_delta_e_sigma)
from Funcs_Qubo_ProblemGeneration import random_start_states
from Funcs_Qubo_Randomizers import BulkRandomizer
from Funcs_Qubo_TempSchedules import generate_cooling_schedule, auto_offset_rate

pio.renderers.default = "browser"


def _rgba(hexcol, alpha):
    """'#1616A7' -> 'rgba(22,22,167,alpha)'; None if not a hex colour."""
    try:
        h = str(hexcol).lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return "rgba(%d,%d,%d,%.2f)" % (r, g, b, alpha)
    except Exception:
        return None



def _snap_equal_for_display(values, rtol: float = 1e-9, atol: float = 0.0):
    """
    Group values that are equal within a tolerance onto one representative —
    FOR DISPLAY ONLY. The returned array never replaces the real results.

    Why this exists: when every Monte-Carlo trial converges to the same
    solution, their energies still differ in the last few float64 digits,
    because each trial accumulates its incremental E updates in a different
    order. The Mins panel then auto-scales onto that 1e-12 spread and shows
    what looks like a dozen distinct solutions. Snapping within a tolerance
    makes identical solutions plot as identical.

    Clusters are anchored on their first member (no unbounded chaining), and
    each cluster is represented by its smallest value — these are minima, so
    the best value found is the honest representative.

    Returns (snapped, tol, n_clusters).
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v, 0.0, 0
    scale = float(np.max(np.abs(v)))
    tol = max(atol, rtol * (scale if scale > 0.0 else 1.0))
    order = np.argsort(v)
    sv = v[order]
    out = v.copy()
    start, clusters = 0, 0
    for k in range(1, len(sv) + 1):
        if k == len(sv) or sv[k] - sv[start] > tol:
            out[order[start:k]] = sv[start]      # cluster minimum
            clusters += 1
            start = k
    return out, tol, clusters


def VisualizeRuns(n_vars, type_alg, labels, Trajectories, Mins, ExecTimes, T,
                  Offsets=None, groups=None, title_suffix="",
                  min_display_rtol: float = 1e-9):
    """
    Same layout as the reference (digitalAnnealing branch):
    Energy / Delta Energy / Temp / (Exec Times, Mins) / E_Offset.

    ALL trials go into ONE figure. COLOUR = START GROUP: every Monte-Carlo
    trial starting from the same vector gets the same colour (groups[i] =
    start group of trial i; without it each trial counts as its own group).
    That shows at a glance which starting point runs into which basin. The
    legend holds one entry per group, and clicking it hides the whole group.

    Delta Energy and Mins are AGGREGATED PER START GROUP:
      - Delta Energy: one median curve per group plus a 25/75 quantile band,
        instead of num_MC individual noise curves. This makes each group's dE
        scale directly comparable with that group's T curve in the panel
        below — and that is the quantity its cooling was calibrated against.
      - Mins: a marker swarm per group at its own x position, i.e. the spread
        of final results for that starting point.
    Both hang off legendgroup, so one click hides trajectories, dE band,
    minima and offsets together.

    min_display_rtol: relative tolerance under which two minima are drawn as
    the SAME value (default 1e-9). Trials that reach the same solution still
    differ in the last float64 digits, because each accumulates its
    incremental energy updates in a different order; without this the Mins
    panel auto-scales onto that noise and suggests many distinct solutions
    where there is one.

    Why 1e-9 and not something looser: the drift is bounded by
    `recompute_every`, which rebuilds E exactly every k steps, so at most k
    incremental updates can accumulate. Measured on this project, eight
    trials on the same solution spread by ~2e-10 at |E| ~ 1.8e5 — a relative
    1e-15. A 1e-9 tolerance clears that by six orders of magnitude while
    staying far below the spacing of genuinely different optima. Going to
    1e-6 would mean an absolute tolerance of 1.0 on a problem with
    |E| ~ 1e6, which would merge distinct solutions.

    This affects the PLOT ONLY — `Mins` and the returned solutions keep their
    exact values, and the hover text shows the raw number to 12 digits. Set
    it to 0.0 to disable snapping entirely.
    """
    n_traj = len(Trajectories)
    groups = list(range(n_traj)) if groups is None else list(groups)
    uniq = sorted(set(groups))

    rows, cols = 5, 2
    subplot_titles = ("Energy", "Delta Energy (median + IQR per start group)",
                      "Temp", "Exec Times [s]", "Mins per start group",
                      "E_Offset")
    specs = [
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{}, {}],
        [{"colspan": 2}, None]]
    row_heights = [0.6, 0.1, 0.1, 0.1, 0.1]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles,
                        specs=specs, row_heights=row_heights)
    fig.update_layout(title_text="Evaluation " + str(type_alg)
                      + ", Optimization variables = " + str(n_vars)
                      + ", start groups: " + str(len(uniq))
                      + ", Trials: " + str(n_traj) + title_suffix)
    colors = []
    for i in range(100):
        k = np.mod(i, 23)
        colors.append(plotly.colors.qualitative.Dark24_r[k])

    seen: set = set()                    # first curve of a group carries the legend
    for i in range(n_traj):
        traj = np.asarray(Trajectories[i])
        xs = np.arange(len(traj))
        g = groups[i]
        color = colors[np.mod(uniq.index(g), 23)]
        lg = "S%s" % g
        first = g not in seen
        seen.add(g)
        fig.add_trace(go.Scatter(x=xs, y=traj, name="Start %s" % g,
                                 legendgroup=lg, showlegend=first,
                                 hovertext=labels[i],
                                 marker=dict(color=color)), row=1, col=1)
        if Offsets is not None:
            fig.add_trace(go.Scatter(x=xs[:-1], y=np.asarray(Offsets[i]),
                                     showlegend=False, legendgroup=lg,
                                     hovertext=labels[i],
                                     marker=dict(color=color)), row=5, col=1)
    # ── Delta Energy: one median curve per start group + 25/75 band ─────────
    #   num_MC individual curves on top of each other are just noise; what
    #   matters is the group's dE SCALE relative to its own T curve.
    idx_of = {g: [i for i in range(n_traj) if groups[i] == g] for g in uniq}
    for g in uniq:
        idx = idx_of[g]
        color = colors[np.mod(uniq.index(g), 23)]
        lg = "S%s" % g
        L = min(len(Trajectories[i]) for i in idx)
        if L < 2:
            continue
        D = np.stack([np.diff(np.asarray(Trajectories[i])[:L]) for i in idx])
        xs = np.arange(1, L)
        med = np.median(D, axis=0)
        band = _rgba(color, 0.18) if len(idx) > 1 else None
        if band is not None:
            q25 = np.quantile(D, 0.25, axis=0)
            q75 = np.quantile(D, 0.75, axis=0)
            fig.add_trace(go.Scatter(x=xs, y=q75, mode="lines",
                                     line=dict(width=0), showlegend=False,
                                     legendgroup=lg, hoverinfo="skip"),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=xs, y=q25, mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor=band, showlegend=False,
                                     legendgroup=lg, hoverinfo="skip"),
                          row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=med, name="dE Start %s" % g,
                                 legendgroup=lg, showlegend=False,
                                 hovertext="Start %s (%d Trials)"
                                           % (g, len(idx)),
                                 marker=dict(color=color)), row=2, col=1)

    # Temperature: with per-group calibration, one curve per start group,
    # in the SAME colour as that group's trajectories
    T_list = T if isinstance(T, (list, tuple)) and np.ndim(T[0]) > 0 else [T]
    for gi, T_g in enumerate(T_list):
        col = colors[np.mod(gi, 23)] if len(T_list) > 1 else None
        fig.add_trace(go.Scatter(x=np.arange(len(T_g)), y=np.asarray(T_g),
                                 name="T(S%d)" % uniq[gi] if len(T_list) > 1 else "T",
                                 legendgroup="S%s" % uniq[gi] if len(T_list) > 1 else None,
                                 showlegend=False,
                                 marker=dict(color=col) if col else None),
                      row=3, col=1)
    # ── Mins: a marker swarm per start group at its own x position ──────────
    #   This shows the spread of final results for ONE starting point and
    #   lets the groups be compared side by side.
    #   Minima equal within min_display_rtol are drawn at one value, so that
    #   trials which found the SAME solution do not look like different ones
    #   just because their float64 accumulation differs (display only — the
    #   hover text carries the exact number).
    mins_raw = np.asarray(Mins, dtype=float)
    if min_display_rtol and min_display_rtol > 0.0:
        mins_plot, snap_tol, n_clusters = _snap_equal_for_display(
            mins_raw, rtol=min_display_rtol)
    else:
        mins_plot, snap_tol, n_clusters = mins_raw, 0.0, len(set(mins_raw.tolist()))

    for g in uniq:
        idx = idx_of[g]
        gi = uniq.index(g)
        color = colors[np.mod(gi, 23)]
        xs = (np.full(len(idx), float(gi)) if len(idx) < 2
              else gi + np.linspace(-0.32, 0.32, len(idx)))
        y = [mins_plot[i] for i in idx]
        fig.add_trace(go.Scatter(x=xs, y=y, name="Min Start %s" % g,
                                 mode="markers", showlegend=False,
                                 legendgroup="S%s" % g,
                                 hovertext=["%s: %.12g" % (labels[i], mins_raw[i])
                                            for i in idx],
                                 marker=dict(color=color, size=5)),
                      row=4, col=2)
        # highlight the group's best result
        j = int(np.argmin(y))
        fig.add_trace(go.Scatter(x=[xs[j]], y=[y[j]], mode="markers",
                                 name="best S%s" % g, showlegend=False,
                                 legendgroup="S%s" % g,
                                 hovertext="group minimum %s: %.12g"
                                           % (g, mins_raw[idx[j]]),
                                 marker=dict(color=color, size=11,
                                             symbol="star",
                                             line=dict(width=1,
                                                       color="#222"))),
                      row=4, col=2)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(uniq))),
                     ticktext=["S%s" % g for g in uniq], row=4, col=2)

    # Every trial landed on the same solution: plotly would otherwise zoom
    # the y-axis onto float noise. Pin a readable window around the value and
    # say so, instead of showing a meaningless spread.
    if n_clusters == 1 and mins_plot.size:
        v = float(mins_plot[0])
        pad = max(abs(v) * 1e-3, 1e-9)
        fig.update_yaxes(range=[v - pad, v + pad], row=4, col=2)
        fig.add_annotation(row=4, col=2, x=0.5, xref="x domain",
                           y=0.97, yref="y domain", showarrow=False,
                           font=dict(size=10, color="#666"),
                           text="all %d trials equal within %.1e" % (n_traj, snap_tol))
    fig.add_trace(go.Scatter(x=list(range(len(ExecTimes))), y=ExecTimes,
                             name="Exec_Times", showlegend=False,
                             mode="markers",
                             marker=dict(color=[colors[np.mod(uniq.index(g), 23)]
                                                for g in groups])),
                  row=4, col=1)
    return fig


def qubo_min_solver(A, b, c=0.0, type_alg: str = "digitalAnnealing",
                    steps: int = 1000, num_MC: int = 8,
                    cooling_param=None, seed_rand=42,
                    seed_gen_initial_varAssignment=42,
                    save_csv=False, save_addinfo=True, visual_inst=False,
                    initial_varAssignements_pre=None,
                    offset_increase_rate=0.0, random_start=False,
                    cooling_per_group: bool = True, hold_steps: int = 0,
                    offset_k_escape: float = 25.0,
                    recompute_every: int = 1024, mem_budget_mb: float = 64.0,
                    kick: dict = None):
    """
    QUBO minimisation by batched Digital Annealing.

    A, b, c       : QUBO in canonical form (see Funcs_Qubo_Annealing3);
                    A dense (float) or scipy CSR
    cooling_param : as in the reference, e.g. ["logarithmic", c] or the
                    "da_gp" contract (slot 2 = None is filled here at the
                    real starting point); default ["logarithmic", 10]
    seed_rand     : int or list of ints (master seed of the RNG blocks)
    initial_varAssignements_pre :
        None            -> random start(s) according to random_start
        vector (n,)     -> one start group, num_MC trials from that vector
        list/array (S,n) -> S start groups x num_MC trials each
    random_start  : True -> each trial gets its own random start (only when
                    no start vectors were supplied)
    offset_increase_rate : number or "auto_gp" (derived from the dE scale)
    cooling_per_group :
        True (default) -> EACH start group gets its OWN schedule, calibrated
        at ITS start vector (dE vector, E_warm, imbalance correction and the
        auto_gp offset all come from the same group). Necessary because
        different starting points see landscapes of different steepness: the
        best warm-start solution sits deeper in the funnel and has the
        largest dE scale — its c is roughly 2x too hot for the flatter
        starting points and melts them.
        False -> one shared schedule calibrated on group 0 (the earlier
        behaviour; useful for Markov comparisons along a single chain).
    hold_steps :
        A plateau at the start — the initial temperature is held constant for
        this many steps before the cooling curve begins (0/1 = off). It is
        ADDED ON TOP: the run is steps+hold-1 steps long, the cooling curve
        keeps its full `steps` support points and therefore exactly the same
        end temperature as without a plateau. This gives the annealer a real
        exploration phase so it can leave the warm-start basin at all.
    kick :
        Optional sigma-calibrated offset kick, passed straight to the kernel
        (see KICK_DEFAULTS in Funcs_Qubo_Annealers). It needs "sigma", and
        WHICH sigma matters: on a penalty encoding the Walsh sigma is
        dominated by constraint-violating directions the chain never takes
        (420 on cdc7-4-3-2 against 16.7 measured at the actual operating
        state), so pass the state-local one there.
    offset_k_escape :
        Scales the E_Offset when offset_increase_rate="auto_gp":
        rate = q50(dE+) / k_escape, i.e. after k_escape rejected steps the
        offset has grown to a median uphill barrier. SMALLER = stronger
        offset = more aggressive escape from local minima (25 was the
        previous fixed value). Unrelated to gamma, which only caps T(1) via
        the landscape depth.

    Returns (same order as the reference):
        Min_varAssignements  list (S*num_MC) of 0/1 lists
        Mins                 list (S*num_MC)
        Trajectories         list (S*num_MC) of E traces (steps+1)
        Infos                dict: labels, group_of_trial, T (group 0),
                             Ts (per group), ExecTimes, Offsets,
                             offset_rates, cooling_c (per group),
                             best (index), X_best, E_best, X_final
    """
    if type_alg != "digitalAnnealing":
        raise ValueError("qubo_min_solver: only 'digitalAnnealing' — use "
                         "the reference library for other algorithms")
    if cooling_param is None:
        cooling_param = ["logarithmic", 10]
    b = np.asarray(b, dtype=np.float64)
    n = len(b)

    # ── Build the start groups ──────────────────────────────────────────────
    if initial_varAssignements_pre is not None:
        starts = np.asarray(initial_varAssignements_pre, dtype=np.int8)
        if starts.ndim == 1:
            starts = starts[None, :]
        X0_groups = [np.repeat(s[None, :], num_MC, axis=0) for s in starts]
    elif random_start:
        X0_groups = [random_start_states(n, num_MC,
                                         seed_gen_initial_varAssignment)]
    else:
        s = random_start_states(n, 1, seed_gen_initial_varAssignment)[0]
        X0_groups = [np.repeat(s[None, :], num_MC, axis=0)]
    S = len(X0_groups)

    print("Evaluation %s (batched), Optimization variables = %d, "
          "start groups = %d, num_MC = %d" % (type_alg, n, S, num_MC))

    # ── Calibration: per start group, at ITS own start vector ───────────────
    #   Different starting points see landscapes of different steepness — the
    #   best warm-start solution sits deeper in the funnel and has the largest
    #   dE scale. A shared c (from group 0) is roughly 2x too hot for the
    #   flatter starting points and melts them.
    #   cooling_per_group=False restores the old shared schedule.
    x_rand_cache = None
    _sigma_cache = [None]        # sigma is global to the PBF, computed once
    time_init1 = time_init2 = 0.0

    def _calibrate(x_ref, quiet=False):
        """(T, offset_rate, c_cache) for the start vector x_ref."""
        nonlocal x_rand_cache, time_init1, time_init2
        t0 = time.time()
        dE0 = eval_delta_energy(A, b, x_ref)
        time_init1 += time.time() - t0
        t0 = time.time()
        E0 = float(eval_qubo(A, b, c, x_ref))
        time_init2 += time.time() - t0

        cp = list(cooling_param)               # copy: c/d cache is per group
        if str(cp[0]).startswith("da_gp"):     # da_gp and da_gp_floor
            cp[2] = dE0                        # dE scale of THIS start point
            cp[3] = x_ref.tolist()             # plus the imbalance correction
            cp[4] = E0                         # E_warm of THIS group
            if len(cp) > 5 and cp[5] is None:
                if x_rand_cache is None:       # landscape depth: global
                    x_rand_cache = float(eval_qubo(
                        A, b, c,
                        random_start_states(n, 1,
                                            seed_gen_initial_varAssignment)[0]))
                cp[5] = x_rand_cache
        elif str(cp[0]) == "sigma":
            # sigma = std(dE) is a property of the PBF, not of a start point,
            # so it is computed once for the whole call and shared by every
            # start group — unlike the da_gp calibration above, which is
            # deliberately per-group.
            nonlocal_sigma = _sigma_cache[0]
            if nonlocal_sigma is None:
                nonlocal_sigma = qa3_delta_e_sigma(A, b)
                _sigma_cache[0] = nonlocal_sigma
            while len(cp) < 5:
                cp.append(None)
            if cp[4] is None:
                cp[4] = nonlocal_sigma
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            T_g = generate_cooling_schedule(cp, steps, hold_steps=hold_steps)
        if not quiet:
            sys.stdout.write(buf.getvalue())
        rate = offset_increase_rate
        if rate == "auto_gp":
            with contextlib.redirect_stdout(io.StringIO()) if quiet \
                    else contextlib.nullcontext():
                rate = auto_offset_rate(
                dE0, cp[3] if str(cp[0]).startswith("da_gp") else None,
                k_escape=offset_k_escape)
        return T_g, float(rate), (cp[9] if len(cp) > 9 else 0.0)

    # ── Monte-Carlo batches, one per start group ────────────────────────────
    track_offsets = bool(visual_inst or save_addinfo)
    seed_seq = seed_rand if isinstance(seed_rand, (int, np.integer)) \
        else list(np.asarray(seed_rand, dtype=np.int64).ravel())
    master = BulkRandomizer(seed_seq, mem_budget_mb)

    T_shared = None if cooling_per_group else _calibrate(X0_groups[0][0])

    Trajectories, Mins, Min_varAssignements, Final_varAssignements = [], [], [], []
    Offsets, ExecTimes, labels, group_of_trial = [], [], [], []
    Ts, offset_rates, cooling_cs = [], [], []
    for g, X0 in enumerate(X0_groups):
        if T_shared is not None:
            T, rate_g, c_g = T_shared
        else:
            T, rate_g, c_g = _calibrate(X0[0])
        Ts.append(T)
        offset_rates.append(rate_g)
        cooling_cs.append(c_g)
        start_time_round = time.time()
        out = digital_annealing_batch(A, b, c, X0, T, rate_g,
                                      master.spawn(),
                                      save_addinfo=save_addinfo,
                                      track_offsets=track_offsets,
                                      recompute_every=recompute_every,
                                      kick=kick)
        endtime = time.time() - start_time_round
        ExecTimes.append(endtime)
        for i in range(num_MC):
            Trajectories.append(out["Trajectories"][i])
            Mins.append(float(out["E_min"][i]))
            if save_addinfo:
                Min_varAssignements.append(out["X_min"][i].tolist())
            Final_varAssignements.append(out["X_final"][i].tolist())
            if track_offsets:
                Offsets.append(out["Offsets"][i])
            labels.append("S%d_MC%d" % (g, i))
            group_of_trial.append(g)
        print("%d-th start group (%d trials): %s sec, Average time rest "
              "execution %s sec, Min %s"
              % (g, num_MC, endtime,
                 np.mean(ExecTimes) * (S - g - 1), np.min(out["E_min"])))

    best = int(np.argmin(Mins))
    Infos = {"labels": labels, "group_of_trial": group_of_trial,
             "T": Ts[0], "Ts": Ts,               # T = group 0 (compatibility)
             "ExecTimes": ExecTimes, "Offsets": Offsets if track_offsets else None,
             "offset_increase_rate": offset_rates[0], "offset_rates": offset_rates,
             "cooling_c": cooling_cs, "cooling_per_group": cooling_per_group,
             "sigma": _sigma_cache[0],
             "T_start": float(Ts[0][0]), "T_end": float(Ts[0][-1]),
             "best": best, "E_best": Mins[best],
             "X_best": Min_varAssignements[best] if save_addinfo else None,
             "X_final": Final_varAssignements}

    # ── CSV persistence (the reference's Runs/ layout) ──────────────────────
    ID_run = None
    if save_csv:
        eval_directory = os.path.join(os.getcwd(), "Runs")
        day_dir = os.path.join(eval_directory,
                               "Evaluation_" + str(datetime.date.today()))
        traj_dir = os.path.join(day_dir, "Trajectories")
        addinfo_dir = os.path.join(day_dir, "AddInfo")
        for d in (eval_directory, day_dir, traj_dir, addinfo_dir):
            os.makedirs(d, exist_ok=True)
        filename = "Evaluation_" + str(datetime.date.today()) + ".csv"
        summary_path = os.path.join(day_dir, filename)
        if not os.path.exists(summary_path):
            columns = ["type_alg", "ID", "time", "timegen1", "timegen2",
                       "bestMin", "seed_gen", "variables", "start_groups",
                       "num_MC", "type_cooling", "offset_rate",
                       "T_start", "T_end", "sigma"]
            pd.DataFrame(list(), columns=columns).to_csv(summary_path)
        ID_run = str(len(os.listdir(traj_dir)) + 1)

        df = pd.DataFrame({labels[i]: np.asarray(Trajectories[i])
                           for i in range(len(Trajectories))})
        df.to_csv(os.path.join(traj_dir, "Evaluation_"
                               + str(datetime.date.today()) + "_" + ID_run
                               + "_trajectories.csv"))
        df = pd.DataFrame()
        df["label"] = labels
        df["group"] = group_of_trial
        df["Min"] = Mins
        df["Exec_group"] = [ExecTimes[g] for g in group_of_trial]
        if save_addinfo:
            df["Min_var"] = ["".join(map(str, v)) for v in Min_varAssignements]
        df.to_csv(os.path.join(addinfo_dir, "Evaluation_"
                               + str(datetime.date.today()) + "_" + ID_run
                               + "_addinfo.csv"))
        with open(summary_path, "a") as csvfile:
            csv.writer(csvfile).writerow(
                [type_alg, ID_run, np.sum(ExecTimes), time_init1, time_init2,
                 np.min(Mins), seed_gen_initial_varAssignment, n, S, num_MC,
                 cooling_param[0],
                 # with per-group calibration, report the range not one value
                 offset_rates[0] if len(set(offset_rates)) == 1
                 else "%.4g..%.4g" % (min(offset_rates), max(offset_rates)),
                 "%.4g..%.4g" % (min(t[0] for t in Ts), max(t[0] for t in Ts)),
                 "%.4g..%.4g" % (min(t[-1] for t in Ts), max(t[-1] for t in Ts)),
                 "" if _sigma_cache[0] is None else "%.6g" % _sigma_cache[0]])

    if visual_inst:
        fig = VisualizeRuns(n, type_alg, labels, Trajectories, Mins,
                            ExecTimes, Ts, Offsets if track_offsets else None,
                            groups=group_of_trial)
        fig.show()
        if save_csv:
            fig.write_html(os.path.join(day_dir, "Evaluation_" + str(type_alg)
                                        + "_" + str(datetime.date.today())
                                        + "_" + ID_run + ".html"))

    return Min_varAssignements, Mins, Trajectories, Infos
