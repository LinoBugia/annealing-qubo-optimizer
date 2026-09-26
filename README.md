# annealing-qubo-optimizer

A batched Digital Annealing engine for QUBOs. **Pure numpy and scipy** — no
GPU, no solver licence, no compiled extensions, no cloud.

On a 2024 laptop (Apple M4, 24 GB):

| problem | variables | monomials | QUBO in RAM | steps | runtime | reached |
|---|---|---|---|---|---|---|
| **G1** — Max-Cut | 800 | 19 176 | 463 KB | 30 000 | 2.0 min | 11 624 of 11 624 · **100.00%** |
| **G22** — Max-Cut | 2 000 | 19 990 | 488 KB | 30 000 | 1.8 min | 13 317 of 13 359 · **99.69%** |
| **G70** — Max-Cut | 10 000 | 9 999 | 280 KB | 300 000 | 7.4 min | 9 373 of 9 591 · **97.73%** |
| **G81** — Max-Cut | 20 000 | 40 000 | 1.04 MB | 300 000 | 7.7 min | 13 658 of 14 060 · **97.14%** |
| **cdc7-4-3-2** — set packing | 11 811 | 1 240 000 | 29.8 MB | 1 000 000 | 31 min | 289 of 307 · **94.14%** |

Nine Gset instances average **98.74 %** of the best published value, and G1
reaches it exactly. Each cell is one run at one seed, reproducible with the
commands under [Quickstart](#quickstart).

## What makes it fast

| | |
|---|---|
| **~12 ns** | per evaluated bit flip |
| **O(`num_MC`·n)** | per step, not O(n²) — the ΔE vector is maintained incrementally from a gradient |
| **41–76×** | against the dict-based [reference implementation](https://github.com/LinoBugia/annealing-cop-approximator), per trial and once eight trials are batched |
| **262 144 variables in 202 MB** | measured ceiling for a bounded-degree problem; dense storage would need 550 GB |

The restriction to degree 2 is what buys all of it: row `k` of the matrix `A`
*is* the monomial list of `x_k`, so no dictionary of terms exists at any
point. Beyond that, the ceiling is RAM rather than the algorithm. Details in
[Performance](docs/performance.md).

## Reading the table

- **monomials**, not `n`, is the real size: it is what a dict-based solver
  holds as individual Python objects, and it is why G70 (10 000 variables,
  9 999 terms) is a *smaller* problem than G1 with 800.
- **runtime** is wall clock for the whole script — reading the instance,
  building the QUBO, calibrating the schedule, every step, final evaluation.
  One process, one core's worth of numpy.
- **reached** is against the best published value. `cdc7-4-3-2` is an **open**
  MIPLIB 2017 instance with no proven optimum, and its feasibility is checked
  against the original MPS rows, not against the QUBO.

The step counts differ deliberately. One accepted flip per step means passes
over the variables are `steps / n`, so a flat budget starves large instances:
G81 gains 1.8 points from budget alone. And the variable count can mislead in
the other direction — G70 splits into 1 598 independent components with 1 354
variables that touch no edge, so its effective size is 8 646. The loader
reports this. See [Benchmark instances](docs/benchmarks.md).

There are **no bounds on solution quality** here. This is a heuristic and
claims nothing about optimality — what it offers is throughput and honest
measurement of where it stands.

## Contents

| Document | Contents |
|---|---|
| [Tuning](docs/tuning.md) | **Read this first** — the two temperature regimes, how to tell them apart, the step budget, the E_Offset |
| [Performance](docs/performance.md) | Step time, scaling, memory limits, where the time goes, what stronger hardware would buy |
| [Benchmark instances](docs/benchmarks.md) | Gset (Max-Cut) and MIPLIB (set packing) results, the lossless importers, and which problems suit this engine |
| [Parameter reference](docs/configuration.md) | Every `qubo_min_solver` parameter, every cooling schedule type, what the call returns |
| [Design notes](docs/design-notes.md) | Each change against the reference in full, verification, roadmap |
| [Graph-partitioning specifics](docs/graph-partitioning.md) | The `da_gp` schedule family, calibrated for warm-started Gram bisection |
| [Visualiser](docs/visualiser.md) | What `VisualizeRuns` draws and its display conventions |

Below: [What changed against the reference](#what-changed-against-the-reference) ·
[Canonical form](#canonical-form) · [Modules](#modules-code) ·
[Requirements](#requirements) · [Quickstart](#quickstart)

## What changed against the reference

[annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator) holds the generic,
dict-based PBF implementation: arbitrary degree, Markov-chain view, SA / SCA /
TSP variants. It remains the reference in both senses — the source of the
semantics, and the thing this code is checked against. For PBFs of degree > 2
you still want it.

**This repository is standalone.** No module under `Funcs_Qubo_*` imports
anything from the reference; the kernel and `qubo_min_solver` run with nothing
but the packages in [Requirements](#requirements). The reference is a
*verification* dependency, needed only by `Test_Compare_Reference.py` and by
the optional S8 block of `Bench_Performance.py` — both detect its absence and
skip rather than fail.

Restricting the scope to degree 2 is what buys everything below.

- **Storage: monomial dict → matrix.** Row `k` of `A` *is* the monomial list
  of `x_k`, so the reference's `pbf_var_dict` disappears entirely —
  **8–11× less memory than the dict**, and at n=2000 the dict costs more than
  three times the *dense* matrix. This is what lifts the practical ceiling
  from a few thousand variables to
  [262 144](docs/performance.md#large-n-bounded-degree-via-csr).
- **ΔE from a gradient, not monomial iteration.** With `G = X·A` the whole ΔE
  vector is `(1−2X)·(b+2G)`, and an accepted flip of bit `k` updates it with
  `G += s_k · A[k,:]`. This is also what makes the acceptance scan
  vectorisable at all.
- **All Monte-Carlo trials in one batch.** One `(mc, n)` array instead of a
  Python loop, so extra trials cost far less than linear — see
  [the batching measurements](docs/performance.md#monte-carlo-trials-batching-and-where-it-stops-paying).
  One flip per step *per trial* is preserved; the DA semantics are unchanged.
- **Bulk RNG.** Blocks under a memory budget instead of one draw at a time,
  and `standard_exponential` instead of `uniform + log`. RNG is still the
  single largest cost in a step, at
  [37–44%](docs/performance.md#where-the-time-goes).
- **A guard against float drift.** `G` and `E` are reconstructed exactly from
  `X` every `recompute_every` steps, at about 6 µs per step — the reason
  million-step runs stay trustworthy.
- **One reference bug, deliberately not reproduced.** Min-tracking is seeded
  with the initial state, so a warm start no longer returns an empty minimum
  assignment.

Net effect on the same problem (n=2000, density 0.3, one trial each):
**≈41× per trial**, rising to **≈76×** once eight trials are batched — full
table in [Against the reference library](docs/performance.md#against-the-reference-library),
each change in full in [Design notes](docs/design-notes.md).

## Canonical form

```math
E(x) = x^{\top} A x + b^{\top} x + c , \qquad x \in \lbrace 0,1 \rbrace^{n}
```

with `A` symmetric and zero on the diagonal (dense float or scipy CSR), `b`
linear and `c` constant. The energy change from flipping bit `k` is

```math
\Delta E_k = (1 - 2x_k)\bigl(b_k + 2(Ax)_k\bigr)
```

Row `k` of `A` *is* the monomial list of `x_k`, which is why no `pbf_var_dict`
is needed. **One flip per step and per trial** — that is the Digital Annealing
semantics and it is preserved exactly. What gets vectorised around it is the
acceptance scan over all `n` flips and the Monte-Carlo trials as an `(mc,n)`
batch.

## Temperature: two regimes

One cheap measurement decides which rules apply — the **spread** of the uphill
ΔE spectrum, `q50/min`. It comes out around 1–5 on Max-Cut, where the spectrum
is dense, and 15 or more on a penalty encoding, where the coefficients split
into "moves along the feasible set" and "moves that break a constraint" with
nothing in between.

**Dense spectrum → scale by σ.** With `σ = std(ΔE)`, computed in closed form
from the Walsh coefficients, temperatures expressed as multiples of σ transfer
between instances:

```python
cooling_param = ["sigma", 0.3, 0.11]      # geometric, 0.3σ down to 0.11σ
```

The measured best constant temperature is `0.116σ` on G1 and `0.134σ` on G22 —
two graphs whose degrees differ by a factor of twelve. Sweeping a grid of
windows, the result peaks at `c_end ≈ 0.11`: **cool down to the optimal
temperature and stop there rather than through it.** The hot end barely
matters. On G1 that reached the best known value at 30 000 steps, where a flat
100 000 on the older scan schedule fell 33 short.

**Gapped spectrum → σ stops working.** A penalty encoding breaks the
assumption that the chain lives where σ is measured: on `cdc7-4-3-2` the chain
never leaves the feasible set, where ΔE is ±1, while σ averages over uniformly
random states — all deeply infeasible — and reads 420 against a local scale of
16.7. The σ defaults would run it 117× too hot. There the scan correction
`T ≈ ΔE_min / ln(n/p)` applies, held constant: cooling windows into the
optimum all lost against simply sitting at it.

How far the all-flip scan is ahead of single-flip simulated annealing turns
out to depend only on `T/σ`; a spectral-gap analysis of small instances is
what locates where that advantage saturates. Out of scope here.

## The E_Offset

It raises the acceptance threshold on steps where **nothing** was admissible
and resets on every accepted flip, so it only builds up across consecutive
blocked steps — which makes its value predictable before you spend a run:

| | blocked steps | worth |
|---|---|---|
| Max-Cut (G1, G11, G14, G22) | 0.00–0.19 % | nothing; it never grew once in 10 000 steps |
| set packing (cdc7-4-3-2) | 11.7 % | **+3**, about a doubling of the budget |

Acceptance scales as `exp(offset/T)`, so the unit is `T`, not the barrier
height. Rates of 0.05 and 0.6 performed identically on cdc7 for opposite
reasons — the small one creeps up over several blocked steps, the large one
clears the step at once and resets, so it cannot compound.

Full rules and the phase-plan variant: [Tuning](docs/tuning.md).

## Modules (`Code/`)

| Module | Contents |
|---|---|
| `Funcs_Qubo_ProblemGeneration` | Generators (random QUBO, number partitioning, Gram clustering), Lloyd warm start, converters pbf ↔ (A,b,c) — tuple **and** packed int keys |
| `Funcs_Qubo_Optimizers` | `qubo_min_solver` (mirror of `pbf_min_solver`), Plotly `VisualizeRuns`, CSV persistence in the `Runs/` layout |
| `Funcs_Qubo_Annealers` | Batched DA kernel: `(mc,n)` scan, E_Offset mechanics, min tracking seeded with the initial state, periodic exact G/E reconstruction |
| `Funcs_Qubo_TempSchedules` | Cooling schedules: generic ones, the σ-scaled family (`["sigma", c_start, c_end, form]`), the calibrated `da_gp`/`da_gp_floor` pair, plateau phase (`hold_steps`) |
| `Funcs_Qubo_Annealing3` | Class-free base: `eval_qubo`, `eval_delta_energy`, `delta_e_sigma` (exact σ via Walsh coefficients), int-key pack/unpack, backend hook (`xp` → CuPy) |
| `Funcs_Qubo_Randomizers` | Bulk RNG blocks (budget-limited), `standard_exponential` Metropolis; the seed is the only persistence |
| `Funcs_Qubo_MaxCut` | Gset/Max-Cut import — natively unconstrained, no penalty needed |
| `Funcs_Qubo_MpsImport` | MPS/MIPLIB import — pure set packing only, rejects anything lossy |

Runnable scripts in the same folder:

| Script | Purpose |
|---|---|
| `Skript_Solve_Random_QUBO.py` | Demo of the multi-start mode: several start vectors, `num_MC` trials each, with the visualiser and CSV output |
| `Skript_Solve_Gset.py` | Max-Cut on Gset — `num_MC` and `T` from measurements. `--schedule sigma` plus a `--c-start`/`--c-end` grid for the window matrix, `--offset-sweep` for the E_Offset |
| `Skript_Solve_MIPLIB.py` | MIPLIB set packing — `--probe` temperature grid, `--cool T_hi:T_lo` windows, `--offset` |
| `Bench_Performance.py` | The nine benchmark blocks behind [Performance](docs/performance.md) |
| `Bench_Diagnostics.py` | One instrumented run: admissible-set size, blocked share, displacement against flips, how much of the problem the chain ever touches |
| `Test_Sigma_Schedule.py` | σ against brute force over all `n·2ⁿ` directed edges (n ≤ 10), plus the schedule shapes |
| `Test_Compare_Reference.py` | Verification against the reference library (needs it present) |

## Requirements

| Package | Needed for |
|---|---|
| `numpy` | everything — the annealing kernel needs nothing else |
| `scipy` | sparse (CSR) problems and both benchmark importers |
| `pandas`, `plotly` | `qubo_min_solver` only: CSV persistence and the visualiser |

`digital_annealing_batch` runs on numpy alone. The extra two enter through
`Funcs_Qubo_Optimizers`, which imports them at module level — so a
`qubo_min_solver` call needs all four even when you never write a CSV.

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from Funcs_Qubo_ProblemGeneration import create_random_qubo_signed, random_start_states
from Funcs_Qubo_Optimizers import qubo_min_solver

A, b, c = create_random_qubo_signed(500, density=0.5, seed=42)
starts = random_start_states(500, 3, seed=42)        # 3 start vectors

Min_varAss, Mins, Trajectories, Infos = qubo_min_solver(
    A, b, c, steps=3000, num_MC=5,
    cooling_param=["logarithmic", 50, 0],
    initial_varAssignements_pre=starts,              # num_MC trials per start vector
    hold_steps=600,                                  # hold T_start for 600 steps
    save_csv=True, visual_inst=True)
```

New relative to the reference: `initial_varAssignements_pre` accepts a **list
of start vectors** — each one runs `num_MC` trials as a batch, and each group
is calibrated at its own starting point via `cooling_per_group`.
`save_csv=True` writes `Runs/Evaluation_<date>/` (summary, trajectory and
add-info CSV plus the plot as HTML); `visual_inst=True` opens the Plotly
visualiser. Every parameter is in the
[Parameter reference](docs/configuration.md); the two that decide whether a run
works at all are in [Tuning](docs/tuning.md).

### Reproducing the table

Every row above, with the exact flags. The instances download on first use.

```bash
cd Code
# Gset — the sigma schedule, budget scaled to n
python Skript_Solve_Gset.py --instances G1,G11,G14,G22,G32 --steps 30000 \
       --schedule sigma --c-start 0.5 --c-end 0.05
python Skript_Solve_Gset.py --instances G55,G60 --steps 100000 \
       --schedule sigma --c-start 0.5 --c-end 0.05
python Skript_Solve_Gset.py --instances G70,G81 --steps 300000 \
       --schedule sigma --c-start 0.5 --c-end 0.05

# MIPLIB set packing — constant T, E_Offset on
python Skript_Solve_MIPLIB.py --instance cdc7-4-3-2 --steps 1000000 --offset 0.6
```

The `--c-start/--c-end` window is given explicitly because the built-in
defaults (`0.3 / 0.11`) were calibrated at 30 000 steps on a different grid
and land one cut lower on G1 — within noise, but the table should be
reproducible exactly rather than approximately.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Lino Bugia.
