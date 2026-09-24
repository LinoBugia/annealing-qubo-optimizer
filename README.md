# annealing-qubo-optimizer

**Optimize thousands of binary optimization variables on local hardware.**
Hundreds of thousands with sparse storage — and millions on stronger hardware.

A batched Digital Annealing engine **for QUBOs only** — the fast,
industrialized version of
[annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator). Pure numpy, no
GPU, no solver licence.

Public benchmark instances, solved on a 2024 laptop (Apple M4, 24 GB) with
nothing but numpy and scipy:

| problem | variables | monomials | QUBO in RAM | runtime | reached |
|---|---|---|---|---|---|
| **G1** — Max-Cut | 800 | 19 176 | 463 KB | 3.1 min | 11 591 of 11 624 · **99.72%** |
| **G22** — Max-Cut | 2 000 | 19 990 | 488 KB | 3.0 min | 13 154 of 13 359 · **98.47%** |
| **G70** — Max-Cut | 10 000 | 9 999 | 280 KB | 2.5 min | 9 173 of 9 591 · **95.64%** |
| **G81** — Max-Cut | 20 000 | 40 000 | 1.04 MB | 2.6 min | 13 402 of 14 060 · **95.32%** |
| **cdc7-4-3-2** — set packing | 11 811 | 1 240 000 | 29.8 MB | 14 min | 287 of 307 · **93.5%** |

- **variables** — `n`, the number of binary unknowns the annealer flips.
- **monomials** — quadratic terms `x_i·x_j` with a nonzero coefficient. This,
  not `n`, is the real size of the problem: it is what a dict-based
  implementation has to hold as individual Python objects, and it is why
  G70 (10 000 variables, 9 999 terms) is a *smaller* problem than G1.
- **QUBO in RAM** — the matrix `A` as scipy CSR. Stored densely, G81 would
  need 3.20 GB instead of 1.04 MB.
- **runtime** — elapsed time on the clock from launching the script to the
  printed result: reading the instance, building the QUBO, calibrating the
  cooling schedule, every annealing step, and the final evaluation. Not CPU
  time — one process, one core's worth of numpy, no GPU.
- **reached** — against the best value published for that instance.
  `cdc7-4-3-2` is an **open** MIPLIB 2017 instance: no optimum is proven, and
  feasibility here is verified against the original MPS structure rather than
  against the QUBO.

The Max-Cut runs all used a flat 100 000 steps — 125 passes over the variables
for G1, but only 5 for G81. Given a budget proportional to `n`, G81 reaches
**97.61%**. All nine Gset instances, the temperature and trial counts (derived
from measurements, never hand-tuned per instance), and why set packing is the
hard case: [Benchmark instances](docs/benchmarks.md).

There are **no bounds yet on solution quality** — this is a heuristic and
claims nothing about optimality. What it offers is throughput: ~12 ns per
evaluated bit flip, a step that costs O(`num_MC`·n) rather than O(n²), and a
measured ceiling of **262 144 variables in 202 MB**, where dense storage would
need 550 GB. See [Performance](docs/performance.md).

MIT licensed, see [LICENSE](LICENSE).

## Contents

| Document | Contents |
|---|---|
| [Tuning](docs/tuning.md) | **Read this first** — temperature and step budget, the two settings that decide whether a run works at all |
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

## Modules (`Code/`)

| Module | Contents |
|---|---|
| `Funcs_Qubo_ProblemGeneration` | Generators (random QUBO, number partitioning, Gram clustering), Lloyd warm start, converters pbf ↔ (A,b,c) — tuple **and** packed int keys |
| `Funcs_Qubo_Optimizers` | `qubo_min_solver` (mirror of `pbf_min_solver`), Plotly `VisualizeRuns`, CSV persistence in the `Runs/` layout |
| `Funcs_Qubo_Annealers` | Batched DA kernel: `(mc,n)` scan, E_Offset mechanics, min tracking seeded with the initial state, periodic exact G/E reconstruction |
| `Funcs_Qubo_TempSchedules` | Cooling schedules, generic ones plus the calibrated `da_gp`/`da_gp_floor` family, plateau phase (`hold_steps`) |
| `Funcs_Qubo_Annealing3` | Class-free base: `eval_qubo`, `eval_delta_energy`, int-key pack/unpack, backend hook (`xp` → CuPy) |
| `Funcs_Qubo_Randomizers` | Bulk RNG blocks (budget-limited), `standard_exponential` Metropolis; the seed is the only persistence |
| `Funcs_Qubo_MaxCut` | Gset/Max-Cut import — natively unconstrained, no penalty needed |
| `Funcs_Qubo_MpsImport` | MPS/MIPLIB import — pure set packing only, rejects anything lossy |

Runnable scripts in the same folder:

| Script | Purpose |
|---|---|
| `Skript_Solve_Random_QUBO.py` | Demo of the multi-start mode: several start vectors, `num_MC` trials each, with the visualiser and CSV output |
| `Skript_Solve_Gset.py` | Max-Cut on Gset — `num_MC` and `T` derived from measurements, not hand-set |
| `Skript_Solve_MIPLIB.py` | MIPLIB set packing, with a `--probe` temperature grid |
| `Bench_Performance.py` | The nine benchmark blocks behind [Performance](docs/performance.md) |
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

Or reproduce a published benchmark directly:

```bash
cd Code
python Skript_Solve_Gset.py --instances G1,G11,G14 --steps 100000
python Skript_Solve_MIPLIB.py --instance cdc7-4-3-2 --probe
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Lino Bugia.
