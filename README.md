# annealing-qubo-optimizer

**Optimize thousands of binary optimization variables on local hardware.**
Hundreds of thousands with sparse storage — and millions on stronger hardware.

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
| [Changes against the reference](docs/reference-changes.md) | Each change in full, what it bought, how it is verified, roadmap |
| [Graph-partitioning specifics](docs/graph-partitioning.md) | The `da_gp` schedule family, calibrated for warm-started Gram bisection |
| [Codebase](docs/codebase.md) | What each module and script does |
| [Visualiser](docs/visualiser.md) | What `VisualizeRuns` draws and its display conventions |

Below: [What changed against the reference](#what-changed-against-the-reference) ·
[Canonical form](#canonical-form) · [Requirements](#requirements) ·
[Quickstart](#quickstart)

## What changed against the reference

[annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator) is the generic,
dict-based PBF implementation — arbitrary degree, SA / SCA / TSP variants —
and remains both the source of the semantics and the thing this code is
checked against. For degree > 2 you still want it.

**This repository is standalone**: nothing under `Funcs_Qubo_*` imports it.
Restricting the scope to degree 2 replaces the monomial dictionary with a
matrix, turns ΔE into a gradient update, and lets all Monte-Carlo trials run
as one `(mc, n)` batch — together **≈41× per trial and ≈76× once eight trials
are batched**, with the DA semantics unchanged. Each change, its measurement
and the verification script: [Changes against the reference](docs/reference-changes.md).

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

## Temperature and the E_Offset

Which temperature rule applies depends on one cheap measurement — the spread
of the uphill ΔE spectrum. Dense spectra (Max-Cut) scale by `σ = std(ΔE)` and
want a geometric sweep; gapped ones (penalty encodings) need the scan
correction held constant, because there σ is dominated by directions the chain
never takes. The E_Offset is worth exactly as much as blocked steps are
frequent — measurable before you spend a run.

All of it, with the measurements behind it: **[Tuning](docs/tuning.md)**.

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
