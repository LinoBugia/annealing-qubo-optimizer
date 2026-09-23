# annealing-qubo-optimizer

Digital Annealing reduces a combinatorial problem to a sequence of single bit
flips. In Python that usually dies on the data structure: a dict of monomials,
one Monte-Carlo trial after another, and a few thousand variables is the
ceiling. This library keeps the semantics and swaps the structure — the QUBO is
a matrix, all trials run as one batch, and the ceiling moves to hundreds of
thousands of variables on a laptop.

It is a heuristic: it returns the best state found across `S × num_MC`
annealing chains and claims nothing about optimality. What it offers is
throughput, full instrumentation of the run, and lossless importers for two
public benchmark libraries so the quality claims can be checked.

```console
$ python Skript_Solve_Gset.py --instances G1 --steps 100000

Gset / Max-Cut — 100000 steps, T from the scan correction, mc on the plateau

inst       n   edges   deg    mc  T_scan      cut bestknown     gap      %       s
G1       800   19176  47.9   188   0.634    11591     11624      33  99.72     186
```

`G1` is a standard Max-Cut benchmark; 11 624 is the best value anyone has
published for it. Neither `mc` nor `T_scan` was set by hand — both are derived
from measurements, starting from a uniformly random point with no
problem-specific tuning. The run is deterministic for a given seed, so every
column but the last reproduces exactly.

## Quickstart

```bash
pip install -r requirements.txt
cd Code
```

```bash
# Solve a random QUBO: 3 start vectors × 5 trials each, with plot and CSV
python Skript_Solve_Random_QUBO.py
```

```bash
# Reproduce the Max-Cut benchmark (downloads the instances on first run)
python Skript_Solve_Gset.py --instances G1,G11,G14 --steps 100000
```

Only `numpy` is needed for the annealing kernel. `scipy` adds sparse (CSR)
problems and both importers; `pandas` and `plotly` are pulled in by
`qubo_min_solver` for CSV output and the visualiser, so a solver call needs
all four.

## Commands

```python
from Funcs_Qubo_ProblemGeneration import create_random_qubo_signed, random_start_states
from Funcs_Qubo_Optimizers import qubo_min_solver

A, b, c = create_random_qubo_signed(500, density=0.5, seed=42)
starts = random_start_states(500, 3, seed=42)      # 3 start vectors

Min_varAss, Mins, Trajectories, Infos = qubo_min_solver(
    A, b, c, steps=3000, num_MC=5,                # num_MC trials per start vector
    cooling_param=["logarithmic", 50, 0],
    initial_varAssignements_pre=starts,
    save_csv=True, visual_inst=True)
```

Every parameter is listed in [docs/configuration.md](docs/configuration.md).
Two of them decide whether a run works at all — see
[docs/tuning.md](docs/tuning.md).

## Core concept

The whole library works on one canonical form:

```math
E(x) = x^{\top} A x + b^{\top} x + c , \qquad x \in \lbrace 0,1 \rbrace^{n}
```

with `A` symmetric and zero on the diagonal (dense float or scipy CSR), `b`
linear, `c` constant. Row `k` of `A` *is* the monomial list of `x_k`, which is
why no separate variable index is needed. Flipping bit `k` changes the energy by

```math
\Delta E_k = (1 - 2x_k)\bigl(b_k + 2(Ax)_k\bigr)
```

**One flip per step and per trial** — that is the Digital Annealing semantics
and it is preserved exactly. What is vectorised around it is the acceptance
scan over all `n` flips and the Monte-Carlo trials as an `(mc,n)` batch. The
consequences of that choice, measured, are in
[docs/performance.md](docs/performance.md).

## Output

`qubo_min_solver` returns four values, each of length `start groups × num_MC`:

```
Min_varAssignements   best 0/1 state per trial
Mins                  best energy per trial
Trajectories          energy over time, steps+1 values per trial
Infos                 dict: schedules per group, offsets, exec times,
                      group_of_trial, best index, X_best, E_best
```

With `save_csv=True` each run also writes `Code/Runs/Evaluation_<date>/` —
summary, trajectory and add-info CSV plus the plot as HTML. Nothing outside
this repository reads that layout; it is a record, not an interchange format.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/tuning.md](docs/tuning.md) | Temperature and step budget — the two settings that decide everything |
| [docs/configuration.md](docs/configuration.md) | Every `qubo_min_solver` parameter, every cooling schedule type |
| [docs/performance.md](docs/performance.md) | Step time, scaling, memory limits, where the time goes, hardware outlook |
| [docs/benchmarks.md](docs/benchmarks.md) | Gset (Max-Cut) and MIPLIB (set packing) results, and the importers |
| [docs/graph-partitioning.md](docs/graph-partitioning.md) | The `da_gp` schedule family, calibrated for warm-started Gram bisection |
| [docs/visualiser.md](docs/visualiser.md) | What `VisualizeRuns` draws and its display conventions |
| [docs/design-notes.md](docs/design-notes.md) | Why this differs from the reference implementation, and the roadmap |

## Project structure

| Path | Purpose |
|------|---------|
| `Code/Funcs_Qubo_Annealers.py` | `digital_annealing_batch` — the batched DA kernel; everything else is around it |
| `Code/Funcs_Qubo_Optimizers.py` | `qubo_min_solver`, the Plotly visualiser, CSV persistence |
| `Code/Funcs_Qubo_Annealing3.py` | Canonical form helpers, packed int keys, `xp` backend hook |
| `Code/Funcs_Qubo_TempSchedules.py` | Cooling schedules, generic and calibrated |
| `Code/Funcs_Qubo_Randomizers.py` | Bulk RNG blocks, `standard_exponential` Metropolis |
| `Code/Funcs_Qubo_ProblemGeneration.py` | Problem generators, Lloyd warm start, pbf ↔ (A,b,c) |
| `Code/Funcs_Qubo_MaxCut.py` | Gset / Max-Cut import |
| `Code/Funcs_Qubo_MpsImport.py` | MPS / MIPLIB import, pure set packing only |
| `Code/Skript_Solve_*.py` | Runnable entry points: random QUBO, Gset, MIPLIB |
| `Code/Bench_Performance.py` | The nine benchmark blocks behind `docs/performance.md` |
| `Code/Test_Compare_Reference.py` | Verification against [annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator), cloned as a sibling directory |

## License

MIT — see [LICENSE](LICENSE).
