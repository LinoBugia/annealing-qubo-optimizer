# Codebase

What lives where. Every module under `Code/` runs on numpy alone except
where noted; see [Requirements](../README.md#requirements).

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
| `Bench_Performance.py` | The nine benchmark blocks behind [Performance](performance.md) |
| `Bench_Diagnostics.py` | One instrumented run: admissible-set size, blocked share, displacement against flips, how much of the problem the chain ever touches |
| `Test_Sigma_Schedule.py` | σ against brute force over all `n·2ⁿ` directed edges (n ≤ 10), plus the schedule shapes |
| `Test_Compare_Reference.py` | Verification against the reference library (needs it present) |
