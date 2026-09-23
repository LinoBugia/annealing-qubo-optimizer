# annealing-qubo-optimizer

**Optimize thousands of binary optimization variables on local hardware.**
Hundreds of thousands with sparse storage — and millions on stronger hardware.

A batched Digital Annealing engine **for QUBOs only** — the fast,
industrialized version of
[annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator). Pure numpy, no
GPU, no solver licence.

Measured on a 2024 laptop (Apple M4, 24 GB), `num_MC=8`:

| scale | storage for the problem | step time | dense would need |
|---|---|---|---|
| **10³ – 10⁴ variables** | dense, 34 MB – 537 MB | 0.26 – 0.77 ms | — |
| **3.3 × 10⁴ variables** | dense, 8.6 GB | 4.3 ms | — |
| **2.6 × 10⁵ variables** | sparse, 202 MB | 24.9 ms | 550 GB |
| **10⁶ variables** | sparse, **0.81 GB** | not timed | 9 TB |

The million-variable row is a memory measurement only — the problem builds in
3.4 s and occupies 1.25 GB of process RSS, but the step time was not measured
on this machine. Memory is what scales linearly here, which is why more RAM
carries this straight into the millions. Full numbers in
[Performance](#performance).

This repository uses Digital Annealing. There are **no bounds yet on solution
quality** — but it is a highly effective optimization engine: from uniformly
random starting points it reaches **99.1–99.7% of the best-known cut** on the
n=800 Gset Max-Cut instances in about three minutes each, with no
problem-specific tuning. See [Benchmark instances](#benchmark-instances).

MIT licensed, see [LICENSE](LICENSE).

## Contents

- [What changed against the reference](#what-changed-against-the-reference) — matrix storage, batching, the float-drift guard
- [Canonical form](#canonical-form) · [Modules](#modules-code) · [Requirements](#requirements) · [Quickstart](#quickstart)
- [Performance](#performance) — step time, scaling, memory limits, where the time goes
- [Benchmark instances](#benchmark-instances) — Gset (Max-Cut) and MIPLIB (set packing) results
- [Tuning](#tuning-the-two-settings-that-decide-everything) — **read this first**: temperature and step budget
- [Parameter reference](#parameter-reference) · [Graph-partitioning specifics](#graph-partitioning-specifics-gp) · [Visualiser](#visualiser-visualizeruns)
- [Verification](#verification) · [Roadmap](#roadmap) · [License](#license)

## What changed against the reference

[annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator) holds the generic,
dict-based PBF implementation: arbitrary degree, Markov-chain view, SA / SCA /
TSP variants. It remains the reference in both senses — the source of the
semantics, and the thing this code is checked against. For PBFs of degree > 2
you still want it.

### This repository is standalone

No module under `Funcs_Qubo_*` imports
anything from the reference; the kernel and `qubo_min_solver` run with nothing
but the packages in [Requirements](#requirements). The reference is a
*verification* dependency, needed only by `Test_Compare_Reference.py` and by
the optional S8 block of `Bench_Performance.py` — both detect its absence and
skip rather than fail.

Restricting the scope to degree 2 is what buys everything below.

### Storage: monomial dict → matrix

The reference keeps a PBF as `{(i,j): coefficient}` plus a `pbf_var_dict` mapping each variable to the
monomials containing it. Here the QUBO is a matrix, dense or CSR — and row `k`
of `A` *is* the monomial list of `x_k`, so `pbf_var_dict` disappears entirely.
Measured on random QUBOs of density 0.3:

| n | monomials | dict + var_dict | CSR | dense |
|---|---|---|---|---|
| 500 | 38 122 | 9.5 MB | 0.9 MB | 2.0 MB |
| 1 000 | 150 715 | 33.2 MB | 3.6 MB | 8.0 MB |
| 2 000 | 602 404 | 108.5 MB | 14.4 MB | 32.0 MB |

Roughly **8–11× less memory than the dict**, and at n=2000 the dict costs more
than three times the *dense* matrix — Python object overhead dominates at
~180 bytes per monomial. This is what lifts the practical ceiling from a few
thousand variables to
[262 144](#large-n-bounded-degree-via-csr).

### ΔE from a gradient, not monomial iteration

With `G = X·A` the whole
ΔE vector is `(1−2X)·(b+2G)`, and an accepted flip of bit `k` updates it with
`G += s_k · A[k,:]`. The reference walks the monomial lists for the same
result. This is also what makes the acceptance scan vectorisable.

### All Monte-Carlo trials in one batch

The reference runs `num_MC` trials in
a Python loop; here they are one `(mc, n)` array, so extra trials cost far
less than linear — see
[the batching measurements](#monte-carlo-trials-batching-and-where-it-stops-paying).
One flip per step *per trial* is preserved; the DA semantics are unchanged.

### Bulk RNG

Random numbers are drawn in large blocks under a memory budget
rather than one at a time, and the Metropolis test uses
`standard_exponential` (accept iff `ΔE − offset < T·ε`) instead of
`uniform + log`. RNG is still the single largest cost in a step, at
[37–44%](#where-the-time-goes).

### A guard against float drift

Updating `G` and `E` incrementally
accumulates float64 rounding error over long runs, and the runs here are long.
Every `recompute_every` steps both are reconstructed exactly from `X`. At the
default this costs about 6 µs per step — see
[the measurement](#where-the-time-goes) — and it is the reason million-step
runs stay trustworthy.

### One reference bug, deliberately not reproduced

Min-tracking is seeded
with the initial state. The reference returns an empty minimum assignment when
no improvement ever occurs, which with a warm start is the normal case rather
than an edge case.

Net effect on the same problem (n=2000, density 0.3, one trial each):
**≈41× per trial**, rising to **≈76×** once eight trials are batched — full
table in [Against the reference library](#against-the-reference-library).

## Canonical form

```math
E(x) \;=\; x^{\top} A x \;+\; b^{\top} x \;+\; c ,
\qquad x \in \{0,1\}^{n}
```

with $A \in \mathbb{R}^{n \times n}$ symmetric and $\operatorname{diag}(A) = 0$
(dense float or scipy CSR), $b \in \mathbb{R}^{n}$ linear and $c$ constant.
The energy change from flipping bit $k$ is

```math
\Delta E_k \;=\; (1 - 2x_k)\,\bigl(b_k + 2\,(Ax)_k\bigr)
```

Row $k$ of $A$ *is* the monomial list of $x_k$, which is why no `pbf_var_dict`
is needed. **One flip per step and per trial** — that is the Digital Annealing
semantics and it is preserved exactly. What gets vectorised around it is the
acceptance scan over all $n$ flips and the Monte-Carlo trials as an `(mc,n)`
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
| `Bench_Performance` | Runtime and limit benchmark (see [Performance](#performance)) |

Runnable scripts in the same folder:

| Script | Purpose |
|---|---|
| `Skript_Solve_Random_QUBO.py` | Demo of the multi-start mode: several start vectors, `num_MC` trials each, with the visualiser and CSV output |
| `Skript_Solve_Gset.py` | Max-Cut on Gset — `num_MC` and `T` derived from measurements, not hand-set |
| `Skript_Solve_MIPLIB.py` | MIPLIB set packing, with a `--probe` temperature grid |
| `Bench_Performance.py` | The nine benchmark blocks behind [Performance](#performance) |
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
visualiser.

## Performance

All numbers measured on **Apple M4** (4 performance + 6 efficiency cores),
24 GB unified memory, macOS 26.5, Python 3.12, numpy 1.26.4 (OpenBLAS, capped
at 3 threads), scipy 1.16. Reproduce with:

```bash
python Bench_Performance.py --out bench.csv
```

### How it is measured

Step time is taken as a two-point slope,
`(t(2s) − t(s)) / s`, so the one-off O(n²) setup (`G = X@A`, dE init, the
64 MB RNG block) cancels instead of being smeared into the per-step cost.
Each point is measured three times and the **minimum** is kept — on a
big.LITTLE CPU the process migrates between core clusters, which otherwise
adds up to 1.7× spread upward. The measurement window is at least
`recompute_every` steps long so the periodic exact reconstruction is
amortised into the number rather than skipped. Problem construction is
outside the clock.

**How far to trust the absolute numbers.** Repeating a whole block reproduces
it to a few percent — the n-sweep re-run gave 466.6 µs against 474.6 µs at
n=4096. But the *same* configuration measured in different blocks, at
different times, spreads much further: n=4096 dense at `mc=8` came out
anywhere between 367 µs and 778 µs across the nine blocks, because the
process lands on performance or efficiency cores depending on what else the
machine is doing. **Compare numbers within one table, not across tables**, and
carry only the scaling to other hardware — not the constants.

Two metrics are used throughout:

- **µs/step** — one step is one acceptance scan over all `n` flips for all
  `mc` trials, plus at most one accepted flip per trial.
- **ns/flip** — the same divided by `n · mc`, i.e. the cost of a single
  evaluated flip proposal. This is the size-independent efficiency number.

### Step time vs. problem size (dense, `num_MC=8`)

| n | memory for A | µs/step | ns/flip | 1000 steps × 8 trials |
|---|---|---|---|---|
| 128 | 131 KB | 30 | 29.1 | 0.03 s |
| 256 | 524 KB | 58 | 28.4 | 0.06 s |
| 512 | 2.1 MB | 100 | 24.3 | 0.10 s |
| 1 024 | 8.4 MB | 171 | 20.9 | 0.17 s |
| 2 048 | 33.6 MB | 259 | 15.8 | 0.26 s |
| 4 096 | 134 MB | 475 | 14.5 | 0.47 s |
| 8 192 | 537 MB | 768 | 11.7 | 0.77 s |
| 16 384 | 2.15 GB | 1 721 | 13.1 | 1.7 s |
| 32 768 | 8.59 GB | 4 320 | 16.5 | 4.3 s |

**The step is O(mc·n), not O(n²).** Time grows linearly with n across the
whole range; the quadratic term only appears in the memory for `A` and in the
periodic reconstruction. Below n ≈ 2000 the cost per flip is dominated by
numpy dispatch overhead (~15 array operations per step, each with fixed
per-call cost), which is why ns/flip *falls* from 29 to 12 as n grows — larger
problems use the hardware better, not worse.

### Large n: bounded degree via CSR

Constant density is the wrong parametrisation at scale — a large real QUBO has
a bounded **degree** (neighbours per variable), not a constant fraction of n²
nonzeros. With `nnz = n · degree`, memory grows *linearly* and the step is
O(mc·n) regardless:

| n (degree 64) | memory for A | dense would need | µs/step | ns/flip | 1000 steps × 8 trials |
|---|---|---|---|---|---|
| 16 384 | 12.6 MB | 2.1 GB | 1 291 | 9.9 | 1.3 s |
| 65 536 | 50.6 MB | 34.4 GB | 6 083 | 11.6 | 6.1 s |
| 262 144 | 202 MB | 550 GB | 24 883 | 11.9 | 24.9 s |
| 1 048 576 | 810 MB | 9 TB | *not timed* | — | — |

**262 144 variables in 202 MB of problem data**, with ns/flip flat at ~10–12
across a 16× span in n. This is the path to large problems, not the dense one.

The last row is a memory measurement only: a million-variable instance at
degree 64 holds 67.1 M nonzeros, builds in 3.4 s and occupies 810 MB as CSR
(1.25 GB process RSS). Its step time was **not** measured here. Extrapolating
the flat ns/flip would put it near 100 ms/step, but that is a projection, not
a measurement. What the row does establish is that **memory is not what stops
you at a million variables** — the same problem stored densely would need 9 TB.

### Monte-Carlo trials: batching, and where it stops paying

All `num_MC` trials of a group run as one `(mc,n)` batch. Extra trials are
much cheaper than linear — until the working set falls out of cache:

| `num_MC` (n=4096) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| µs/step | 112 | 284 | 405 | 645 | 997 | 1 565 | 3 098 | 10 236 |
| ns/flip | 27.2 | 34.7 | 24.7 | 19.7 | 15.2 | **11.9** | **11.8** | 19.5 |
| factor vs. `mc=1` | 1.0 | 2.5 | 3.6 | 5.8 | 8.9 | 14.0 | 27.8 | 91.7 |

64 trials cost 27.8× one trial, not 64×. But at `mc=128` it inverts: the step
holds roughly eight `(mc,n)` arrays (dE, G, eps, the boolean mask, the int64
cumsum, …), and at `mc·n ≈ 5·10⁵` elements that working set exceeds L2 and the
step becomes DRAM-bandwidth bound.

**Rule of thumb: keep `num_MC · n` between 10⁵ and ~2.5·10⁵.** That is the
efficiency plateau at ~12 ns/flip. Below it you pay dispatch overhead, above
it you pay memory bandwidth.

### Sparse storage buys memory, not speed

| n=16384, `mc=8` | dense | CSR d=0.001 | CSR d=0.01 | CSR d=0.05 |
|---|---|---|---|---|
| memory for A | 2.15 GB | 3.3 MB | 32 MB | 159 MB |
| µs/step | 2 435 | 1 284 | 1 499 | 1 407 |
| relative to dense | 1.00 | 0.53 | 0.62 | 0.58 |
| setup (first step) | 353 ms | 34 ms | 39 ms | 61 ms |

CSR is a flat ~1.8× faster *independent of density* — 50× more nonzeros does not change the step time,
because the dense O(mc·n) vector work dominates, not the matrix row access.
What it does buy is up to 650× less memory and a 10× cheaper setup.

### float32 is a memory lever only — and it costs time

| n | float64 | float32 | memory saved | time cost |
|---|---|---|---|---|
| 2 048 | 269 µs | 351 µs | 50% | +30% |
| 4 096 | 541 µs | 592 µs | 50% | +9% |
| 8 192 | 825 µs | 989 µs | 50% | +20% |

The kernel computes in float64 (`Xf`, `b`, `G`), so a float32 `A` is upcast
per element on every access. Halve the memory only if memory is the binding
constraint.

### Problem type does not matter

At n=2048, `mc=8`: random signed 367 µs, number partitioning 384 µs, Gram
clustering 346 µs — a spread of 11% across three unrelated problem
structures. The step cost depends on shape and storage, not on where the
coefficients came from.

These three were timed back to back in one block, so compare them to *each
other*, not to the 259 µs in the
[step-time table](#step-time-vs-problem-size-dense-num_mc8) above: that entry
comes from a different run, and across runs the same configuration spreads
considerably — see
[How it is measured](#how-it-is-measured).

### Where the time goes

Per-operation breakdown of one step, `mc=8`, measuring the true amortised RNG
*generation* cost (not just the slice handed out by the bulk block):

| share of step | n=1024 | n=4096 | n=8192 |
|---|---|---|---|
| RNG generation | 37% | 39% | **44%** |
| flip selection (`cumsum`+`argmax`) | 16% | 19% | 17% |
| dE reconstruction | 22% | 18% | 15% |
| G update | 15% | 15% | 10% |
| acceptance compare | 9% | 7% | 10% |
| count (`sum`) | 3% | 2% | 3% |

RNG is the single largest component and its share grows with n — but at
**37–44%**, not the ~90% an earlier estimate in `Funcs_Qubo_Randomizers`
assumed. The difference matters for planning: by Amdahl's law, making RNG
entirely free would still cap the gain at ~1.8× on this hardware. The second-largest item, flip selection, is pure algorithmic
overhead — a full `cumsum` over the mask to pick the r-th allowed flip — and
is a cheaper target than the RNG.

`recompute_every` (periodic exact reconstruction of G and E, guarding against
float drift) at n=4096, measured at a fixed 4096 steps so every setting
amortises over the same window:

| `recompute_every` | 128 | 512 | 1024 (default) | 4096 | off |
|---|---|---|---|---|---|
| µs/step | 537 | 551 | 457 | 397 | 367 |

The reconstruction matvec itself costs 6.1 ms, i.e. 6 µs/step amortised at the
default — the rest of the ~90 µs gap is the periodic reallocation disturbing
the working set. Below 512 the cost is clearly visible; at 4096 it is
negligible.

### End to end

Full `qubo_min_solver` call including setup, dE init, cooling calibration and
min tracking — 1000 steps, 8 trials:

| n | total wall time |
|---|---|
| 512 | 0.12 s |
| 2 048 | 0.35 s |
| 8 192 | 0.92 s |

### Against the reference library

Same problem (density 0.3), one trial each, so the comparison is per trial and
not helped by batching. The reference is dict-based and runs out of road
around n ≈ 2000:

| n | monomials | ref SA µs/proposal | ref DA µs/step | ref DA ns/proposal | this lib µs/step | this lib ns/proposal | speedup (DA) |
|---|---|---|---|---|---|---|---|
| 100 | 1 556 | 4.9 | 351 | 3 509 | 18.3 | 182.8 | 19× |
| 200 | 5 991 | 7.3 | 517 | 2 583 | 23.0 | 115.1 | 22× |
| 500 | 37 411 | 20.4 | 1 120 | 2 241 | 36.9 | 73.8 | 30× |
| 1 000 | 150 125 | 69.5 | 1 703 | 1 703 | 41.7 | 41.7 | 41× |
| 2 000 | 601 314 | 139.0 | 2 458 | 1 229 | 60.0 | 30.0 | 41× |

With 8 trials batched instead of one, the same n=2000 case is ~76× per trial.

The cost structure per flip proposal at n=2000 is more informative than the
speedup:

- **reference SA: 139 000 ns.** SA proposes one random flip per step and
  recomputes `Eval_Delta_Energy` from scratch over ~600 monomials each time,
  so its per-proposal cost grows linearly with n.
- **reference DA: 1 229 ns.** DA keeps the ΔE vector cached and only updates
  after an accepted flip — roughly 113× cheaper per proposal, purely from the
  algorithm.
- **this library: 30 ns.** Same DA semantics, but the scan over all n flips is
  vectorised — another ~41×.

### Limits, and what stronger hardware would buy

| what binds | where it binds | how to move it |
|---|---|---|
| **Memory for `A`** (dense, O(n²)) | n≈32 768 needs 8.6 GB; n≈65 536 would need 34 GB | CSR with bounded degree — n=262 144 fits in 202 MB |
| **Cache** for the `(mc,n)` working set | `mc·n ≳ 5·10⁵` | keep `num_MC · n` ≤ ~2.5·10⁵ |
| **Single-threaded numpy** | everything except the reconstruction matvec | a backend with real parallel elementwise ops |
| **RNG throughput** | 37–44% of the step | GPU, or a cheaper Metropolis draw |
| **numpy dispatch overhead** | n ≲ 2000 (29 → 12 ns/flip) | batch more trials, or multi-arm batching |

Concretely, on this laptop-class CPU:

- **n up to ~30 000 dense** and **n in the hundreds of thousands sparse** are
  reachable *today*, at ~12 ns per evaluated flip.
- **More RAM is the only lever for the dense path** and it is a weak one:
  memory goes as n², so 4× the RAM buys 2× the n. Sparse storage buys three
  orders of magnitude instead, which is why the CSR path is the answer to
  "how big can this get".
- **More CPU cores buy very little.** Only the periodic reconstruction goes
  through BLAS; the O(mc·n) elementwise work that dominates every step is
  single-threaded numpy, and OpenBLAS is capped at 3 threads here anyway.
- **A GPU is the real lever**, and the `xp` backend hook in
  `Funcs_Qubo_Annealing3` already exists for it. The step is a handful of
  elementwise passes over `(mc,n)` plus `mc·n` random numbers — a shape GPUs
  are built for, with both far higher memory bandwidth and far higher RNG
  throughput. The cache cliff at `mc·n ≈ 5·10⁵` also moves out, so much wider
  Monte-Carlo batches become worthwhile. Expect the win to come from
  *simultaneously* raising `num_MC` and n, not from the same run going faster.

Caveat on all projections: these are single-machine, single-process numbers on
a big.LITTLE laptop CPU with a 3-thread BLAS. Treat the scaling laws
(linear in `mc·n`, quadratic memory for dense) as the transferable result and
the absolute constants as specific to this machine.

## Benchmark instances

Two importers turn public benchmark libraries into the canonical `(A, b, c)`
form. Both follow the same rule: **convert losslessly or refuse.** A converter
that silently approximates would make every number on this page meaningless.

| Library | Module | Script | Converts losslessly |
|---|---|---|---|
| Gset (Max-Cut) | `Funcs_Qubo_MaxCut` | `Skript_Solve_Gset.py` | everything — Max-Cut *is* a QUBO |
| MIPLIB 2017 (MPS) | `Funcs_Qubo_MpsImport` | `Skript_Solve_MIPLIB.py` | pure set packing only; anything else is rejected |

### Gset — Max-Cut

Max-Cut needs **no penalty term**: it is natively unconstrained, so there is a
single energy scale rather than two annealing against each other. With `W` the
weighted adjacency matrix, `A = W` and `b = −W·1`, and the cut is exactly `−E`.

```bash
python Skript_Solve_Gset.py --steps 100000            # all nine below
python Skript_Solve_Gset.py --instances G1 --steps 300000
```

Random starting points, no warm start, no problem-specific moves. `num_MC` is
derived from the [efficiency plateau](#monte-carlo-trials-batching-and-where-it-stops-paying)
and `T` from the [scan correction](#temperature-correct-for-the-scan) — neither
is hand-tuned per instance.

| Instance | n | degree | `num_MC` | cut found | best known | % | time |
|---|---|---|---|---|---|---|---|
| G1 | 800 | 47.9 | 188 | 11 591 | 11 624 | **99.72** | 186 s |
| G11 | 800 | 4.0 | 188 | 562 | 564 | **99.65** | 190 s |
| G14 | 800 | 11.7 | 188 | 3 038 | 3 064 | **99.15** | 192 s |
| G22 | 2 000 | 20.0 | 75 | 13 154 | 13 359 | 98.47 | 180 s |
| G32 | 2 000 | 4.0 | 75 | 1 388 | 1 410 | 98.44 | 170 s |
| G55 | 5 000 | 5.0 | 30 | 9 983 | 10 299 | 96.93 | 163 s |
| G60 | 7 000 | 4.9 | 21 | 13 739 | 14 188 | 96.84 | 155 s |
| G70 | 10 000 | 2.0 | 15 | 9 173 | 9 591 | 95.64 | 147 s |
| G81 | 20 000 | 4.0 | 8 | 13 402 | 14 060 | 95.32 | 155 s |

All nine ran at a flat 100 000 steps, which is 125 sweeps for G1 but only 5
for G81 — see [Step budget](#step-budget-it-must-scale-with-n). Re-running the
four large instances at 60 sweeps each (`steps = 60·n`) separates a
measurement artefact from a real deficit:

| Instance | sweeps before → after | % before | % after |
|---|---|---|---|
| G55 | 20 → 60 | 96.93 | 96.93 |
| G60 | 14 → 60 | 96.84 | 96.84 |
| G70 | 10 → 60 | 95.64 | 95.73 |
| G81 | 5 → 60 | 95.32 | **97.61** |

**G81 was genuinely starved** and gains 2.3 points once fed. **G55 and G60 do
not move at all** despite 3–4× the steps, so for the mid-size sparse instances
the deficit is *not* the step budget — and this README has no confirmed
explanation for it. The untested candidate is the number of independent
chains, which falls from 188 to 8 as n grows because `num_MC·n` is held on the
[efficiency plateau](#monte-carlo-trials-batching-and-where-it-stops-paying);
chain count and instance size are confounded in this table and were never
varied independently.

Graph *degree*, which one might expect to matter through the ΔE spectrum, does
not separate the results at all: G1 (degree 48) and G11 (degree 4) both land at
~99.7%. That hypothesis was tested and failed.

Best-known values are compiled from
[this benchmark repo](https://github.com/0816keisuke/max-cut-problem-benchmark)
(tracing back to Stanford's Gset page and the Toshiba SBM benchmark), G81 from
[arXiv:2505.18508](https://arxiv.org/abs/2505.18508) where it is reported as
proven optimal. **These values drift** — a new G63 record was published in
October 2025 — so treat the table in `Skript_Solve_Gset.py` as a snapshot, not
a constant. Sources disagree on G55 (10 299 vs 10 116); the higher, less
flattering value is used.

### MIPLIB — set packing

A MIPLIB instance is a MIP, not a QUBO. Only **pure set packing** converts
without loss:

```math
\max \sum_i c_i x_i \quad\text{s.t.}\quad \sum_{i \in S_r} x_i \le 1
\;\Longrightarrow\;
E(x) = -\sum_i c_i x_i + P \sum_r \sum_{i<j \in S_r} x_i x_j
```

With `M` the 0/1 constraint matrix, `(MᵀM)_ij` counts the rows containing both
`i` and `j` — exactly the violated pairs — so `A = (P/2)·(MᵀM − diag)` in one
sparse matmul, with **no slack variables and no encoding**. `P = 2·max(c)`
makes any violation unprofitable, so every local minimum is feasible.

The importer **rejects** `≥` rows (set covering — needs slack variables), `=`
rows, RHS ≠ 1, coefficients ≠ 1, and non-binary variables, each with a message
naming the reason. Set covering and general linear constraints are deliberately
out of scope rather than silently approximated.

```bash
python Skript_Solve_MIPLIB.py --instance cdc7-4-3-2 --probe
```

On `cdc7-4-3-2` (11 811 binary vars, 14 478 set-packing rows → 1.24 M conflict
pairs, 29.8 MB as CSR where dense would be 1.1 GB):

| | selected | of best known |
|---|---|---|
| greedy baseline | 232 | 75.6% |
| after 4 000 steps (5 s) | 263 | 85.7% |
| after 200 000 steps (14 min) | 287 | 93.5% |
| after a further 600 000 steps | 289 | **94.1%** |
| best known (open instance) | 307 | |

Feasibility is verified against the original MPS structure, not against the
QUBO. **This is the hard case**, and the reason is structural rather than
numerical: every maximal feasible packing is a strict local optimum, because
switching a variable off always costs exactly `+1` and switching one on costs
`−1 + 2·(conflicts) ≥ +1`. No single flip ever improves a feasible solution;
progress requires a compound (1,2)-swap that one-flip-per-step DA can only
stumble into by chance. 32 diverse restarts and 1.2 M further steps all
returned exactly 289.

### What the two cases show

The difference between 99.7% and 94.1% is the **ΔE spectrum**, not the size
or the difficulty. Max-Cut has real-valued, spread-out ΔE, so temperature can
actually sort moves and the useful range is a broad plateau (on G1 anything
from 0.4 to 1.2 works). Unweighted set packing collapses ΔE onto `{−1, +1, +3,
…}` with a uniform `+1` barrier, leaving temperature nothing to sort — the
workable window shrank to 0.15–0.18, and outside it the run either froze at the
greedy value or turned into a random walk.

A rough suitability checklist, in the order the evidence supports it:

1. **Natively unconstrained** — penalties create two competing energy scales.
2. **Weighted, spread-out coefficients** — a degenerate ΔE spectrum starves
   the Metropolis criterion.
3. **Solution density near 50%** — at 2.6% (as in set packing) single flips
   are almost always meaningless.
4. **Budget `steps ∝ n`** — otherwise large instances are merely starved.

## Tuning: the two settings that decide everything

Everything else is secondary. Get these two wrong and no schedule saves you.

### Temperature: correct for the scan

A DA step evaluates **all `n` flips** and accepts at most one. So the
acceptance probability *per flip* must be `p/n`, not `p` — which makes the
right temperature far lower than intuition suggests:

```math
T \;\approx\; \frac{\Delta E_{q50}}{\ln(n / p)}
```

with `ΔE_q50` the median **uphill** move at your starting point and `p` the
target chance that some uphill flip is accepted in a step (0.3 is a good
default). Measured against a temperature sweep on three unrelated problem
classes:

| Instance | n | median uphill ΔE | formula says | measured optimum |
|---|---|---|---|---|
| G1 (Max-Cut) | 800 | 5.0 | 0.63 | 0.8 |
| G22 (Max-Cut) | 2 000 | 3.0 | 0.34 | 0.6 |
| cdc7-4-3-2 (set packing) | 11 811 | 1.0 (min), 15 (q50) | 0.095 | 0.18 |

The formula lands within a factor 1.3–1.8, consistently **low**. That
direction is the safe one: too cold degrades gently, too hot collapses. On G1,
`T=4` (6× the optimum) costs 5% of the cut; `T=0.2` (¼ of it) costs 0.4%.
**When unsure, go colder.**

This correction is not specific to any problem family — it follows from the
scan itself. The `da_gp` family in
[Graph-partitioning specifics](#graph-partitioning-specifics-gp) applies the
same formula, but wraps it in a calibration tuned for one problem type; the
formula above is the part that generalises.

### Step budget: it must scale with n

One accepted flip per step means the number of passes over the variables is
`steps / n`. A fixed step count therefore starves large instances:

| | G1 (n=800) | G81 (n=20 000) |
|---|---|---|
| 100 000 steps | 125 sweeps | **5 sweeps** |
| quality reached | 99.7% | 95.3% |

Those two numbers do **not** show that the annealer degrades with size — they
show the large instance was given a hundredth of the work per variable. Budget
`steps ≈ k · n` and pick `k` (sweeps) as the quantity you actually tune; `k` in
the low hundreds is a reasonable starting point. The `steps` default of `1000`
is meant for `n` in the hundreds and is far too small above that.

Note this is a property of the **one-flip-per-step DA semantics**, not of the
implementation: the per-step cost is O(mc·n), so `steps ∝ n` makes total work
O(n²) even on a sparse problem.

## Parameter reference

### `qubo_min_solver(A, b, c, ...)`

Entries marked **(GP)** exist for the warm-started graph/Gram bisection
workflow — see [Graph-partitioning specifics](#graph-partitioning-specifics-gp).
They are not general-purpose defaults.

| Parameter | Default | Meaning |
|---|---|---|
| `A, b, c` | — | QUBO in canonical form. `A` symmetric with **zero diagonal**, dense float32/64 or scipy CSR; `b` is `(n,)`; `c` shifts the level only, never the argmin. |
| `type_alg` | `"digitalAnnealing"` | Only value. SA/SCA/TSP live in the reference library. |
| `steps` | `1000` | Annealing steps **per trial**. One step = one acceptance scan over all `n` flips and **at most one** accepted flip — so budget `steps ≈ k·n`, see [Step budget](#step-budget-it-must-scale-with-n). The default suits `n` in the hundreds only. |
| `num_MC` | `8` | Monte-Carlo trials **per start group**. All trials of a group run as one `(num_MC, n)` batch — see the [batching numbers](#monte-carlo-trials-batching-and-where-it-stops-paying). |
| `cooling_param` | `None` → `["logarithmic", 10]` | Schedule contract, see [below](#cooling_param-by-schedule-type). |
| `seed_rand` | `42` | Master seed of the RNG blocks. **The only persistence of randomness** — same seed and parameters give a bit-identical run. Random numbers are never reused. |
| `seed_gen_initial_varAssignment` | `42` | Seed for generated random start vectors, and for the reference random state used by `da_gp`. |
| `initial_varAssignements_pre` | `None` | `None` → random start(s) per `random_start`. `(n,)` → **one** start group. `(S,n)` → **S start groups**, `num_MC` trials each. |
| `random_start` | `False` | Only without supplied start vectors: `True` = every trial starts at its own random point. |
| `offset_increase_rate` | `0.0` | E_Offset increment on steps where **no** flip was accepted. `0.0` = off. `"auto_gp"` **(GP)** = derived from the measured ΔE scale. |
| `offset_k_escape` | `25.0` | **(GP)** Scales the E_Offset under `"auto_gp"`: `rate = q50(ΔE⁺) / k_escape`. Smaller = stronger offset = more aggressive escape. Unrelated to `gamma`. |
| `cooling_per_group` | `True` | Each start group gets its **own** schedule, calibrated at its own start vector. Needed whenever start vectors sit at different depths — a schedule calibrated on a deep warm start is too hot for shallower ones. `False` = one shared schedule from group 0. |
| `hold_steps` | `0` | **Plateau**: hold the start temperature for this many steps before the curve falls. Added **on top** — the run is `steps + hold − 1` long and the cooling curve keeps its full `steps` support points, so the end temperature is unchanged. `0` and `1` both mean off. |
| `save_addinfo` | `True` | Carry the minimum state vectors `X_min` (otherwise energies only). |
| `save_csv` | `False` | Write `Runs/Evaluation_<date>/`: summary, trajectory and add-info CSV plus the plot as HTML. **Writes one directory per run.** |
| `visual_inst` | `False` | Open the Plotly visualiser. |
| `recompute_every` | `1024` | Every k steps, recompute gradient `G` and energy `E` **exactly** from `X` instead of updating incrementally — guards against float drift on long runs. Costs ~6 µs/step at the default; see [the measurements](#where-the-time-goes). |
| `mem_budget_mb` | `64.0` | Memory budget of the bulk RNG blocks. Larger = fewer refills, more RAM. Does **not** affect the result — only the block size, not the number sequence. |

**Returns** `(Min_varAssignements, Mins, Trajectories, Infos)` — same order as
the reference. All lists have length `S × num_MC`, sorted by group.
`Infos` carries: `labels`, `group_of_trial`, `T` (group 0's schedule, for
compatibility), `Ts` (schedule **per group**), `ExecTimes`, `Offsets`,
`offset_increase_rate` and `offset_rates`, `cooling_c` (calibrated `c` per
group), `cooling_per_group`, `best`, `E_best`, `X_best`.

### `cooling_param` by schedule type

`t` is the step count **after** the plateau (`t = max(1, step − hold_steps + 1)`),
`S` = `steps`.

| `cooling_param` | Curve |
|---|---|
| `["constant", T]` | `T` |
| `["linear", T_start, T_end]` | linear from `T_start` to `T_end` |
| `["rising", a]` | `a·t` (heating, for diagnostics) |
| `["logarithmic", c]` | `c / ln(1 + t^2.22)` |
| `["logarithmic_step", c, k]` | as `logarithmic`, each temperature held `k` steps |
| `["exponential", c]` | `exp(t/c) − 1` |
| `["geometric", T0, α]` | `T0·α^(t−1)` |
| `["hyperbolic", T0]` | `T0 / t` |
| `["da_gp", …]` **(GP)** | calibrated: `c / ln(1 + t^d)` |
| `["da_gp_floor", …]` **(GP)** | calibrated: `T_freeze + (T_hot − T_freeze)·ln2 / ln(1 + t^d)` |

For a new problem, start with the generic schedules above — `logarithmic` or
`geometric` — and tune `c` against the ΔE scale at your starting point.

## Graph-partitioning specifics (GP)

> **Scope.** Everything in this section is calibrated for **one** problem
> family: warm-started binary bisection of a Gram matrix, as used by the
> `gp-qubo-rag-indexer`. The workflow is `create_graph_binary_clustering_qubo`
> → `lloyd_bisect` warm start → `da_gp` cooling → `auto_gp` offset. The
> calibration assumes a Lloyd solution exists (`E_warm`, called `E_Lloyd` in
> the code) and that the landscape depth `D` is meaningful. It is a
> schedule *generator for that problem*, not a general-purpose default —
> applying it elsewhere is untested and the `gamma` cap in particular has no
> meaning without a warm start to protect.

### The `da_gp` contract (slots)

```
[0] "da_gp" | "da_gp_floor"
[1] n_steps        number of steps (enters the end-temperature condition)
[2] delta_E_vec    ΔE vector at the start — leave None, the optimizer fills it
                   at the REAL start vector of each group
[3] start_assign   start vector (imbalance correction only)
[4] E_warm         energy at the warm start ("E_Lloyd")
[5] E_start        energy of a random state — leave None, the optimizer fills it
                   once globally (landscape depth D)
[6] p_hot          target acceptance of a median uphill flip at t=1
[7] p_cold         residual acceptance of a small (q25) flip at the end
[8] gamma          cap: T(1) <= gamma · landscape depth D
[9] [10]           cache: calibrated c and d (written by the calibrator)
[11] [12]          cache: T_hot_capped and T_freeze (for da_gp_floor)
```

### Calibration

```
q25, q50   quantiles of the UPHILL flips (ΔE > 0) at the start
T_hot      = q50 / ln(n / p_hot)          scan correction — see Tuning above,
T_freeze   = q25 / ln(n / p_cold)         this part is NOT gp-specific
d          = 2.22  (fixed)
c          = min( T_hot·ln2 ,  T_freeze·ln(1+S^d) ,  gamma·D·ln2 )
D          = E_start − E_warm             landscape depth
```

- **`p_hot`** (0.3) sets how mobile the start is, **`p_cold`** (0.01) how hard
  the end freezes — larger is hotter in both cases.
- **`gamma`** (0.05) caps `T(1)` at 5% of the landscape depth so the warm
  start cannot melt. Affects temperature **only**, never the E_Offset.
- **Prefer `da_gp_floor`**: when the `T_hot` term binds (usually), plain
  `da_gp` ends *below* the freezing bound — measured 0.134 against 1.175, 8.8×
  too cold. `da_gp_floor` runs toward `T_freeze` and never undercuts it, with
  `d` controlling only the curvature.

### The three exploration controls

| Control | When it acts | Effect |
|---|---|---|
| `hold_steps` | start | extends the hot phase before cooling begins |
| `offset_k_escape` **(GP)** | when a trial is stuck | raises the energy level until a flip gets through — acts **only** on fully rejected steps |
| `p_hot` / `gamma` **(GP)** | whole curve | how hot the curve starts at all |

## Visualiser `VisualizeRuns(...)`

All trials in ONE figure, **colour = start group** — one legend click hides
the whole group (trajectory, ΔE band, minima and offsets together).

- **Energy** every chain individually; **Temp** one curve per group.
- **Delta Energy** aggregated per group (median + 25/75 band) rather than
  `num_MC` noise curves, drawn above that group's T curve so the two are
  directly comparable — it is the quantity the cooling was calibrated against.
- **Mins** a marker swarm per group at its own x position, group minimum as a star.
- **E_Offset** genuinely tracked, not reconstructed from `ΔE == 0`.

### Minima that are equal are drawn as equal

When several trials reach the *same* solution their energies still differ in
the last float64 digits, because each accumulates its incremental updates in a
different order. Left alone, the Mins panel auto-scales onto that 1e-15 spread
and a single solution looks like a dozen competing ones.

`VisualizeRuns(..., min_display_rtol=1e-9)` therefore snaps minima that agree
within a relative tolerance onto one value, and when *every* trial agrees it
pins the y-axis to a readable window and labels it rather than zooming into
noise.

**This is a display convention, not a change to the results.** `Mins` and the
returned solutions keep their exact values, and the hover text shows the raw
number to 12 digits. `min_display_rtol=0.0` turns it off.

The default is deliberate. The drift is bounded by `recompute_every`, which
rebuilds `E` exactly every k steps, so at most k incremental updates can
accumulate: measured here, eight trials on one solution spread by ~2e-10 at
|E| ~ 1.8e5, a relative 1e-15. A 1e-9 tolerance clears that by six orders of
magnitude and still keeps genuinely distinct optima apart — whereas 1e-6 would
mean an absolute tolerance of 1.0 on a problem with |E| ~ 1e6, quietly merging
real solutions.

## Verification

```bash
cd Code && python Test_Compare_Reference.py
```

Part A: `E(x)`, the ΔE vector, the incremental ΔE update and the converter
round-trips, all to machine precision against the reference. Part B: minimum
statistics and runtime, old vs. new on the same problem.

This is the one script that needs the reference library present. It is
resolved as a **sibling directory**, so clone it next to this repository:

```bash
git clone https://github.com/LinoBugia/annealing-cop-approximator
```

Without it the script reports that and exits; nothing else in the project is
affected.

## Roadmap

- Multi-arm batching (block-diagonal QUBO per tree level, one flip per block,
  min/offset per block) — also fixes the numpy dispatch overhead at small n
- Cheaper flip selection: the `cumsum`+`argmax` over the mask is 16–19% of the
  step and is pure bookkeeping
- GPU: CuPy as the `xp` backend — RNG is 37–44% of the step and the remaining
  work is elementwise over `(mc,n)`
- Docker image

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Lino Bugia.
