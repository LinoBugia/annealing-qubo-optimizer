# Performance

Runtime and memory measurements of the batched kernel, how they were taken,
where the hard limits are, and what stronger hardware would buy.

All numbers measured on **Apple M4** (4 performance + 6 efficiency cores),
24 GB unified memory, macOS 26.5, Python 3.12, numpy 1.26.4 (OpenBLAS, capped
at 3 threads), scipy 1.16. Reproduce with:

```bash
python Bench_Performance.py --out bench.csv
```

## How it is measured

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

## Step time vs. problem size (dense, `num_MC=8`)

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

## Large n: bounded degree via CSR

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

## Monte-Carlo trials: batching, and where it stops paying

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

## Sparse storage buys memory, not speed

| n=16384, `mc=8` | dense | CSR d=0.001 | CSR d=0.01 | CSR d=0.05 |
|---|---|---|---|---|
| memory for A | 2.15 GB | 3.3 MB | 32 MB | 159 MB |
| µs/step | 2 435 | 1 284 | 1 499 | 1 407 |
| relative to dense | 1.00 | 0.53 | 0.62 | 0.58 |
| setup (first step) | 353 ms | 34 ms | 39 ms | 61 ms |

CSR is a flat ~1.8× faster *independent of density* — 50× more nonzeros does not change the step time,
because the dense O(mc·n) vector work dominates, not the matrix row access.
What it does buy is up to 650× less memory and a 10× cheaper setup.

## float32 is a memory lever only — and it costs time

| n | float64 | float32 | memory saved | time cost |
|---|---|---|---|---|
| 2 048 | 269 µs | 351 µs | 50% | +30% |
| 4 096 | 541 µs | 592 µs | 50% | +9% |
| 8 192 | 825 µs | 989 µs | 50% | +20% |

The kernel computes in float64 (`Xf`, `b`, `G`), so a float32 `A` is upcast
per element on every access. Halve the memory only if memory is the binding
constraint.

## Problem type does not matter

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

## Where the time goes

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

## End to end

Full `qubo_min_solver` call including setup, dE init, cooling calibration and
min tracking — 1000 steps, 8 trials:

| n | total wall time |
|---|---|
| 512 | 0.12 s |
| 2 048 | 0.35 s |
| 8 192 | 0.92 s |

## Against the reference library

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

## Limits, and what stronger hardware would buy

| what binds | where it binds | how to move it |
|---|---|---|
| **Memory for `A`** (dense, O(n²)) | n≈32 768 needs 8.6 GB; n≈65 536 would need 34 GB | CSR with bounded degree — n=262 144 fits in 202 MB |
| **Cache** for the `(mc,n)` working set | `mc·n ≳ 5·10⁵` | stay on the [efficiency plateau](#monte-carlo-trials-batching-and-where-it-stops-paying) |
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
