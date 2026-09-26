# Changes against the reference

Each deliberate difference from the reference implementation, what it bought,
how the result is verified, and what is still open.

[annealing-cop-approximator](https://github.com/LinoBugia/annealing-cop-approximator) holds the generic,
dict-based PBF implementation: arbitrary degree, Markov-chain view, SA / SCA /
TSP variants. It remains the reference in both senses — the source of the
semantics, and the thing this code is checked against. For PBFs of degree > 2
you still want it.

## This repository is standalone

No module under `Funcs_Qubo_*` imports anything from the reference; the kernel
and `qubo_min_solver` run with nothing but the packages listed in the
[Requirements](../README.md#requirements). The reference is a *verification*
dependency, needed only by `Test_Compare_Reference.py` and by the optional S8
block of `Bench_Performance.py` — both detect its absence and skip rather than
fail.

Restricting the scope to degree 2 is what buys everything below.

## Storage: monomial dict → matrix

The reference keeps a PBF as `{(i,j): coefficient}` plus a `pbf_var_dict`
mapping each variable to the monomials containing it. Here the QUBO is a
matrix, dense or CSR — and row `k` of `A` *is* the monomial list of `x_k`, so
`pbf_var_dict` disappears entirely. Measured on random QUBOs of density 0.3:

| n | monomials | dict + var_dict | CSR | dense |
|---|---|---|---|---|
| 500 | 38 122 | 9.5 MB | 0.9 MB | 2.0 MB |
| 1 000 | 150 715 | 33.2 MB | 3.6 MB | 8.0 MB |
| 2 000 | 602 404 | 108.5 MB | 14.4 MB | 32.0 MB |

Roughly **8–11× less memory than the dict**, and at n=2000 the dict costs more
than three times the *dense* matrix — Python object overhead dominates at
~180 bytes per monomial. This is what lifts the practical ceiling from a few
thousand variables to
[262 144](performance.md#large-n-bounded-degree-via-csr).

## ΔE from a gradient, not monomial iteration

With `G = X·A` the whole ΔE vector is `(1−2X)·(b+2G)`, and an accepted flip of
bit `k` updates it with `G += s_k · A[k,:]`. The reference walks the monomial
lists for the same result. This is also what makes the acceptance scan
vectorisable.

## All Monte-Carlo trials in one batch

The reference runs `num_MC` trials in a Python loop; here they are one
`(mc, n)` array, so extra trials cost far less than linear — see
[the batching measurements](performance.md#monte-carlo-trials-batching-and-where-it-stops-paying).
One flip per step *per trial* is preserved; the DA semantics are unchanged.

## Bulk RNG

Random numbers are drawn in large blocks under a memory budget rather than one
at a time, and the Metropolis test uses `standard_exponential` (accept iff
`ΔE − offset < T·ε`) instead of `uniform + log`. RNG is still the single
largest cost in a step, at [37–44%](performance.md#where-the-time-goes).

## A guard against float drift

Updating `G` and `E` incrementally accumulates float64 rounding error over long
runs, and the runs here are long. Every `recompute_every` steps both are
reconstructed exactly from `X`. At the default this costs about 6 µs per step
— see [the measurement](performance.md#where-the-time-goes) — and it is the
reason million-step runs stay trustworthy.

## One reference bug, deliberately not reproduced

Min-tracking is seeded with the initial state. The reference returns an empty
minimum assignment when no improvement ever occurs, which with a warm start is
the normal case rather than an edge case.

Net effect on the same problem (n=2000, density 0.3, one trial each):
**≈41× per trial**, rising to **≈76×** once eight trials are batched — full
table in [Against the reference library](performance.md#against-the-reference-library).

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

The RNG streams naturally differ from the reference, so the comparison is made
over `E`/`ΔE` to machine precision and over trajectory statistics — not over
identical paths.

## Roadmap

- Multi-arm batching (block-diagonal QUBO per tree level, one flip per block,
  min/offset per block) — also fixes the numpy dispatch overhead at small n
- Cheaper flip selection: the `cumsum`+`argmax` over the mask is 16–19% of the
  step and is pure bookkeeping
- A compound **(1,2)-swap** as an atomic move. On set packing every maximal
  feasible packing is a strict local optimum under single flips, which is the
  measured ceiling on `cdc7-4-3-2` — see
  [MIPLIB — set packing](benchmarks.md#miplib--set-packing)
- GPU: CuPy as the `xp` backend — RNG is 37–44% of the step and the remaining
  work is elementwise over `(mc,n)`
- Docker image
