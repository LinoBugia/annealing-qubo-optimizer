# Benchmark instances

Results on two public benchmark libraries, and the importers that convert them
losslessly into the canonical `(A, b, c)` form. Both importers follow the same
rule: **convert losslessly or refuse.** A converter that silently approximates
would make every number on this page meaningless.

| Library | Module | Script | Converts losslessly |
|---|---|---|---|
| Gset (Max-Cut) | `Funcs_Qubo_MaxCut` | `Skript_Solve_Gset.py` | everything — Max-Cut *is* a QUBO |
| MIPLIB 2017 (MPS) | `Funcs_Qubo_MpsImport` | `Skript_Solve_MIPLIB.py` | pure set packing only; anything else is rejected |

## Gset — Max-Cut

Max-Cut needs **no penalty term**: it is natively unconstrained, so there is a
single energy scale rather than two annealing against each other. With `W` the
weighted adjacency matrix, `A = W` and `b = −W·1`, and the cut is exactly `−E`.

```bash
python Skript_Solve_Gset.py --steps 100000            # all nine below
python Skript_Solve_Gset.py --instances G1 --steps 300000
```

The nine instances as QUBOs. *Monomials* counts the quadratic terms
`x_i·x_j` with a nonzero coefficient — one per edge, since `A = W`. Note that
it does not track `n`: G70 has 10 000 variables but fewer terms than G1 with
800, which is why problem size cannot be read off the variable count alone.

| Instance | n | monomials | degree | `A` as CSR | dense would be |
|---|---|---|---|---|---|
| G1 | 800 | 19 176 | 47.9 | 463 KB | 5.1 MB |
| G11 | 800 | 1 600 | 4.0 | 42 KB | 5.1 MB |
| G14 | 800 | 4 694 | 11.7 | 116 KB | 5.1 MB |
| G22 | 2 000 | 19 990 | 20.0 | 488 KB | 32.0 MB |
| G32 | 2 000 | 4 000 | 4.0 | 104 KB | 32.0 MB |
| G55 | 5 000 | 12 498 | 5.0 | 320 KB | 200 MB |
| G60 | 7 000 | 17 148 | 4.9 | 440 KB | 392 MB |
| G70 | 10 000 | 9 999 | 2.0 | 280 KB | 800 MB |
| G81 | 20 000 | 40 000 | 4.0 | 1.04 MB | 3.20 GB |

Random starting points, no warm start, no problem-specific moves. `num_MC`
comes from the [efficiency plateau](performance.md#monte-carlo-trials-batching-and-where-it-stops-paying)
and the temperature from σ ([Tuning](tuning.md)); neither is hand-tuned per
instance. Budget is `steps ∝ n`, which is why the step counts differ.

| Instance | `num_MC` | steps | cut found | best known | % | time |
|---|---|---|---|---|---|---|
| G1 | 188 | 30 000 | 11 624 | 11 624 | **100.00** | 117 s |
| G11 | 188 | 30 000 | 560 | 564 | 99.29 | 115 s |
| G14 | 188 | 30 000 | 3 055 | 3 064 | **99.71** | 108 s |
| G22 | 75 | 30 000 | 13 317 | 13 359 | **99.69** | 105 s |
| G32 | 75 | 30 000 | 1 376 | 1 410 | 97.59 | 120 s |
| G55 | 30 | 100 000 | 10 189 | 10 299 | 98.93 | 315 s |
| G60 | 21 | 100 000 | 13 984 | 14 188 | 98.56 | 315 s |
| G70 | 15 | 300 000 | 9 373 | 9 591 | 97.73 | 443 s |
| G81 | 8 | 300 000 | 13 658 | 14 060 | 97.14 | 461 s |

Mean 98.74 %. One run per cell, one seed. Against the previous configuration —
a flat 100 000 steps and the scan-correction schedule — the mean was 97.80 %
and G1 stood at 99.72 %.

**Two things the variable count hides.** Budget is one: 100 000 steps is 125
sweeps for G1 and 5 for G81, and G81 gains 1.8 points from budget alone once
that is fixed. Structure is the other:

| Instance | components | isolated vars | effective n |
|---|---|---|---|
| G1, G11, G14, G22, G32, G81 | 1 | 0 | = n |
| G55 | 32 | 31 | 4 969 |
| G60 | 45 | 43 | 6 957 |
| **G70** | **1 598** | **1 354** | **8 646** |

G70 is not one 10 000-variable problem but 1 598 independent ones, 1 354 of
its variables affect no edge, and a single chain accepts one flip per step
across the whole vector — so every component competes for the same budget.
`load_gset` reports this in `info`. Component-wise decomposition would be the
obvious fix and is not implemented.

That does **not** explain G55 and G60, where under 1 % of variables are
isolated and the results still plateau. The untested candidate there is the
number of independent chains, which falls from 188 to 8 as `n` grows because
`num_MC·n` is held on the efficiency plateau — chain count and instance size
are confounded in this table and were never varied independently.

Graph *degree* does not separate the results: G1 (degree 48) and G11
(degree 4) both land near 99.5 %. That hypothesis was tested and failed.

Best-known values are compiled from
[this benchmark repo](https://github.com/0816keisuke/max-cut-problem-benchmark)
(tracing to Stanford's Gset page and the Toshiba SBM benchmark), G81 from
[arXiv:2505.18508](https://arxiv.org/abs/2505.18508). **These values drift** —
a new G63 record appeared in October 2025 — so treat the table in
`Skript_Solve_Gset.py` as a snapshot. Sources disagree on G55 (10 299 vs
10 116); the higher, less flattering value is used.

## MIPLIB — set packing

A MIPLIB instance is a MIP, not a QUBO. Only **pure set packing** converts
without loss:

```math
\max \sum_i c_i x_i \quad \text{s.t.} \quad \sum_{i \in S_r} x_i \le 1
\Longrightarrow
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

## What the two cases show

The gap between 100 % and 94 % is the **ΔE spectrum**, not size or difficulty.
Max-Cut has spread-out ΔE, so temperature can sort moves and the useful range
is broad. Unweighted set packing collapses ΔE onto `{−1, +1, +3, …}` with a
uniform `+1` barrier, leaving temperature nothing to sort — the workable
window shrank to 0.15–0.18, and outside it the run either froze at the greedy
value or became a random walk. Details in [Tuning](tuning.md).

Suitability, in the order the evidence supports it:

1. **Natively unconstrained** — penalties create two competing energy scales.
2. **Weighted, spread-out coefficients** — a degenerate spectrum starves the
   Metropolis criterion.
3. **Solution density near 50 %** — at 2.6 %, single flips are almost always
   meaningless.
4. **Budget `steps ∝ n`**, and check the component structure first.
