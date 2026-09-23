# Benchmark instances

Results on two public benchmark libraries, and the importers that convert
them losslessly into the canonical QUBO form.

Two importers turn public benchmark libraries into the canonical `(A, b, c)`
form. Both follow the same rule: **convert losslessly or refuse.** A converter
that silently approximates would make every number on this page meaningless.

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

Random starting points, no warm start, no problem-specific moves. `num_MC` is
derived from the [efficiency plateau](performance.md#monte-carlo-trials-batching-and-where-it-stops-paying)
and `T` from the [scan correction](tuning.md#temperature-correct-for-the-scan) — neither
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

All nine ran at a flat 100 000 steps, which starves the large instances —
see [Step budget](tuning.md#step-budget-it-must-scale-with-n). Re-running the
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
[efficiency plateau](performance.md#monte-carlo-trials-batching-and-where-it-stops-paying);
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

## MIPLIB — set packing

A MIPLIB instance is a MIP, not a QUBO. Only **pure set packing** converts
without loss:

```math
\max \sum_i c_i x_i \quad\text{s.t.}\quad \sum_{i \in S_r} x_i \le 1
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
