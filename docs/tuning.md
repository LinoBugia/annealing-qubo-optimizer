# Tuning: the two settings that decide everything

The temperature and the step budget. Everything else is secondary — get these
two wrong and no schedule saves you.

## Temperature: correct for the scan

A DA step evaluates **all `n` flips** and accepts at most one. So the
acceptance probability *per flip* must be `p/n`, not `p` — which makes the
right temperature far lower than intuition suggests:

```math
T \;\approx\; \frac{\Delta E_{\text{barrier}}}{\ln(n / p)}
```

with `p` the target chance that some uphill flip is accepted in a step (0.3 is
a good default) and `ΔE_barrier` **the barrier that actually blocks the
search** — which is *not* always the median:

- **Spread-out spectrum** (real-valued coefficients, as in Max-Cut): use the
  **median** uphill ΔE. There is no mass at any one value, so the median is
  representative.
- **Degenerate spectrum** (unweighted problems, where ΔE collapses onto
  `{+1, +3, +5, …}`): use the **minimum** uphill ΔE. Thousands of flips sit on
  the lowest barrier, and it is those that flood the admissible set. Taking the
  median here gives a temperature several times too hot.

Measured against a temperature sweep on four unrelated problem classes:

| Instance | n | uphill ΔE used | formula says | measured optimum | ratio |
|---|---|---|---|---|---|
| G1 (Max-Cut) | 800 | 5.0 (median) | 0.63 | 0.8 | 1.3 |
| G22 (Max-Cut) | 2 000 | 3.0 (median) | 0.34 | 0.6 | 1.8 |
| cdc7-4-3-2 (set packing) | 11 811 | 1.0 (min) | 0.095 | 0.18 | 1.9 |
| z26 (independent set) | 17 937 | 1.0 (min) | 0.091 | 0.15 | 1.65 |

The formula lands within a factor 1.3–1.9, consistently **low**. That
direction is the safe one: too cold degrades gently, too hot collapses. On G1,
`T=4` (6× the optimum) costs 5% of the cut; `T=0.2` (¼ of it) costs 0.4%.
**When unsure, go colder.**

**Picking the wrong statistic is not a small error.** On z26 the median uphill
move is 7 and the minimum is 1. Using the median gives `T = 0.64`, which is
4.2× the measured optimum — and at that temperature roughly 590 of the 2 832
minimum-barrier flips are admissible in *every* step. With hundreds of
simultaneously admissible moves the DA picks uniformly among them and the run
degenerates into a random walk: six temperatures from 0.32 to 3.8 all returned
exactly the starting solution, while `T = 0.15` improved it by 9% in under one
sweep.

A cheap diagnostic, needing no reference solution: if the best state returned
is **infeasible** for a problem whose start was feasible, the run is too hot.
On z26 that flips sharply — 0 of 12 trials need repair at `T ≤ 0.18`, 12 of 12
at `T = 0.22`.

This correction is not specific to any problem family — it follows from the
scan itself. The `da_gp` family in
[Graph-partitioning specifics](graph-partitioning.md) applies the same formula,
but wraps it in a calibration tuned for one problem type; the formula above is
the part that generalises.

## Step budget: it must scale with n

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
