# Tuning

The two settings that decide whether a run works at all: the temperature
and the step budget. Read this before tuning anything else.

Everything else is secondary. Get these two wrong and no schedule saves you.

## Temperature: correct for the scan

A DA step evaluates **all `n` flips** and accepts at most one. So the
acceptance probability *per flip* must be `p/n`, not `p` — which makes the
right temperature far lower than intuition suggests:

```math
T  \approx  \frac{\Delta E_{q50}}{\ln(n / p)}
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
[Graph-partitioning specifics](graph-partitioning.md#graph-partitioning-specifics-gp) applies the
same formula, but wraps it in a calibration tuned for one problem type; the
formula above is the part that generalises.

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
