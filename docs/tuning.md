# Tuning

Temperature and step budget decide almost everything here. What follows is
what the measurements actually said, on Max-Cut and on set packing — the two
behave differently enough that the same settings cannot serve both.

## Two regimes, told apart by one number

Take the uphill ΔE spectrum at the start state and look at
`spread = q50(ΔE⁺) / min(ΔE⁺)`.

On Max-Cut it comes out around 1–5: the spectrum is dense, no barrier height
is special. On a penalty encoding it is 15 or more, because the coefficients
split into "moves along the feasible set" and "moves that break a constraint",
with nothing in between. That gap changes which rules apply.

## Unconstrained: temperatures as multiples of σ

For a dense spectrum the useful unit is `σ = std(ΔE)`, computed in closed form
from the Walsh coefficients (`qa3.delta_e_sigma`) — linear in the monomials,
about 100 ms even at five million nonzeros.

```python
cooling_param = ["sigma", 0.3, 0.11]      # geometric, 0.3σ down to 0.11σ
```

The point of σ is that it transfers. The measured best constant temperature
sits at `0.116σ` on G1 and `0.134σ` on G22 — two graphs whose degrees differ
by a factor of twelve. An absolute temperature would have to be re-found for
each.

Two numbers are enough, and the cold one carries the result. Sweeping a grid
of windows, the mean outcome peaks at `c_end ≈ 0.11` and falls off on both
sides: **cool down to the optimal temperature and stop there rather than
through it.** Ending at `0.01σ` cost about 30 cut on G1. The hot end barely
matters — anything from `0.15σ` to `0.5σ` landed within noise, and only
starting *below* the optimum broke things.

What it was worth: with 30 000 steps the σ schedule reached 11 624 on G1,
matching the best known value, and 13 317 on G22 — against 11 591 and 13 154
from a flat 100 000 steps on the old scan schedule. The reason is visible in
the bands. On G22 the old curve started at 0.511, already below the optimum of
0.6, and fell from there; the whole run sat too cold, and more steps at the
wrong temperature bought little.

The advantage shrinks when the budget does. At 100 000 steps the large
instances gained +86 on G70 and tied on G81, where five sweeps over 20 000
variables leave the run far from converged and the curve shape cannot help.

`staircase` and `linear` also exist; linear spends most of its budget above
the useful band and measurably loses.

## With penalty terms: σ stops working

σ assumes the chain lives where σ is measured, and a penalty encoding breaks
that. On `cdc7-4-3-2` the chain starts on a feasible packing and never leaves
it, where ΔE is ±1 — but σ averages over uniformly random states, which are
all deeply infeasible, and reads **420**. At the state the chain actually
occupies the scale is **16.7**. The measured optimum is 0.18, so `T_opt/σ` is
0.0004 there against 0.12 on Max-Cut, and the Max-Cut defaults would run it
117× too hot.

Here the older rule is the right one. A DA step scans all `n` flips and
accepts at most one, so the per-flip acceptance has to be `p/n`:

```math
T \approx \frac{\Delta E_B}{\ln(n / p)}
```

with `p ≈ 0.3` and `ΔE_B` the barrier that actually blocks the search — on a
gapped spectrum the **minimum** uphill ΔE, not the median, since thousands of
flips sit on the lowest barrier and those are what flood the admissible set.

| Instance | n | ΔE used | formula | measured | ratio |
|---|---|---|---|---|---|
| G1 (Max-Cut) | 800 | 5.0 (median) | 0.63 | 0.8 | 1.3 |
| G22 (Max-Cut) | 2 000 | 3.0 (median) | 0.34 | 0.6 | 1.8 |
| cdc7-4-3-2 (set packing) | 11 811 | 1.0 (min) | 0.095 | 0.18 | 1.9 |
| z26 (independent set) | 17 937 | 1.0 (min) | 0.091 | 0.15 | 1.65 |

It lands low by 1.3–1.9 every time, which is the safe direction: too cold
degrades gently, too hot collapses. On G1, six times the optimum costs 5 % of
the cut while a quarter of it costs 0.4 %. **When unsure, go colder.**

Taking the median on a gapped spectrum is not a small error. On z26 it gives
`T = 0.64`, four times the optimum, and six temperatures from 0.32 to 3.8 all
returned exactly the starting solution — the admissible set was so large the
run was a random walk. At 0.15 the same instance improved 9 % in one sweep. A
free check: if a run that started feasible returns an **infeasible** best
state, it is too hot.

And here the temperature wants to stay put. Cooling windows into the optimum
all lost against simply holding 0.18 — 277 to 282 against 283–285 — and the
warmer ones were nearly twice as slow, since more accepted flips mean more
sparse updates.

**One caveat on the evidence.** The dense-spectrum rules rest on nine Gset
instances; the gapped-spectrum rules rest on `cdc7-4-3-2` plus one instance
(`z26`) that is not in this repository, because the MPS importer accepts pure
set packing only and MIPLIB holds exactly two such instances — the other being
solved optimally by the greedy baseline. So the two halves of this page are
not equally supported, and the gapped half should be read as one well-measured
case rather than a law.

## The E_Offset

It raises the acceptance threshold on steps where *nothing* was admissible and
resets on every accepted flip, so it only builds up across consecutive blocked
steps. That makes its value easy to predict: measure the blocked share first
(`Bench_Diagnostics.diagnose()`).

On the Max-Cut instances that share is 0.00–0.19 % and the offset never grew
once in 10 000 steps. On cdc7 it is 11.7 % and the offset is worth about +3,
roughly a doubling of the budget.

Acceptance scales as `exp(offset/T)`, so the unit is **T**, not the barrier.
At `T = 0.18`, rates of 0.05 and 0.6 performed identically for opposite
reasons — the small one creeps up over several blocked steps and slips through
the smallest gap, the large one clears the step at once and resets, so it
never compounds.

Since the right setting is instance-dependent, `kick` can replace that single
number with a phase plan fixed before the run — `none` for the opening, where
the chain is rarely blocked, then `wide` (a broad escape) or `low` (a narrow
one just above the smallest gap), alternating as you like.

**So far it has not paid.** On cdc7 at 100 000 steps, against a fixed rate of
0.6 giving 282: the plan with its default widths also gave 282, and with
widths scaled to the instance, 281 — inside the ±2 spread, and 3–7 % slower.
That is one instance, and it is the only one we have where the mechanism can
fire at all, since Max-Cut never blocks. Until that changes, a single
well-chosen rate is the better bet. Contract in
[Parameter reference](configuration.md#the-e_offset).

## Step budget

One accepted flip per step means the passes over the variables are
`steps / n`. A flat step count starves large instances — 100 000 steps is 125
sweeps for G1 and 5 for G81, and the resulting 99.7 % against 95.3 % says
nothing about how the annealer scales, only that one instance got a hundredth
of the work per variable.

Budget `steps ≈ k·n` and tune `k`; the low hundreds is a sensible start. The
default of 1000 suits `n` in the hundreds and nothing above. This follows from
the one-flip-per-step semantics, not the implementation: per-step cost is
O(mc·n), so `steps ∝ n` makes the total O(n²) even on a sparse problem.
