# Tuning

Temperature and step budget decide almost everything here. Temperatures are
written in units of σ, a number the PBF defines before any run; on problems
with constraints that one number splits into two.

## σ, the unit of temperature

σ is the standard deviation of ΔE over all `n·2ⁿ` directed single-flip
edges of the hypercube — a property of the PBF, not of a run, and our
reference point for temperature. In the Walsh basis it is one sum. With
`x_i = (1 + s_i)/2` and the characters `χ_U = Π_{i∈U} s_i`, a PBF
`E(x) = Σ_S a_S Π_{i∈S} x_i` has the coefficients

```math
\hat e(U) = \sum_{S \supseteq U} a_S \, 2^{-|S|}
```

Flipping bit `k` negates exactly the characters that contain `k`, and the
characters are orthonormal under the uniform measure, so

```math
\sigma^2 = \frac{4}{n} \sum_U |U| \, \hat e(U)^2 .
```

The mean of ΔE over the directed edges is zero — the reverse of every edge
carries `−ΔE` — so this is the whole variance. For a QUBO with
`Asym = A + Aᵀ` the coefficients are `ê_i = b_i/2 + rowsum(Asym)_i/4` and
`ê_ij = Asym_ij/4`: one row sum and one sum of squares, linear in the
monomials (`qa3.delta_e_sigma`, about 100 ms at five million nonzeros).

### Background: why the schedules end where they do

σ is also the quantity our high-temperature approximation of the chain is
built on. We studied single-flip SA and DA exactly on small instances, up to
`n = 8`, where the whole transition matrix can be written down. Its
second-largest eigenvalue `λ₂` sets how fast a chain relaxes: the spectral
gap `1 − λ₂` is the inverse of the relaxation time. At `T → ∞` both chains
are the same random walk on the hypercube and their gaps agree. As `T`
falls they part, on the scale of σ:

- above `T ≈ σ` DA is practically SA — at `T = σ` its gap is only about 1.7
  times that of SA;
- between about `0.4σ` and `0.1σ` the advantage switches on;
- at `0.1σ` the gap of DA is 98 % of `n` times that of SA on instances with a
  single local minimum, and about two thirds of that on instances with
  several.

Colder, the ratio gains little more, while on landscapes with several minima
both chains slow down. That band is where Digital Annealing gets its speed —
a step that scans all `n` flips against one that tries a single flip — and
it is why our cooling schedules run into it and end at its cold edge. All
of this is still under investigation; if it holds up, we will publish it
together with the derivation.

## Without constraints: stop in the band

```python
cooling_param = ["sigma", 0.3, 0.11]      # geometric, 0.3σ down to 0.11σ
```

The best constant temperature sits at `0.116σ` on G1 and `0.134σ` on G22.

The cold end of a window carries the result, the hot end barely matters.
Over a grid of windows at 30 000 steps the outcome peaks at `c_end ≈ 0.11`
and falls off on both sides — ending at `0.01σ` cost about 30 cut on G1 —
while any start from `0.15σ` to `0.5σ` lands within noise; only a start
below the band breaks things. With ten times the steps the best end moves
slightly colder: at 300 000 steps no window beat an end at `0.08σ` on G1,
G11, G14, G22 and G32, where G1 and G11 reached the best known cut and G14
and G22 ended one below it. **Cool into the band and stop there, not
through it.**

## With constraints: two scales

A penalty encoding `E = f + P·g` has a gapped spectrum: moves along the
feasible set cost the objective, moves that break a row cost `P`, and there
is little in between. The spread `q50(ΔE⁺) / min(ΔE⁺)` of the uphill flips
at the start state tells the two cases apart before a run — 1 to 5 on
Max-Cut, 15 or more on a penalty encoding. One σ over all of `E` then
measures neither scale: on `cdc7-4-3-2` it reads 420, because random states
break hundreds of rows, while the chain on its feasible packing sees ±1 from
the objective and `P` per broken row. So each part gets its own σ:

- **σ_c**, the objective alone: the Walsh σ above, of `f` without the
  penalty.
- **σ_P**, the penalty alone, where the chain is: at a state where every row
  a variable touches sits at its bound, flipping it costs `Σ_r P_r a_rj²` —
  `P` times its number of rows, for unit coefficients — and σ_P is the RMS
  of that over the variables. At the actual start state only the rows at
  their bound count. The Walsh σ of the penalty over random states is the
  number to avoid: 1.3·10⁷ on `t1722`, where a random state sets about half
  of every row that should hold exactly one.

Each scale becomes a temperature through the scan. A DA step looks at all
`n` flips and accepts at most one, so the per-flip acceptance has to be
about `p/n`, and a barrier `ΔE` is crossed at

```math
T \approx \frac{\Delta E}{\ln(n / p)}, \qquad p \approx 0.3 .
```

That gives a **cost limit** `T_c = σ_c / ln(n/p)`, below which the objective
freezes, and a **constraint limit** `T_P`, the cheapest flip that breaks a
row over the same logarithm, below which no row breaks any more. On
`cdc7-4-3-2` they are 0.095 and 0.38, and the best constant temperature
measured there, 0.18, sits near their geometric mean; with the full `P`
throughout, holding it beat every cooling window into it (283–285 against
277–282). On a weighted 3-colouring of a random graph the limits are 0.46
and 1.78, on `t1722` 208 and 689. On a gapped spectrum the barrier that counts is the smallest one, not
the median: on an independent-set instance the median put `T` at four times
the best value and the run became a random walk. The rule lands low by a
factor of 1.3 to 1.9 on every instance we checked, which is the safe
direction — too cold degrades gently, too hot collapses. **When unsure, go
colder**, and if a run that started feasible returns an infeasible best
state, it was too hot.

**Single flips between the two limits.** With only single flips, every move
from one feasible state to another on an equality row passes a state with a
broken row, so the objective moves only while `T` is near `T_P`; cooling
further freezes the chain. We found no temperature schedule that gets around
this. Cycles of cooling to `T_c` and reheating to just below or just above
`T_P`, with the reheats shrinking from cycle to cycle, never beat a single
curve over the same span, on any instance. Holding at `0.9 T_P` helps on
equality rows and hurts on packing rows, where the chain moves through
smaller packings at the objective's scale.

**What works is lowering `P`.** In the cost phase `P` goes down to where one
broken row costs what a step at `0.2 σ_c` accepts,

```math
P_{\text{low}} \approx \frac{0.2\,\sigma_c\,\ln(n/p)}{\text{rows a flip breaks}}
```

and the last fifth of the run puts the full `P` back so that the result
holds. That brings the constraint limit down to the objective's scale: the
chain can break a row and still tell a good move from a bad one. On the
3-colouring the weight of monochromatic edges drops from 17.4 under a
geometric schedule to 1.6, against 1.28 as the best value found by any
method; a balanced bisection reaches the best known cut in every chain; on
`cdc7-4-3-2` the packing grows to 283.5, against 275 for the best constant
temperature. The rule is right within a factor of two either way — the best
factor was 0.5, up to 1 and 2 on these three — so where it is not known,
run cost phases at 0.5, 1 and 2 times the rule with the full `P` between
them: loosen, tighten, loosen less, tighten, then leave it. That was never
the best and never bad.

Single flips fall short where the feasible set itself is hard to reach: on
`supportcase14/16`, with ±1 equalities that need slack variables, at most 7
of 64 chains became feasible under any schedule, and on `t1722` (36 630
columns, 338 exactly-one rows) none in 300 000 steps. That is a limit of the
move set, not of the temperature. These results rest on two generated
problems and four MIPLIB instances, two to five seeds each, measured with
code that is not in this repository yet — the MPS importer here reads pure
set packing only.

**What comes next.** We are developing a framework that lets Digital
Annealing tunnel through constraint barriers, which we call
semi-modifications: moves that change several variables at once and keep
the rows satisfied, where a single flip has to break one. The first results
are very good. We are turning them into a clean framework and will release
it here soon.

## The E_Offset

The offset raises the acceptance threshold on steps where nothing was
admissible and resets on every accepted flip, so it only builds up across
consecutive blocked steps. Its worth follows from the blocked share, which
`Bench_Diagnostics.diagnose()` measures before a run: 0.00–0.19 % on the
Max-Cut instances, where it never fires, and 11.7 % on `cdc7-4-3-2`, where
it is worth about +3, roughly a doubling of the budget. Its unit is `T`,
since acceptance scales as `exp(offset/T)`. A phase plan, `kick`, exists but
has not beaten a single well-chosen rate. Contract in
[Parameter reference](configuration.md#the-e_offset).

## Step budget

One accepted flip per step means `steps / n` passes over the variables, so a
flat step count starves large instances: 100 000 steps is 125 sweeps for G1
and 5 for G81. Budget `steps ≈ k·n` with `k` in the low hundreds; the
default of 1000 suits `n` in the hundreds and nothing above. The per-step
cost is O(mc·n), so `steps ∝ n` makes the total O(n²) even on a sparse
problem.
