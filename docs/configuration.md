# Parameter reference

Every parameter of `qubo_min_solver`, what the call returns, and every cooling
schedule type. The two parameters that decide whether a run works at all have
their own document: [Tuning](tuning.md).

## `qubo_min_solver(A, b, c, ...)`

Entries marked **(GP)** exist for the warm-started graph/Gram bisection
workflow — see [Graph-partitioning specifics](graph-partitioning.md).
They are not general-purpose defaults.

| Parameter | Default | Meaning |
|---|---|---|
| `A, b, c` | — | QUBO in canonical form. `A` symmetric with **zero diagonal**, dense float32/64 or scipy CSR; `b` is `(n,)`; `c` shifts the level only, never the argmin. |
| `type_alg` | `"digitalAnnealing"` | Only value. SA/SCA/TSP live in the reference library. |
| `steps` | `1000` | Annealing steps **per trial**. One step = one acceptance scan over all `n` flips and **at most one** accepted flip — so budget `steps ≈ k·n`, see [Step budget](tuning.md#step-budget). The default suits `n` in the hundreds only. |
| `num_MC` | `8` | Monte-Carlo trials **per start group**. All trials of a group run as one `(num_MC, n)` batch — see the [batching numbers](performance.md#monte-carlo-trials-batching-and-where-it-stops-paying). |
| `cooling_param` | `None` → `["logarithmic", 10]` | Schedule contract, see [below](#cooling_param-by-schedule-type). |
| `seed_rand` | `42` | Master seed of the RNG blocks. **The only persistence of randomness** — same seed and parameters give a bit-identical run. Random numbers are never reused. |
| `seed_gen_initial_varAssignment` | `42` | Seed for generated random start vectors, and for the reference random state used by `da_gp`. |
| `initial_varAssignements_pre` | `None` | `None` → random start(s) per `random_start`. `(n,)` → **one** start group. `(S,n)` → **S start groups**, `num_MC` trials each. |
| `random_start` | `False` | Only without supplied start vectors: `True` = every trial starts at its own random point. |
| `offset_increase_rate` | `0.0` | E_Offset increment on steps where **no** flip was accepted; resets to 0 on every accepted flip, so it only accumulates across *consecutive* blocked steps. `0.0` = off. `"auto_gp"` **(GP)** = derived from the measured ΔE scale. See [The E_Offset](#the-e_offset). |
| `offset_k_escape` | `25.0` | **(GP)** Scales the E_Offset under `"auto_gp"`: `rate = q50(ΔE⁺) / k_escape`. Smaller = stronger offset = more aggressive escape. Unrelated to `gamma`. |
| `kick` | `None` | σ-calibrated offset kick, a **phase plan** fixed before the run. `None` = off, and the step loop is then byte-identical. See [The E_Offset](#the-e_offset). |
| `cooling_per_group` | `True` | Each start group gets its **own** schedule, calibrated at its own start vector. Needed whenever start vectors sit at different depths — a schedule calibrated on a deep warm start is too hot for shallower ones. `False` = one shared schedule from group 0. |
| `hold_steps` | `0` | **Plateau**: hold the start temperature for this many steps before the curve falls. Added **on top** — the run is `steps + hold − 1` long and the cooling curve keeps its full `steps` support points, so the end temperature is unchanged. `0` and `1` both mean off. |
| `save_addinfo` | `True` | Carry the minimum state vectors `X_min` (otherwise energies only). |
| `save_csv` | `False` | Write `Runs/Evaluation_<date>/`: summary, trajectory and add-info CSV plus the plot as HTML. **Writes one directory per run.** |
| `visual_inst` | `False` | Open the Plotly visualiser. |
| `recompute_every` | `1024` | Every k steps, recompute gradient `G` and energy `E` **exactly** from `X` instead of updating incrementally — guards against float drift on long runs. Costs ~6 µs/step at the default; see [the measurements](performance.md#where-the-time-goes). |
| `mem_budget_mb` | `64.0` | Memory budget of the bulk RNG blocks. Larger = fewer refills, more RAM. Does **not** affect the result — only the block size, not the number sequence. |

**Returns** `(Min_varAssignements, Mins, Trajectories, Infos)` — same order as
the reference. All lists have length `S × num_MC`, sorted by group.
`Infos` carries: `labels`, `group_of_trial`, `T` (group 0's schedule, for
compatibility), `Ts` (schedule **per group**), `ExecTimes`, `Offsets`,
`offset_increase_rate` and `offset_rates`, `cooling_c` (calibrated `c` per
group), `cooling_per_group`, `best`, `E_best`, `X_best`.

## `cooling_param` by schedule type

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


## The E_Offset

The offset raises the acceptance threshold on steps where **nothing** was
admissible, and resets to zero on every accepted flip. It therefore only
accumulates across *consecutive* blocked steps, and it is worth exactly as
much as blocked steps are frequent.

Measure that share before reaching for it — `Bench_Diagnostics.diagnose()`
reports it, and it is the ceiling on what the mechanism can do:

| | blocked steps | E_Offset worth |
|---|---|---|
| Max-Cut (G1, G11, G14, G22) | 0.00–0.19 % | nothing measurable; the offset never grew once in 10 000 steps |
| set packing (cdc7-4-3-2) | 11.7 % | **+3**, about a doubling of the budget |

Acceptance scales as `exp(offset / T)`, so the meaningful unit is **T**, not
the height of the barrier being crossed. Measured on cdc7 at `T = 0.18`, a
rate of `0.05` and one of `0.6` both give the same result for opposite
reasons: the small one creeps up over several blocked steps and escapes
through the smallest gap, the large one clears the step in one go and then
resets, so it cannot compound. Rates far below that (`0.02`, `0.01`) never
approach the barrier at all and lose ground.

### `kick` — an offset plan, not a rule inside the step

The plain rate is one number for the whole run. `kick` replaces it with a
schedule of *methods*, resolved once before the loop into a per-step mode
array. The step itself only reads an integer, so the policy is entirely
external and the kernel makes no decisions.

```python
kick = {
    "sigma": sigma_local,                  # required
    "phases": [(0.5, "none"), (0.25, "wide"), (0.25, "low")],
    "w": 3,                                # blocked steps before it may fire
    "c_low": 0.1, "c_wide": 1.0,           # widths, in units of sigma
    "check_every": 16,
}
```

Three modes, and what each is for. `g_min` is the smallest ΔE at the current
state, taken fresh at every firing:

| mode | rule | for |
|---|---|---|
| `none` | no offset, reset the counter, escape thermally | the **opening phase** — early on the chain is far from any minimum and blocked steps are rare |
| `wide` | `E_off = g_min + c_wide·σ` | a broad escape. Works well when there are **no penalty terms**; alternate it with `none` |
| `low` | `E_off = g_min + c_low·σ` | a narrow escape just above the smallest gap |

A fourth, `band`, picks directly and uniformly among the directions in
`[g_min + c_b1·σ, g_min + c_b2·σ]` — the only mode that strictly prefers the
higher ones. It falls back to `wide` when the band is empty.

**Measured so far: no gain.** On cdc7-4-3-2 at 100 000 steps a fixed rate of
0.6 gives 282; the plan gives 282 with its default widths and 281 with widths
scaled to the instance, at 3–7 % more wall clock. That is a single instance,
and the only one available where the trigger fires at all.

**Every combination is expressible**, because the plan is just a list of
fractions. Offset always on: `[(1, "wide")]`. Always off: `kick=None`.
Alternation needs no extra machinery — it is simply more phases:

```python
"phases": [(0.4, "none"), (0.1, "wide"), (0.1, "none"),
           (0.1, "wide"), (0.1, "none"), (0.2, "low")]
```

**Which σ.** The widths are multiples of σ, and *which* σ matters. The exact
Walsh σ (`qa3.delta_e_sigma`) averages over uniformly random states. For a
penalty encoding those are all deeply infeasible: on cdc7-4-3-2 it reads 420
while the feasible packing the chain actually occupies has a local scale of
16.7. With 420, even `c_low` would be a `+42` offset and the kick would accept
anything. **On a constrained problem, pass σ measured at the start state.**

**Cost.** `None` is the default and leaves four scalar `is not None` checks
per step, with byte-identical output — verified against the kernel as it was
before the mechanism existed. When on, the trigger test and the RNG draw
happen only every `check_every` steps; per step only the blocked counter is
updated, as one fused `(mc,)` operation. `check_every` changes *when* a kick
fires, never *what* it does: 1, 4, 16 and 64 all produce the same result,
because a blocked chain stays blocked and only the deadline moves.
