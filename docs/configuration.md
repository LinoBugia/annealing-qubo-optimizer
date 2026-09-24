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
| `steps` | `1000` | Annealing steps **per trial**. One step = one acceptance scan over all `n` flips and **at most one** accepted flip — so budget `steps ≈ k·n`, see [Step budget](tuning.md#step-budget-it-must-scale-with-n). The default suits `n` in the hundreds only. |
| `num_MC` | `8` | Monte-Carlo trials **per start group**. All trials of a group run as one `(num_MC, n)` batch — see the [batching numbers](performance.md#monte-carlo-trials-batching-and-where-it-stops-paying). |
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
