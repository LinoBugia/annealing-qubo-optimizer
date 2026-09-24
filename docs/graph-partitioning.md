# Graph-partitioning specifics (GP)

The `da_gp` schedule family and the parameters marked **(GP)** in the
[Parameter reference](configuration.md).

> **Scope.** Everything in this document is calibrated for **one** problem
> family: warm-started binary bisection of a Gram matrix, as used by the
> `gp-qubo-rag-indexer`. The workflow is `create_graph_binary_clustering_qubo`
> → `lloyd_bisect` warm start → `da_gp` cooling → `auto_gp` offset. The
> calibration assumes a Lloyd solution exists (`E_warm`, called `E_Lloyd` in
> the code) and that the landscape depth `D` is meaningful. It is a
> schedule *generator for that problem*, not a general-purpose default —
> applying it elsewhere is untested and the `gamma` cap in particular has no
> meaning without a warm start to protect.

## The `da_gp` contract (slots)

```
[0] "da_gp" | "da_gp_floor"
[1] n_steps        number of steps (enters the end-temperature condition)
[2] delta_E_vec    ΔE vector at the start — leave None, the optimizer fills it
                   at the REAL start vector of each group
[3] start_assign   start vector (imbalance correction only)
[4] E_warm         energy at the warm start ("E_Lloyd")
[5] E_start        energy of a random state — leave None, the optimizer fills it
                   once globally (landscape depth D)
[6] p_hot          target acceptance of a median uphill flip at t=1
[7] p_cold         residual acceptance of a small (q25) flip at the end
[8] gamma          cap: T(1) <= gamma · landscape depth D
[9] [10]           cache: calibrated c and d (written by the calibrator)
[11] [12]          cache: T_hot_capped and T_freeze (for da_gp_floor)
```

## Calibration

```
q25, q50   quantiles of the UPHILL flips (ΔE > 0) at the start
T_hot      = q50 / ln(n / p_hot)          scan correction — see Tuning,
T_freeze   = q25 / ln(n / p_cold)         this part is NOT gp-specific
d          = 2.22  (fixed)
c          = min( T_hot·ln2 ,  T_freeze·ln(1+S^d) ,  gamma·D·ln2 )
D          = E_start − E_warm             landscape depth
```

- **`p_hot`** (0.3) sets how mobile the start is, **`p_cold`** (0.01) how hard
  the end freezes — larger is hotter in both cases.
- **`gamma`** (0.05) caps `T(1)` at 5% of the landscape depth so the warm
  start cannot melt. Affects temperature **only**, never the E_Offset.
- **Prefer `da_gp_floor`**: when the `T_hot` term binds (usually), plain
  `da_gp` ends *below* the freezing bound — measured 0.134 against 1.175, 8.8×
  too cold. `da_gp_floor` runs toward `T_freeze` and never undercuts it, with
  `d` controlling only the curvature.

## The three exploration controls

| Control | When it acts | Effect |
|---|---|---|
| `hold_steps` | start | extends the hot phase before cooling begins |
| `offset_k_escape` **(GP)** | when a trial is stuck | raises the energy level until a flip gets through — acts **only** on fully rejected steps |
| `p_hot` / `gamma` **(GP)** | whole curve | how hot the curve starts at all |
