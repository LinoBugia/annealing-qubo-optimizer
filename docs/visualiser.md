# Visualiser `VisualizeRuns(...)`

What `VisualizeRuns` draws, and the display conventions it applies.

All trials in ONE figure, **colour = start group** — one legend click hides
the whole group (trajectory, ΔE band, minima and offsets together).

- **Energy** every chain individually; **Temp** one curve per group.
- **Delta Energy** aggregated per group (median + 25/75 band) rather than
  `num_MC` noise curves, drawn above that group's T curve so the two are
  directly comparable — it is the quantity the cooling was calibrated against.
- **Mins** a marker swarm per group at its own x position, group minimum as a star.
- **E_Offset** genuinely tracked, not reconstructed from `ΔE == 0`.

## Minima that are equal are drawn as equal

When several trials reach the *same* solution their energies still differ in
the last float64 digits, because each accumulates its incremental updates in a
different order. Left alone, the Mins panel auto-scales onto that 1e-15 spread
and a single solution looks like a dozen competing ones.

`VisualizeRuns(..., min_display_rtol=1e-9)` therefore snaps minima that agree
within a relative tolerance onto one value, and when *every* trial agrees it
pins the y-axis to a readable window and labels it rather than zooming into
noise.

**This is a display convention, not a change to the results.** `Mins` and the
returned solutions keep their exact values, and the hover text shows the raw
number to 12 digits. `min_display_rtol=0.0` turns it off.

The default is deliberate. The drift is bounded by `recompute_every`, which
rebuilds `E` exactly every k steps, so at most k incremental updates can
accumulate: measured here, eight trials on one solution spread by ~2e-10 at
|E| ~ 1.8e5, a relative 1e-15. A 1e-9 tolerance clears that by six orders of
magnitude and still keeps genuinely distinct optima apart — whereas 1e-6 would
mean an absolute tolerance of 1.0 on a problem with |E| ~ 1e6, quietly merging
real solutions.
