"""
Funcs_Qubo_Randomizers — bulk RNG for the batched DA kernel.

Design rationale: the RNG is the single largest item in a vectorised DA step.
Measured (Bench_Performance, S9, mc=8): 37 % at n=1024, 39 % at n=4096,
44 % at n=8192 — the share grows with n. An earlier estimate of ~90 %
overstated it; it came from timing only the handing out of a slice from the
bulk block rather than the generation itself. The design consequence is the
same either way:
  - draw random numbers in large blocks and keep them in RAM (block size
    bounded by a memory budget, 64 MB by default),
  - run Metropolis through standard_exponential (ziggurat) instead of
    uniform + log:
        accept flip  <=>  dE - offset < T * eps ,  eps ~ Exp(1)
    (equivalent to rand < exp(-(dE-offset)/T); the dE <= 0 case is included
    automatically because offset >= 0 and eps > 0).

Random numbers are NEVER reused; the seed is the only persistence, which
gives reproducibility without an external API and therefore without latency.
"""

import numpy as np


class BulkRandomizer:
    """
    Block-wise supply of exponential and uniform variates.

    exponentials(mc, n): next (mc,n) slice of Exp(1)   (Metropolis scan)
    uniforms(mc):        next (mc,) slice of U[0,1)    (flip selection)
    """

    def __init__(self, seed=None, mem_budget_mb: float = 64.0):
        self.rng = np.random.default_rng(seed)
        self.mem_budget = float(mem_budget_mb) * 1e6
        self._exp_block = None      # (k, mc, n)
        self._exp_pos = 0
        self._uni_block = None      # (k, mc)
        self._uni_pos = 0

    def _exp_refill(self, mc, n):
        k = max(1, int(self.mem_budget / (8.0 * mc * n)))
        self._exp_block = self.rng.standard_exponential(size=(k, mc, n))
        self._exp_pos = 0

    def exponentials(self, mc: int, n: int):
        blk = self._exp_block
        if blk is None or blk.shape[1:] != (mc, n) or self._exp_pos >= blk.shape[0]:
            self._exp_refill(mc, n)
            blk = self._exp_block
        out = blk[self._exp_pos]
        self._exp_pos += 1
        return out

    def _uni_refill(self, mc):
        k = max(1, int(self.mem_budget / (8.0 * mc)))
        k = min(k, 1 << 16)                     # uniforms are cheap — a small block is enough
        self._uni_block = self.rng.random(size=(k, mc))
        self._uni_pos = 0

    def uniforms(self, mc: int):
        blk = self._uni_block
        if blk is None or blk.shape[1] != mc or self._uni_pos >= blk.shape[0]:
            self._uni_refill(mc)
            blk = self._uni_block
        out = blk[self._uni_pos]
        self._uni_pos += 1
        return out

    def spawn(self):
        """Independent child randomizer (e.g. one per start group)."""
        child_seed = self.rng.integers(0, 2 ** 63 - 1)
        return BulkRandomizer(child_seed, self.mem_budget / 1e6)
