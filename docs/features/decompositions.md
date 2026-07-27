# Decompositions - LU, CHO/CHOP, QR/LQ, QRCP, Bidiag

Dense factorizations. Every family follows the same four-token grid - `decomp` (factor, input
preserved), `decompInPlace` (factor into the input's own storage, input destroyed), `decompSolve`
(solve from existing factors, solve-many tier), `solveInPlace` (one-shot fused solve, fastest path,
destructive). Each also has a zero-alloc
`ref`-workspace overload plus an allocating convenience wrapper; several route their allocating
overload through a blocked (level-3, compact-WY/SYRK/GETRF-style) core above a measured size
crossover - the gain caps out around 1.3-1.4× (bounded by `matMatDot`'s own ~70 GFLOP/s ceiling, see
[la-primitives.md](la-primitives.md)).

- **`LU.decomp(in A, ref L, ref U, ref Pivot P)`** - partial-pivoting LU, `PA = LU`, A preserved.
  Blocked (GETRF-style: panel factor + TRSM + one GEMM trailing update) above `n ≥ 256` float /
  `n ≥ 128` double. Also
  `LU.decompInPlace(ref A_to_LU, ref Pivot P)` (compact in-place form), `LU.decompSolve`/
  `LU.solveInPlace`. (`determinant`/`logDeterminant` are scalar characterizations and live on
  [`Analysis`](la-primitives.md).)
- **`CHO.decomp(in A, ref L)`**- `A = LLᵀ`, SPD, A preserved. Blocked (POTRF: panel + TRSM + SYRK
  trailing update) above `n ≥ 1024` float / `n ≥ 512` double. Also `CHO.decompInPlace(ref A_to_L)`,
  `CHO.decompSolve`, `CHO.solveInPlace`. **`CHOP.decomp(in A, ref L, ref Pivot P, ref ws)`** -
  rank-revealing (xPSTRF-style) pivoted Cholesky, upper-triangle-only working storage, blocked
  (level-3, right-looking PSTRF with 32-wide panels) above `n ≥ 512`; returns
  `RankInfo`. Also `CHOP.decompSolve`/`CHOP.solveInPlace`.
- **`QR.decomp(in A, ref Q, ref R[, ref u[, ref w]])`** - Householder QR, A preserved (one memcpy into
  Q). The fully-allocating overload is blocked (compact-WY) above `N_Cols ≥ 128` float /
  `N_Cols ≥ 512` double; the `ref u`/`ref w`
  zero-alloc overloads stay unblocked. `QR.decompInPlace` is the same kernel factoring A's own storage
  in place. `QR.decompSolve(ref Q, ref R, ref b, ref x)` solves from existing factors (b preserved).
  `QR.solveInPlace(ref A, ref b, ref x[, ref u])` is the fused one-shot kernel: it streams Qᵀb without
  ever forming Q, so it's the fastest path, but A and b exit as undefined scratch (R+reflectors / Qᵀb) -
  not usable factors. **`QRCP.decomp(in A, ref Q, ref R, ref Pivot P[, ref u])`** (Businger-Golub) -
  exact (not downdated) column-norm recompute each step; `QRCP.decompInPlace` factors A_to_Q in place;
  `QRCP.solveInPlace(ref A_to_Q, ref b, ref x, ref R, ref Pivot P, ref u[, relTol])` factors A's own
  buffer directly (no Q scratch, no memcpy) and leaves
  A_to_Q as a *usable* orthogonal factor, unlike QR's solveInPlace.
  - **QR's three scratch tiers** (all producing bit-identical results, cheapest to richest): (1) the
    fully-allocating overload - convenience, `Allocator.Temp` scratch, gets the blocked path; (2) the
    raw `ref u[, ref w]` overloads - level-2 minimal scratch, always unblocked, no cache struct
    needed; (3) `ref floatQRCache cache` - zero-alloc AND blocked, carrying `u`/`w` plus the five
    compact-WY panel buffers (`Vpanel`/`Tbuf`/`Wbuf`/`tcolBuf`/`VfullBuf`). `QR.solveInPlace`'s cache
    overload only threads `u`/`w` from the cache (it never forms Q, so the blocked-WY buffers are
    dead weight for it) - its win over the allocating overload is purely the eliminated per-call
    `Allocator.Temp` allocation, **not** the blocked kernel; it is fused, not blocked. QRCP shares no
    equivalent panel cache: its pivot kernel recomputes column norms after every reflector, so it can
    never be blocked into panels.
- **`LQ.decomp(in A, ref L, ref Q[, ref ws])`** - direct row-Householder (GELQF-style, not
  transpose-to-QR), A preserved (read-only; the safety is free, not bought). The allocating overload
  blocks above `m ≥ 512` (a measured, not derived, crossover - LQ's fold step is reduction-shaped, so
  the double-precision crossover trails float's). `LQ.minNormSolve(in A, in b, ref x[, ref ws])`
  also only reads A and b.
- **`Bidiag.decomp(in A, ref U, ref B, ref V, ref ws)`** / **`Bidiag.values(in A, ref d, ref e, ref
  ws)`** - Golub-Kahan-Householder reduction; feeds [svd.md](svd.md)'s `SVD.thin`/`SVD.values`. Not
  yet raised to level-3 (GEBRD is the hardest of this family to block - interleaved left/right
  reflectors).

## Performance

All dense factorizations are level-3 blocked: compact-WY for QR and LQ, right-looking POTRF/GETRF for
Cholesky and LU, with trailing-matrix updates routed through the register-tiled GEMM (~70-90 GFLOP/s,
see [la-primitives.md](la-primitives.md)). Solve timings are in [solvers.md](solvers.md) and the
[README](../../README.md) benchmark table.
