# Decompositions — LU, Cholesky, QR/LQ, QRCP, Bidiag

Dense factorizations. Each has a zero-alloc `ref`-workspace overload plus an allocating convenience
wrapper (see [naming-style-guide](../naming-style-guide.md)'s workspace-overload pattern); several
route their allocating overload through a blocked (level-3, compact-WY/SYRK/GETRF-style) core above a
measured size crossover — see [level3-blocking-guide](../level3-blocking-guide.md) for how that's done
and why the gain caps out around 1.3–1.4× (bounded by `matMatDot`'s own ~70 GFLOP/s ceiling, see
[blas.md](blas.md)).

- **`LU.luDecomposition(ref U, ref L, ref Pivot P)`** — partial-pivoting LU, `PA = LU`. Blocked
  (GETRF-style: panel factor + TRSM + one GEMM trailing update) above `n ≥ 256`. Also
  `luDecompositionInPlace` (compact in-place form) and `LU.determinant(in LU, in Pivot)`.
- **`Cholesky.choleskyDecomposition(in A, ref L)`** — `A = LLᵀ`, SPD. Blocked (POTRF: panel + TRSM +
  SYRK trailing update) above `n ≥ 256`. **`choleskyDecompositionPivot(in A, ref L, ref Pivot P, ref
  ws)`** — rank-revealing (xPSTRF-style), upper-triangle-only working storage; unblocked by design.
- **`QR.qrDecomposition(ref Q, ref R[, ref u[, ref w]])`** — Householder QR. The fully-allocating
  overload is blocked (compact-WY) above `N_Cols ≥ 64`; the `ref u`/`ref w` zero-alloc overloads stay
  unblocked. **`QR.qrDecompositionColumnPivot(ref Q, ref R, ref Pivot P[, ref u])`** (QRCP,
  Businger-Golub) — exact (not downdated) column-norm recompute each step.
- **`LQ.lqDecomposition(ref A, ref L, ref Q[, ref ws])`** — direct row-Householder (GELQF-style, not
  transpose-to-QR). The allocating overload blocks above `m ≥ 512` (a measured, not derived,
  crossover — LQ's fold step is reduction-shaped, so the double-precision crossover trails float's).
- **`Bidiag.bidiagonalize(in A, ref U, ref B, ref V, ref ws)`** / **`bidiagonalizeValues(in A, ref d,
  ref e, ref ws)`** — Golub-Kahan-Householder reduction; feeds [svd.md](svd.md)'s `svdThin`/`svdValues`.
  Not yet raised to level-3 (tracked in [level3-blocking-guide](../level3-blocking-guide.md) as GEBRD,
  the hardest of this family to block — interleaved left/right reflectors).

## Benchmarks

Single-thread, this machine, float unless noted; each row is a cache-locality or blocking fix, cited
by commit. `float ≈ double` in a row means the fix was cache/ILP, not SIMD — a separate later commit
usually then vectorizes the same loop.

| Kernel | Size | Before → after | Source |
|---|---|---|---|
| LU trailing update → axpy | 1024² | 330.9 → 20.4ms (16.2×, 2.16→35.0 GFLOP/s) | `0afa17a` |
| Cholesky trailing dot → axpy (right-looking) | 1024² | 131.9 → 13.8ms (9.5× cumulative, →25.9 GFLOP/s) | `b9bbe24` |
| Cholesky, blocked POTRF | 1024² | 15.0 → 10.45ms (−30%, 34.3 GFLOP/s) | `d286658` |
| Pivoted Cholesky, upper-triangle storage | 1024² | 912 → 21.1ms (43×, 0.39→17.0 GFLOP/s) | `98519e3` |
| QR Householder apply, row-major reorder | 1024² | 5914 → 881ms (6.7×, cache-cliff removed) | `1548aee` |
| QR Householder apply → axpy (SIMD) | 1024² | 894 → 75.3ms (11.9×, 1.60→19.0 GFLOP/s) | `bee9c22` |
| QR, blocked compact-WY | 2048×1024 (tall, N=1024) | 188 → 141.6ms (1.33×) | `4f04e76` |
| LQ, blocked compact-WY | 1024×2048 (N=1024) | 244 → 149.8ms (−38.7%, ≈QR at flipped dims) | `cc087c9` |
| `QR.qrDirectSolve` apply → axpy | 1024² | 2684ms → 35.9ms (74.7×, 0.53→39.9 GFLOP/s) | `eadf6a8` |
| QRCP (column-pivoted) apply → axpy | 1024² | 5889ms → 95.5ms (61.6×, 0.24→15.0 GFLOP/s) | `eadf6a8` |

No standalone before/after benchmark is recorded for blocked LU (GETRF, `921bcc9`) or for Bidiag —
both are covered only indirectly via the algorithms that call them.
