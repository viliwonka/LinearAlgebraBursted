# SVD

`SVD`, built on [`Bidiag`](decompositions.md)'s Golub-Kahan-Householder reduction. Every entry point
below has a zero-alloc workspace overload alongside the allocating one.

- **`SVD.thin(in A, ref U, ref S, ref V, ...)`** — the main entry point: full/thin SVD via
  bidiagonalization + implicit-shift bidiagonal QR (Golub-Reinsch). `A` (m≥n) unmodified, `U` is
  m×n with orthonormal columns, `S` descending, `V` is n×n.
- **`SVD.values(in A, ref S, ...)`** — values only, skips reconstructing `U`/`V` entirely — the
  cheapest route when you only need singular values (feeds `Analysis.cond`/`rank`/`matrixL2` — see
  [blas.md](blas.md)).
- **`SVD.truncated(in A, ref Uk, ref Sk, ref Vk, int k, int oversample, ...)`** — a true top-k
  Golub-Kahan-Lanczos reduction (not full-then-slice): builds only the requested `k` (+ oversample)
  singular triplets directly. `partialReorth` toggles full DGKS reorthogonalization (stable) vs. an
  ω-recurrence (fast). Fastest *exact* route for small/mid k.
- **`SVD.randomized(in A, ref Uk, ref Sk, ref Vk, int k, int oversample, int powerIters, ...)`** —
  Halko-Martinsson-Tropp: randomized range-finder + subspace iteration + a small exact SVD of the
  sketch. GEMM-dominated; wins over the exact routes when `k ≪ n` and `n` is large.
- **`pinvSolve`/`pseudoInverse`** — minimum-norm least-squares / Moore-Penrose pseudo-inverse via
  `SVD.thin`, any shape/rank (see [least-squares.md](least-squares.md)).
- **`lowRankApprox(in A, ref Ak, int k, ...)`** — best rank-k approximation (Eckart-Young), via full
  `SVD.thin` + slice (exact, not the truncated-GKL route).
- **`nullspaceBasis`/`rangeBasis(in A, ref basis, ...)`** — orthonormal nullspace/range basis from
  trailing/leading singular vectors.

Reuses the fast symmetric eigensolver rather than ever forming `AᵀA`: singular values are the
positive eigenvalues of the Jordan-Wielandt augmented matrix `[[0,A],[Aᵀ,0]]`, so accuracy tracks
`κ(A)`, not `κ(A)²`.

## Benchmarks

Single-thread, this machine, float, `Burst IJob.Run` median of 9 — each row is a vectorization fix,
not an algorithm change (same output, cited by commit):

| Method | Size | Before → after | Source |
|---|---|---|---|
| `SVD.thin` (Golub-Kahan bidiagonal QR rotations → contiguous rows) | 256² | 43.5 → 8.96ms (4.85×) | `92db7b4` |
| `SVD.values` (routed through Bidiag instead of the augmented-matrix eigensolve) | 256² | 21.7 → 3.78ms (5.7×) | `4ac19bd` |
| `SVD.truncated` (GKL basis transposed to rows + routed through vectorized GEMV) | 2048×256, k≈21% | 6.2–6.9× faster | `4203c72` |

`SVD.truncated` is now the fastest *exact* top-k method for small/mid k (beats both `SVD.thin` and
`SVD.randomized` by 3–4×); `SVD.randomized` only wins at high k% on large matrices (not independently
re-benchmarked after the `truncated` fix).

Current absolute numbers at a larger representative size, N=1024 square (`Benchmarks/EigenSvdBenchmark.cs`).
AMD Ryzen 9 9950X3D, single CCD pinned, 2026-07-05, commit `0714c97`, Unity Editor batchmode (checks likely on):

| Method | dtype | med(ms) |
|---|---|---|
| `SVD.thin` (full SVD) | float | 522.45 |
| `SVD.thin` | double | 735.77 |
| `SVD.values` (values only) | float | 188.44 |
| `SVD.values` | double | 233.35 |

`SVD.truncated` absolute numbers, tall 2048×256 (`Benchmarks/SvdComparisonBenchmark.cs`), same
machine/config, commit `95a1897` — for reference, `SVD.thin` (full, k=256) on the same matrix is
52.0ms float / 70.7ms double:

| k (of n=256) | float med(ms) | double med(ms) |
|---|---|---|
| 8 (3%) | 4.13 | 4.41 |
| 18 (7%) | 7.73 | 8.14 |
| 54 (21%) | 27.55 | 27.45 |
