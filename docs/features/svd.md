# SVD

`SVD`, built on [`Bidiag`](decompositions.md)'s Golub-Kahan-Householder reduction. Every entry point
below has a zero-alloc workspace overload alongside the allocating one.

- **`SVD.thin(in A, ref U, ref S, ref V, ...)`** — the main entry point: full/thin SVD via
  bidiagonalization + implicit-shift bidiagonal QR (Golub-Reinsch). `A` (m≥n) unmodified, `U` is
  m×n with orthonormal columns, `S` descending, `V` is n×n.
- **`SVD.values(in A, ref S, ...)`** — values only, skips reconstructing `U`/`V` entirely — the
  cheapest route when you only need singular values (feeds `Analysis.cond`/`rank`/`matrixL2` — see
  [la-primitives.md](la-primitives.md)).
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

## Performance

`SVD.truncated` is the fastest *exact* top-k method for small/mid k, beating both `SVD.thin` and
`SVD.randomized` by 3–4×; `SVD.randomized` only wins at high k% on large matrices.

Ryzen 9 9950X3D, single-thread Burst, median of 9. Square N=1024:

| Method | dtype | med(ms) |
|---|---|---|
| `SVD.thin` (full SVD) | float | 505.80 |
| `SVD.thin` | double | 692.85 |
| `SVD.values` (values only) | float | 182.98 |
| `SVD.values` | double | 229.70 |

`SVD.truncated`, tall 2048×256 — for reference, `SVD.thin`
(full, k=256) on the same matrix is 51.7ms float / 70.5ms double:

| k (of n=256) | float med(ms) | double med(ms) |
|---|---|---|
| 8 (3%) | 4.12 | 4.30 |
| 18 (7%) | 7.68 | 7.94 |
| 54 (21%) | 27.41 | 26.78 |

`SVD.truncated` at a square 1024×1024, same k=54 as the 2048×256 row above (matched-k, different
shape):

| Size | k | float med(ms) | double med(ms) |
|---|---|---|---|
| 1024×1024 | 54 (5%) | 48.5 | 49.8 |

Three-way head-to-head on the same matrix, tall 2048×512, k=21 (~4%) — the low-k% regime where the
exact GKL route beats the randomized sketch:

| Method | float med(ms) | double med(ms) |
|---|---|---|
| `SVD.thin` (full, k=512) | 186.79 | 256.59 |
| `SVD.truncated` | 17.57 | 18.36 |
| `SVD.randomized` (oversample=10, powerIters=2) | 29.47 | 35.83 |

The default `maxIter` scales with the problem: `Consts.sweepBudget(n) = max(75, 6·n)`
(LAPACK dbdsqr's MAXITR=6 per-value heuristic with a small-n backstop). Tested across diverse
spectra (graded — e.g. σᵢ = 100·0.95^i, clustered, random) up to n=1024, convergence uses at
most about a quarter of that budget. On non-convergence the returned `SVDInfo`/`EigenInfo` reports
`MaxIterations` (and SVD-backed solvers report `DirectSolveStatus.NotConverged`) instead of
silently returning garbage.
