# ML — k-means & PCA

`LinearAlgebra.ML`. Design docs: [spec-kmeans.md](../spec-kmeans.md), [spec-pca.md](../spec-pca.md).

## k-means

`KMeans.fit(in X, int k, uint seed, int maxIter, KMeansInit init, ref centroids, ref assignment,
out inertia, out iters, ref ws)` — Lloyd's algorithm with GEMM-accelerated assignment (expands
`‖x-c‖²` into `‖x‖² - 2x·c + ‖c‖²` and computes the cross term as one matrix multiply through
[`Blas`](blas.md)). Squared-Euclidean only; empty clusters reseed to the farthest point.
`KMeansInit{KMeansPlusPlus, Uniform}` (default k-means++). Allocating overload via `ref Arena`.

## PCA

Four fit routes, all `bool fit<Route>(in X, ref model, ...) : converged` (+ an allocating `ref Arena`
overload):

- **`PCA.fitCov`** — eigendecomposes the p×p covariance/correlation matrix. The only route that
  handles wide data (p > n); loses accuracy via κ² (covariance squares the conditioning).
- **`PCA.fitSvd`** — SVD of the centered data directly (needs n ≥ p); the accurate default.
- **`PCA.fitSvdTruncated`** — exact top-k via Golub-Kahan-Lanczos (`SVD.truncated`, see
  [svd.md](svd.md)); needs n ≥ p.
- **`PCA.fitRandomized`** — Halko-Martinsson-Tropp randomized SVD; needs n ≥ p, fastest for large n
  with k ≪ n.
- **`PCA.transform(in X, in model, ref scores)`** — projects new data onto a fitted model.
- `PCAScaling{Covariance, Correlation}` (default `Covariance` = center-only, no rescaling).

`floatPCAModel` — the library's first **buffer-carrying result struct**: `components` (p×k axes,
sign-fixed), `explainedVariance`/`explainedVarianceRatio` (length k, descending), `mean`/`scale`
(length p), `k`, `converged` (also exposed as `.Solved` and an implicit `bool` conversion). All
buffers are arena-owned — allocated via `arena.floatPCAModel(p, k)`, disposed with the arena, no
separate `Dispose()` call needed (same pattern as the `_Cache` workspace structs).

## Benchmarks

Not independently benchmarked for either feature. k-means had one algorithmic fix verified correct
but not re-measured with numbers: the final centroid-assignment sync (an O(N·D·k) GEMM) used to
re-run unconditionally even on the early-convergence exit path, where it's a guaranteed no-op — now
skipped (commit `9b72cba`). PCA's SVD-based routes inherit whatever's measured in
[svd.md](svd.md)'s `SVD.thin`/`SVD.truncated`/`SVD.randomized` tables.
