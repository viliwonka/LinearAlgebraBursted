# ML 

Namespace `LinearAlgebra.ML`.

## KMeans

Accelerated Lloyd's algorithm. Squared-Euclidean only; empty clusters reseed to the farthest point.

`fit(in X, int k, uint seed, int maxIter, KMeansInit init, ref centroids, ref assignment,
out inertia, out iters, ref ws)`.

## PCA


Model struct is `PCAModel`, it carries buffers and results: 
- `components` (p×k axes,
sign-fixed), 
- `explainedVariance`/`explainedVarianceRatio` (length k, descending), 
- `mean`/`scale` (length p), 
- `k`, 
- `converged` (also exposed as `.Solved` and an implicit `bool` conversion). 

Buffers are allocated via `new floatPCAModel(p, k, Allocator.Temp)` (or `Persistent` + `Dispose()`
for a long-lived model).

Four fit routes, all `bool fit<Route>(in X, ref model, ...) : converged`.

PCA fit methods:
- **`PCA.fitCov`** - eigendecomposes the p×p covariance/correlation matrix. The only route that
  handles wide data (p > n); loses accuracy building the covariance matrix.
- **`PCA.fitSvd`** - SVD of the centered data directly (needs n ≥ p); the accurate default.
- **`PCA.fitSvdTruncated`** - exact top-k via Golub-Kahan-Lanczos (`SVD.truncated`, see
  [svd.md](svd.md)); needs n ≥ p.
- **`PCA.fitRandomized`** - Halko-Martinsson-Tropp randomized SVD; needs n ≥ p, fastest for large n
  with k ≪ n.

Transform new data into a fitted model:
- **`PCA.transform(in X, in model, ref scores)`** - projects new data onto a fitted model.



