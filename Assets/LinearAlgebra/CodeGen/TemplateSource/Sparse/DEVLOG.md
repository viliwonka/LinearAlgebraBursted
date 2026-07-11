# DEVLOG — Sparse
Code comments state contracts only; history lives here (see CLAUDE.md).

## fProxyBSRBuilder.cs
- 2026-07-11 | Use-after-free bug fix: the arena's `fProxyBSRBuilder(...)` factory registers a
  VALUE COPY of the builder struct in its own tracking list so it can dispose it later.
  AddBlock/AddValue append to the triplet lists via UnsafeList.Add, which reallocates the backing
  buffer once the initial capacityHint is exceeded. When the three UnsafeLists were plain struct
  fields, that reallocation was only visible on the CALLER's copy of the builder -- the arena's
  tracked copy kept pointing at the freed pre-growth buffer, so arena.Dispose()/Clear() would
  double-free / use-after-free it. Reliably reproducible by adding more than capacityHint
  triplets (e.g. via AddValue) then disposing the arena. Fixed by moving the growable triplet
  state behind a single heap-allocated State* shared by every value-copy. A `NativeReference<State>`
  wrapper would have been an equivalent alternative; raw Malloc/Free was chosen to avoid pulling
  in a second allocation-owning collection type for one field, and keeps Dispose symmetric with
  the constructor's Malloc. (was fProxyBSRBuilder.cs:19-41)

## UnsafeOP.Sparse.fProxy.cs
- 2026-07-11 | Krylov R2/R8 accumulator-pairing verdict: R2 introduced a 2-accumulator even/odd
  pairing in the block-size-specialized bsrMatVec kernels (b=2/3/4/6; b=1 kept single-chain, A/B'd
  as a no-win exception) as an architectural judgment -- the BR=4 benchmark section was too
  machine-noisy at the time to attribute a clean win either way. R8 revisited it with a dedicated,
  repeated (3x) clean-room measurement (BR=4/1.5% fill and the b=1 stencil, both dtypes): pairing
  showed no reproducible win for b=4 -- every paired-vs-unpaired difference was smaller than the
  run-to-run swing measured on the identical kernel across repeats (up to ~10%), with no consistent
  direction for double and a shrinking-to-noise edge for float. REVERTED back to the single
  left-to-right accumulator fold for b=2/3/4/6, matching b=1's own already-settled finding.
  R8 also spiked software prefetch (Common.Prefetch on x[colInd[k+dist]] a few blocks ahead):
  consistently SLOWER, 8-56%, on every dtype/fill/pairing combination tried -- not shipped.
  Do not re-try either. (was UnsafeOP.Sparse.fProxy.cs:156-169)

## SparseOP.fProxy.cs / UnsafeOP.Sparse.fProxy.cs
- 2026-07-11 | ApplyDot/spMVDot fused-kernel regression: an earlier version of BSR.spMVDot
  dispatched genuinely-fused "Dot" kernels (bsrMatVecB1Dot..B6Dot) for full-storage square BSR,
  folding dot(x,y) into the same per-block-row pass that computes y. A/B'd at the b=1 stencil
  section of LargeSparseBenchmark against the compose form (spMV then Blas.dot(x,y)): CG at
  N=5120/float went from ~0.245ms (compose) to ~0.359ms with the fused B1Dot kernel -- a
  reproducible ~45% regression. Root cause: B1Dot's per-row arithmetic is trivial (the b=1 stencil
  is a tridiagonal, ~3 stored blocks per row), so the kernel's cost was dominated by its outer
  cross-row dot fold, which (lacking a contiguous 4-wide block to reinterpret as fProxy4) used two
  alternating scalar accumulators -- far slower than calling the already-tuned SIMD vecDot
  separately (2x fProxy4, 8 lane-chains). Reverted to compose for every case; the fused kernels
  were deleted, not merely unused. Do not re-try. (was SparseOP.fProxy.cs:154-177 and
  UnsafeOP.Sparse.fProxy.cs:1109-1117)
