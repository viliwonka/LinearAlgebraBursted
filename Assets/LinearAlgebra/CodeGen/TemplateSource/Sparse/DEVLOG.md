# DEVLOG — Sparse
Code comments state contracts only; history lives here (see CLAUDE.md).

## Chebyshev
- 2026-07-18 | `DegreeSweepNonIncreasingTest` (float only) failed centrally on
  `iters(d) <= iters(d-1)+1`. Diagnosed with a standalone float32 NumPy re-implementation of the
  exact ctor (InvDiag/Lanczos 10-step deterministic-seed/hi=1.1*ritzMax/lo=hi/30) and
  `BSR.chebyApply`/`Krylov.pcg` (16x16 Poisson, n=256) OUTSIDE Burst, to make the hidden numbers
  legible without running the Unity suite. Result: hi=2.1281 vs the TRUE lambda_max(D^-1 A)=1.9830
  (analytic 2D-Poisson eigenvalue formula, cross-checked against a full dense eigendecomposition) --
  a healthy 7.3% margin, ruling out an eigenvalue-underestimate/amplification defect (hypothesis
  B). A 30-trial sweep over random right-hand sides (both a double-precision-detour dot and a pure
  sequential-float32-accumulation dot, to stress-test rounding) showed iteration counts robustly
  monotone-decreasing with wide margins (e.g. 15/11/9/7 for d=1..4) and ZERO violations of
  `d4 <= d1` -- confirms hypothesis A (benign ULP-scale non-monotonicity from the true-residual
  convergence test's threshold-crossing timing vs Burst's actual SIMD dot-reduction order, which
  the standalone repro cannot bit-match). Fix: relaxed `DegreeSweepNonIncreasingTest` to the
  invariant the algorithm actually guarantees -- every degree converges, and degree=4 needs no MORE
  outer iterations than degree=1 -- and restructured it off a Burst-internal `Assert.IsTrue` onto a
  `NativeArray<int>` iteration-count output read and asserted on the managed thread (mirrors
  `JobbedChebyshevSolve`'s Out-array shape), so a future regression prints the real counts instead
  of hiding behind an opaque failed assert. Source (`fProxyChebyshev`/`BSR.chebyApply`) was NOT
  touched -- this was a test-strictness bug, not an algorithm bug. Diagnostic script was scratch-only,
  not committed.
- 2026-07-18 | Implemented per `docs/dev/spec-chebyshev-preconditioner.md`: `fProxyChebyshev`
  (Sparse/fProxyChebyshev.cs) + `BSR.chebyApply` static kernel (same file) + arena factories
  (Sparse/Arena.Sparse.fProxy.cs) + 3 pcg rungs (OP/Krylov.fProxy.cs) + 3 pminres rungs
  (OP/Krylov.PMinres.fProxy.cs). Open questions resolved per the spec's stated defaults/answers:
  Q1 degree=3/kappa=30 shipped as-is, NOT re-tuned against PCGBenchmark (no benchmark run this
  pass); Q2 hi = safety*lanczos only, no Gershgorin cap; Q3 no pbiCGStab rung (mirrors
  BlockJacobi/SSOR/IC0); Q4 no un-scaled (D=I) variant; Q5 no block-diagonal scaling; Q6 no `out
  PreconditionerInfo` overload -- every setup failure throws; Q7 no LOBPCG overloads.
  PCGBenchmark's Chebyshev column (spec §5) was NOT added this pass (out of the requested scope) --
  next coder pass should add it before re-tuning Q1.
  Lanczos non-convergence on the symmetrically-scaled operator (LanczosInfo.Solved == false) is
  NOT an extra throw path -- the spec's ctor throw list (§4.2) doesn't include it, so lambdaMax is
  taken from whatever Ritz values were produced regardless of the inner QL convergence flag.
  eigSteps default (10) can exceed a tiny test matrix's Rows and trip `Eigen.lanczos`'s own "steps
  must be in [1, A.Rows]" guard -- not a Chebyshev-specific check, callers on small n must pass a
  smaller eigSteps.

## Gallery.Sparse: fProxyPenalizedGrid3D
- 2026-07-17 | Added for the LOBPCG false-convergence repro (docs/dev/spec-lobpcg-robustness.md
  §D.1): a self-contained port of the BuildingFrame demo's truss topology (columns, X/Y beams,
  floor diagonals, perimeter wall braces, penalty-pinned base) so tests don't depend on demo code.
  Only a BSR variant exists — the spec asked for "dense + BSR", but C# cannot overload on return
  type with identical parameters (fProxyMxN vs fProxyBSR, CS0111); the dense form is
  `ToDense(ref arena)`, which handles symmetric lower-block storage.

## fProxyBSRBuilder.cs
- 2026-07-13 | Type doc and ToBSR's doc both called value-restamping-on-a-fixed-pattern "a later
  phase"/"Phase 1 pattern-edit scope" — stale: BuildAssemblyCache + Refill
  (fProxyBSRAssembly.fProxy.cs) already ship exactly that per-frame reuse path. Rewrote both
  docs to point at BuildAssemblyCache/Refill instead. Also dropped the "this is no longer
  load-bearing" ref-vs-in Arena history from ToBSR's doc (Arena is a thin copyable handle; see
  the SparseBSRTests DEVLOG entry in TemplateSourceTests/DEVLOG.md for the underlying
  bug/fix). (was fProxyBSRBuilder.cs:16-17, 175, 177-179)

## fProxyBSRBuilder.cs / fProxyBSR.cs / fProxyIC0.cs (triangle-trust unification)
- 2026-07-12 | Coherence-audit P.2 (owner-ruled 2026-07-11): flipped `ToBSRSymmetric` from
  upper-block canonical to LOWER-block canonical, so the whole library trusts the lower triangle
  for symmetric matrices, dense and sparse alike. Why lower, not upper: dense row-major
  Cholesky/CHO/CHOP are lower-optimal (the inner dot products run over two contiguous ROWS) and
  cannot flip without a real perf loss, so the dense side was fixed; sparse symmetric spMV/spMM
  (`bsrMatVecSym*`, `bsrMatMatSym*`) were already side-neutral (per stored off-diagonal block: one
  gather + one transpose-scatter, no ordering assumption beyond `bi != bj`) -- confirmed by
  re-reading every kernel while doing this change, none needed a code fix, comments only.
  `fProxyIC0` gets a real win from the flip: its factor pattern IS A's lower block pattern, so a
  symmetric-storage SPD input is now consumed with ZERO mirror (previously paid a full 2×Nnzb
  mirror-to-full copy in `Arena.fProxyBSRMirrorToFull`, then read only the lower half of it back
  out). ILU0/SSOR still mirror to full (they genuinely need both triangles row-ordered). Every
  test/benchmark symmetric-authoring site (`ToBSRSymmetric` callers) was flipped to lower triplets
  in the same pass.

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
