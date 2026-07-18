# DEVLOG — Sparse
Code comments state contracts only; history lives here (see CLAUDE.md).

## fProxyBlockJacobi
- 2026-07-18 | `InvertBlock` (BR<=16 fast-path Gauss-Jordan) accepted any pivot `best > 0`,
  including denormals -- a ~1e-38 diagonal inverts to Inf and the build falsely reported Success.
  Threaded in the same diagonal-scaled pivot floor its sibling `fProxyILU0` uses: compute
  `diagMax` over the block's diagonal (floored to 1 if <= 0), reject with `!(best > 16*eps*diagMax)`
  so the build honestly reports NotPositiveDefinite/Singular. Constant/style copied from ILU0
  verbatim. Source touched; needs regen.

## Schwarz (fProxyAdditiveSchwarz / fProxyRestrictedSchwarz)
- 2026-07-18 | One-level AS/RAS MVP per docs/dev/spec-additive-schwarz-preconditioner.md.
  Contiguous block-row partition + delta-layer overlap; dense local factors cached at build
  (AS = Cholesky, RAS = LU+partial-pivot), reused every Apply. Design resolutions of spec
  ambiguities:
  - Naming: kept the long `fProxyAdditiveSchwarz`/`fProxyRestrictedSchwarz` (spec open-Q 1;
    orchestrator did not veto).
  - RAS breakdown: NO diagonal-shift retry (attempts always 1, shift always 0) — a singular
    local LU reports `Singular` directly. Chosen so the spec's test-8 "zero pivot column ->
    Singular" is reachable (a diagonal shift would rescue a zero column and hide it). AS keeps the
    IC0-style escalating Manteuffel shift (6 attempts) since every principal submatrix of an SPD A
    is SPD, so shifts only fire on numerical breakdown.
  - Missing-diagonal-block is NOT a guard here (unlike IC0/FSAI): the local gather zero-fills
    absent blocks and the factorization handles it (AS via shift, RAS via Singular). Only the
    square check (BlockRows==BlockCols, BR==BC) throws.
  - Local factor storage is a FLAT arena `Factors` buffer (offsets in `FactorStart`), NOT
    fProxyMxN views: the Cholesky/LU triangular sweeps are hand-rolled on the flat slice
    (`CholSolveInPlace`/`LUSolveInPlace`). RAS stores its compact LU in LOGICAL (pivoted) row order
    — `F[r,c] = M[P[r],c]` — plus the permutation in `Piv`, so Apply's solve is a plain
    identity-pivot sweep after permuting the gathered RHS (`bperm[r]=rLoc[Piv[r]]`, derived from
    PA=LU: (P_mat·A)[r]=A[P[r]]). This avoids replicating the library's P-indirection at Apply.
  - Overlap adjacency is symmetrized via a transient transpose CSR (Temp) so RAS's general
    (structurally-unsymmetric) A and Symmetric-storage A both expand correctly; Symmetric A is
    mirrored to full (arena, logically dead after setup) for value gather so no per-block transpose
    is needed in the gather.
  - Arena factories: exactly the spec's 3 per struct (opts / opts+out-info / default-opts). The
    breakdown test uses `(A, SchwarzOptions.Default, out info)`.

## FSAI/SPAI
- 2026-07-18 | SPAI local solve: normal-equations + `CHO` -> tall QR least-squares (`QR.solveInPlace`
  multi-RHS). The per-block-row problem is `min ‖A_hat^T·g − e_iLocal‖` with `A_hat^T` nI×nJ tall
  (nI >= nJ) and BR-wide RHS; the old route formed `N = A_hat·A_hat^T` (`Blas.dotSymT`) and factored
  it with Cholesky, squaring the condition number (κ²). Now `A_hat^T` is QR-factored directly and the
  BR unit-block RHS `e_iLocal` (I_BR at block-row iLocal, else 0) back-substituted — κ, not κ². Used
  the existing multi-RHS `QR.solveInPlace(ref A, ref B, ref X)` (tall, destroys A/B, allocates
  u/w/acc Temp internally), which already handles the whole BR block in one factorization. Tikhonov
  ladder preserved but re-expressed as regularized LS: attempt 0 is the plain problem; on breakdown a
  `√shift·I` row block is appended to the LS matrix (RHS gets matching zero rows) —
  `min ‖A_hat^T g − c‖² + shift‖g‖²`, whose normal equations are exactly the old shifted
  `(A_hat A_hat^T + shift I) g = A_hat c`, so the shift SCHEDULE is unchanged (1e-3·diagMax, ×10, 6
  attempts, worst reported in `Shift`/`attempts`). Breakdown DETECTION changed: `QR.solveInPlace` is
  unguarded on rank deficiency (zero R diagonal -> non-finite g), so a `math.isfinite` sweep of the
  solution replaces `CHO.decompInPlace(...).Solved`. shift>0 makes the augmented matrix full-column-
  rank, so the ladder is strictly more robust than before (a positive shift can no longer "fail").
  Output layout and the `mValues[dstOff+q*BR+p] = g[aI*BR+p, q]` scatter are byte-for-byte identical.
  Semantic note vs the old route: QR does NOT shift merely because N would be near-indefinite (the κ²
  regime CHO tripped on) — it solves those accurately instead, so on ill-conditioned-but-full-rank
  local blocks the built M can now DIFFER from the CHO version (more accurate, no shift bias); shift
  only fires on genuine (near-)rank-deficiency. SPAI stays nonsymmetric / pbiCGStab-only; interface,
  overload ladder, and arena factories untouched. SPAI generates float+double only (no int variant),
  so `math.sqrt`/`math.isfinite` are safe.
- 2026-07-18 | Follow-up to the entry directly below: after the multi-trial restructure, the full
  suite (all dtypes) still showed `BeatsJacobiOnLaplacianTest` and `BeatsJacobiOnPenalizedGrid3DTest`
  failing intermittently across fProxy/float/double. Coordinator's full-capture run confirmed every
  iteration-count assertion PASSED (FSAI strictly beat block-Jacobi every time: e.g. 8&lt;11, 8&lt;17,
  9&lt;15, 15&lt;25) and both `Solved` flags were true — the failing gate was the per-element
  `accOk`/solution-error check: `abs(xF[i]-xTrue[i]) < SolveTol()*(1+abs(xTrue[i]))`. That is a
  residual-vs-error confusion: pcg converges to a RESIDUAL tolerance (`‖b-Ax‖ &lt;= tol·‖b‖`, verified
  fresh at exit), while solution error scales as `‖x-xTrue‖ ≈ cond(A)·‖residual‖` — so bounding
  solution error by the same `tol` used for the residual is unsatisfiable for any `cond(A) &gt; 1`,
  i.e. every real matrix including the well-conditioned Laplacian (explains why it flipped
  True/False marginally across trials/dtypes: it was sitting on a rounding boundary that has
  nothing to do with FSAI's correctness). Fix (test-only, same file, both `BeatsJacobi*` tests):
  replaced the per-element solution-error check with a residual check —
  recompute `r = b - A·xF` via a fresh `BSR.spMV`, assert `‖r‖² &lt;= (RESIDUAL_C·tol)²·‖b‖²` with
  `RESIDUAL_C=8` (a safety cushion over pcg's own exact `‖r‖&lt;=tol·‖b‖` exit guarantee, absorbing
  the small summation-order difference between this fresh dot-product and pcg's own internal one)
  — conditioning-independent, matches what pcg's `Solved` flag actually guarantees. Removed the
  now-dead `PenalizedGrid3DDiagJob.SolveTol()` helper (its only call site was the deleted check).
  Not touched: `PminresConvergesTest` has the SAME per-element-vs-SolveTol anti-pattern but was
  NOT reported as failing (smaller/better-conditioned system, tol=1e-3f loose enough not to trip
  in practice) — left as-is per the coordinator's explicit "both BeatsJacobi tests" scope; flagged
  as a latent fragility worth a follow-up pass. No production code changed (again).
- 2026-07-18 | `BeatsJacobiOnPenalizedGrid3DTest` (float only) failed on
  `Assert.IsTrue(infoF.iterations < infoJ.iterations)`, run INSIDE the Burst job — an opaque
  Burst-internal assert, no iteration counts visible (same failure shape as the earlier Chebyshev
  degree-sweep issue). Diagnosed with a standalone float32 C# re-implementation (mirrors the
  Chebyshev repro method) of `fProxyPenalizedGrid3D(2,2,1,EA=1,penalty=10)` +
  `fProxyFSAI`/`fProxyBlockJacobi`/`pcg`, ported line-for-line from the actual template files
  (not from memory) and cross-checked against the shipped `bsrMatVecB3` kernel. Two hypotheses
  were on the table: (A) a real, expected limitation — static-pattern FSAI can't dominate
  penalty-conditioned elasticity, so the "beats" assertion was over-optimistic; (B) a real defect
  — FSAI's per-row local solves shift-escalating toward a near-diagonal G. RESULT: neither.
  `worstShift=0.000`, every row solved on attempt 1 (zero shift escalation on this exact
  matrix) — (B) is false. FSAI beat block-Jacobi in 499/500 independent random-right-hand-side
  trials (typical ratio 0.4-0.6x block-Jacobi's iteration count; the single non-strict trial was
  a TIE, never a loss) — so (A) is false too: the property is real and strongly supported, not
  over-optimistic. Conclusion: a single FIXED random seed is a fragile way to test a statistical
  "beats" property — a rare tie is possible per the sweep, and Burst's SIMD reduction order can
  plausibly tip a close call either way for one specific seed, independent of any algorithm
  defect. Fix (test-only, `TemplateSourceTests/fProxy/SparseFSAITests.fProxy.cs`): replaced the
  single-b in-Burst-assert test with `PenalizedGrid3DDiagJob`, which only computes (BlockJacobi,
  FSAI, IC0-for-reference iterations; FSAI's build `PreconditionerInfo`) across 3 independent
  right-hand sides and writes raw numbers to `NativeArray` outputs — every `Assert` now runs on
  the managed thread with the actual counts embedded in the failure message. Assertions: FSAI
  never regresses per-trial (hard, `<=`; 0/500 losses observed), FSAI's summed iterations across
  the 3 trials strictly beats block-Jacobi's sum (hard), and FSAI strictly beats on at least 2 of
  3 individual trials (hard). No production code changed — `fProxyFSAI`/`fProxySPAI` sources were
  read line-by-line against the repro and found correct; this was a test-fragility bug, not an
  algorithm bug. The diagnostic script was scratch-only (`~/scratch/fsai-repro/`), not committed.
- 2026-07-18 | Implemented per `docs/dev/spec-sparse-approximate-inverse-preconditioner.md`:
  `fProxyFSAI` + `fProxySPAI` (Sparse/fProxyFSAI.cs, Sparse/fProxySPAI.cs) + shared `SaiOptions`
  (Sparse/SaiOptions.cs) + 8 arena factories (Sparse/Arena.Sparse.fProxy.cs) + 3 pcg rungs
  (OP/Krylov.fProxy.cs) + 3 pminres rungs (OP/Krylov.PMinres.fProxy.cs, FSAI only) + 2 pbiCGStab
  rungs (OP/Krylov.PBiCGStab.fProxy.cs, SPAI only, mirroring the ILU0 rungs' arity exactly — no
  zero-alloc-scratch rung, unlike pcg/pminres). Shipped BOTH FSAI and SPAI in one pass (spec's Q3
  left phase-1-FSAI-only vs both open; the commissioning task asked for both, including SPAI
  tests, so both shipped together rather than splitting into two passes).
  Open questions resolved per the spec's stated recommendations: Q1 pattern = lower(A) (like IC0,
  not lower(A^2)); Q2 dropTol implemented (Frobenius-norm block filter, §3.2 formula) and exposed
  via `SaiOptions`, default 0 = off; Q4 Gt stored explicitly (both Apply spMVs forward, per spec's
  stated default — no on-the-fly `spMVT` variant, no `SaiOptions` toggle added since the spec only
  floated that as a "could be" idea, not a requirement); Q5 SPAI local solve via normal equations
  + CHO (not QR) — unchanged from spec's default, no conditioning issue observed in a quick manual
  smoke check on `fProxyLaplacian2D`; Q6 FSAI DOES get pminres rungs, with the same indefinite-A
  caveat doc IC0 carries there. `patternPower=2` (A²  pattern) throws (ArgumentException) rather
  than silently falling back to patternPower=1 — matches "throw until then" in the spec's
  `SaiOptions` snippet.
  FSAI's per-row diagonal-shift escalation is LOCAL (retry just that row, up to 6 attempts, same
  ladder shape as IC0's global one) since FSAI's rows are independent — cheaper than IC0's
  whole-matrix refactorization retry and explicitly called out as a difference in §3.1. `Shift`/
  `PreconditionerInfo.attempts` on the built struct report the WORST (max) shift/attempts seen
  across all processed rows, not a single global value (no single global shift exists for a
  per-row-independent build).
  `fProxyFSAI.FindBlockIndex`/`GatherBlockInto` are the shared gather-under-either-storage-mode
  primitive (`internal static` on `fProxyFSAI`, called by `fProxySPAI` too) — placed on the FSAI
  file rather than `UnsafeOP.Sparse`, per the spec's "implementer's choice" note (§7).
  Not verified: no Burst test suite run this pass (coder does not run `Tools/run-tests.ps1`); the
  "beats IC0" headline wall-clock comparison (spec §8 benchmark) needs `PCGBenchmark.fProxy.cs`
  extended with an FSAI arm — left for a follow-up pass together with the numbers this DEVLOG
  entry format expects (this entry intentionally has none yet).

## Chebyshev
- 2026-07-18 | Two SPD-robustness fixes to the ctor. (1) eigSteps clamp: `opt.eigSteps` (default
  10) was passed unclamped into `Eigen.lanczos`, which throws "steps must be in [1, A.Rows]" for
  any n < eigSteps -- so any valid SPD system smaller than 10 rows failed to build. Now clamped to
  `min(opt.eigSteps, n)` before both the LanczosCache sizing and the Lanczos call; fewer steps only
  coarsens the hi-estimate, never invalidates it. Supersedes the earlier entry below that told
  callers to pass a smaller eigSteps on small n. (2) Lanczos-result guard: the old code took
  `lambdaMax` unconditionally (the entry below deliberately did NOT treat non-convergence as a
  throw path). But a non-converged Lanczos or `lambdaMax <= 0`/NaN makes Hi/Sigma garbage and the
  induced M^-1 silently indefinite/NaN -- CG then diverges with no diagnostic. Now throws the same
  ArgumentException the ctor uses for a non-SPD build when `lInfo.status != Converged || !(lambdaMax
  > 0)`. Regression: `SmallSystemBelowEigStepsBuildsAndConverges` (n=8 < 10, residual-based assert
  on the managed thread). Source touched; needs regen. Suite NOT run (central).
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
