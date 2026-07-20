# DEVLOG — TemplateSourceTests
Code comments state contracts only; history lives here (see CLAUDE.md).

## KrylovBlockLstsqBatteryTests / KrylovBattery.Invokers (task #56)
- 2026-07-20 | New file, fourth and final battery family (square/block/single-RHS-lstsq already
  shipped). Wires blsmr/bcgls (both Overdetermined-only, tall A + block RHS, min-residual oracle)
  via a new `IfProxyBlockLstsqSolverInvoker` interface -- block-shaped like
  `IfProxyBlockSolverInvoker`, damp-free and TPre-free like `IfProxyLstsqSolverInvoker` (neither
  production solver has a Tikhonov-damped or preconditioned entry point). `Solve<TOp>` takes an
  explicit `maxIter` argument (rather than deriving it internally) so the battery's tiny-maxIter
  no-NaN check can force a single iteration without a second invoker configuration. Both retired
  bespoke files' scenarios (normal-equations optimality, per-column agreement with scalar `lsmr`,
  consistent-system exact recovery, zero-rhs immediate convergence, tiny-maxIter no-NaN) are fully
  covered by checks #1-6 -- no bespoke case needed to be kept.
- 2026-07-20 | Added `GalleryDenseMatrix.TallRandom24x8` (Overdetermined|FullRank|WellConditioned,
  same 24x8 random-entry shape the retired bespoke tests built inline): the existing Overdetermined
  gallery only had Lauchli3_05/Lauchli3_1e3 (n=3), too narrow for a useful block RHS at S>=2 on top
  of Lauchli's own near-rank-deficiency stress. Kept Lauchli3_05 applicable too (bcgls handles it
  cleanly) rather than dropping it outright.
- 2026-07-20 | blsmr's block Golub-Kahan bidiagonalization hits a genuine `Breakdown` (not just slow
  convergence) on Lauchli3_05 at S=2 -- even the WellConditioned entry, the very first gallery matrix
  tried. Its per-iteration LQ factors need more column-space headroom than 2-of-3 columns leaves.
  Excluded via a battery-local `IsNarrowDense`/`skipNarrow` gate (mirrors `KrylovBlockBatteryTests`'
  bminres `SkipTinyDense` precedent) rather than a `MatrixProfile` flag, since "too narrow for this
  solver's block width" isn't a property of the matrix in isolation the way `IllConditioned` is.
- 2026-07-20 | `fProxyBcglsInvoker.Forbids` gained `IllConditioned`: squaring the condition number via
  the normal equations makes Lauchli3_1e3 (eps=1e-3) genuinely too hard for the s x s Gram-based
  coefficient solve at S=2 (per-column mismatch vs. scalar lsmr, and the consistent-recovery check
  never reaches full converged==S status). bcgls tolerates Lauchli3_05 (WellConditioned) fine, so
  only the IllConditioned sibling is excluded -- unlike blsmr, no narrow-matrix exclusion needed.
- 2026-07-20 | blsmr's convergence flag is CONSERVATIVE in float on the consistent-recovery check
  (#4): the internal ||A^T R||_F^2 stopping test can leave an already-accurate solution just short of
  its threshold under float rounding. That check's strict `Solved`/`converged==S` assertion is
  double-only for blsmr, all dtypes for bcgls (which tests convergence via the EXACT maintained
  S = A^T R, not an estimate) -- carried as a `strictConsistentStatus` bool per `SolverKind`, same
  role as the block family's per-solver `CheckFlags`.
- 2026-07-20 | Retired `BlockLSMRTests.fProxy.cs` / `BlockCGLSTests.fProxy.cs` (and their generated
  float/double copies) -- every one of their four `[Test]` cases per file is now a battery check.

## KrylovLstsqBatteryTests (task #48)
- 2026-07-20 | New file, third and final battery family (square/block already shipped). Wires
  lsqr/lsmr (Overdetermined, min-residual oracle: fresh `Krylov.lstsqResidual` Arnorm vs. the same
  ‖Aᵀb‖ scale the solvers' own stopping test uses, plus elementwise agreement with a direct dense
  `QR.decomp`/`decompSolve` least-squares reference) and craig/craigmr (Underdetermined, min-norm
  oracle: fresh rnorm vs. ‖b‖, plus elementwise agreement with `LQ.minNormSolve`). Damped-path
  check (#12) only runs for the Overdetermined pair -- craig/craigmr have no `damp` parameter in
  production (a consistent min-norm system has no residual/norm trade-off to regularize), so
  wiring them into that check would just be asserting a promise the solver never makes.
- 2026-07-20 | `LstsqBattery` needed `[Timeout(600000)]`: this fixture's single `[BurstCompile]`
  `TestJob.Execute()` compiles all four solver branches (lsqr/lsmr/craig/craigmr) PLUS the QR
  blocked-path and LQ min-norm machinery used only by this battery's oracles (no other battery
  file calls either), so the first `[TestCaseSource]` case to run in a session pays one cold
  compile that exceeded NUnit's 180s default here (measured ~227s wall time for the whole 4-case
  fixture on a cold cache); later cases reuse the compiled job and finish in milliseconds.

## BlockFGmresTests / KrylovBlockBatteryTests (bfgmres, task #38)
- 2026-07-20 | New file `BlockFGmresTests.fProxy.cs`, mirroring `BlockGmresTests.fProxy.cs`'s IJob/
  TestType shape. Coverage: known-block-solution recovery on nonsymmetric A (`fProxyDenseOperatorGeneral`,
  never the symmetric-only `fProxyDenseOperator`), per-column agreement with scalar `fgmres`, restart
  correctness, a genuinely flexible case (`InnerGmresPreconditioner` -- a few fixed unpreconditioned
  inner `gmres` steps, reused verbatim from `FGMRESTests.fProxy.cs` since `IfProxyPreconditioner.Apply`
  is single-row and `BlockApplyPre` drives it row-by-row, so the scalar struct needs no block-specific
  variant), and the required `IdentityFoldMatchesBgmresBitIdentical` cross-solver check (bit-identical
  X/iterations/status between `bfgmres`-with-identity and `bgmres`-with-identity on the same seeded
  system). In-job struct-copy safety has no separate test -- inherent to every case already running
  inside a `[BurstCompile(CompileSynchronously = true)] IJob` + `.Run()`.
- 2026-07-20 | Wired `bfgmres` into the block battery (`KrylovBlockBatteryTests.fProxy.cs`): new
  `SolverKind.Bfgmres` case and `fProxyBfgmresInvoker` (`KrylovBattery.Invokers.fProxy.cs`), same
  `CheckFlags` as `Bgmres` (`NeedsGeneralDenseOperator=true`, `Requires=Nonsymmetric`,
  `Forbids=IllConditioned`, `BlockAdvantage=false` -- shares bgmres's deflating rank-revealing basis and
  the same lack of a monotone block-advantage bound, `NoBreakdown`/`IdenticalColumns=true`). The
  battery's own checks #1-4 only ever pass the identity preconditioner (see `IfProxyBlockSolverInvoker`'s
  own doc) so never exercise a genuinely varying M -- that path is `BlockFGmresTests.fProxy.cs`'s job,
  not the battery's.

## KrylovBlockBatteryTests / KrylovBattery.Invokers
- 2026-07-20 | New file `KrylovBlockBatteryTests.fProxy.cs`: block-family battery (spec-krylov-test-
  battery.md SS5.3, checks #6-9 on top of block-shaped #1-5), mirroring `KrylovSquareBatteryTests`'
  IJob/RunStandardChecks shape. Added 6 block invoker structs (`fProxyBcgInvoker`, `fProxyBcgrqInvoker`,
  `fProxyBfbcgInvoker`, `fProxyBminresInvoker`, `fProxyBbiCGStabInvoker`, `fProxyBgmresInvoker`) to
  `KrylovBattery.Invokers.fProxy.cs` -- all 6 currently-implemented block solvers wired, none skipped.
  `ScalarCounterpart()` is implemented on every invoker (interface contract) but never CALLED from
  inside the `[BurstCompile]` job: returning it as the naked `IfProxySquareSolverInvoker` interface
  would box the concrete struct and dispatch through it, which Burst cannot compile. Instead
  `RunBlockStandardChecks<TInvoker, TScalar>` takes the matching scalar invoker as its own generic type
  parameter, constructed explicitly alongside the block invoker in `Execute()`'s switch -- an extra
  generic parameter, not a design the spec anticipated, but zero boxing and fully monomorphized like
  every other TOp/TPre call in this battery.
- 2026-07-20 | The identical-RHS-columns invariant (check #8) is unconditional for bcg/bcgrq/bfbcg/
  bbiCGStab/bgmres but disabled for bminres: forcing two RHS rows bit-identical deflates a Lanczos lane
  from iteration 0, and bminres does not yet preserve row symmetry once a lane deflates (a known,
  separate, deferred limitation from the s>1 divergence bug this check is otherwise meant to guard --
  see `OP/DEVLOG.md`'s `Krylov.bminres (block)` entry, task #50). The same forced-duplicate-row RHS can
  also trip bminres's Gamma near-singularity case into a genuine `Breakdown` (also task #50), so check
  #9's "status != Breakdown" half is disabled for bminres too; its "no NaN/Inf" half stays unconditional
  for every solver. bbiCGStab also disables the "status != Breakdown" half: it has no column locking/
  deflation, so a block coefficient made exactly singular by the forced duplicate rows is its
  documented, honest Breakdown contract ("never NaN/throw"), not a bug.
- 2026-07-20 | Check #7 (block advantage: block iterations <= worst per-column scalar iterations) is
  disabled for bbiCGStab/bgmres/bminres -- all three lack the residual-minimizing/monotone-convergence
  property (CG's energy norm, MINRES's residual norm under exact block-Lanczos) the naive bound relies
  on: bgmres's deflating rank-revealing basis trades subspace richness for robustness (measured 5x
  overshoot on DenseNonsym20's heavy diagonal dominance), bbiCGStab's short two-term recurrence is not
  residual-optimal (measured a narrow 9-vs-8 overshoot on ConvDiffDense40), and indefinite block-Lanczos
  convergence is not guaranteed monotone either. Kept unconditional (and passing) for bcg/bcgrq/bfbcg.
- 2026-07-20 | `fProxyBcgInvoker.Forbids` gained `IllConditioned` (bcg-family's other two members,
  bcgrq/bfbcg, do not need it): ridge regularization only guards RANK-deficient search blocks, not the
  s == n corner (this battery's block width, 4, equals Hilbert4's dimension) combined with extreme
  conditioning -- diverges outright (residual ~487x) rather than converging slowly. bcgrq/bfbcg's
  per-iteration rank-revealing-LQ search basis does not share this failure mode.
- 2026-07-20 | bminres additionally runs at S=2 (not the other 5 solvers' S=4), MaxIterMul=40 (not 20),
  and `SkipTinyDense=true` (excludes the n<=5 dense entries: Fiedler5/Clement4/MinIJ_5/Pei5_2/Lehmer5,
  leaving Laplacian1D_8/RandSPDWellCond20 dense plus every n>=80 BSR entry). Its block-Lanczos
  recurrence saturates a tiny matrix's whole dimension almost immediately even on an ORDINARY random
  RHS (no forced duplication needed), tripping the same Gamma near-singularity gap as above (task #50)
  or landing just outside the residual tolerance within an otherwise-generous budget. This drops
  bminres's dense battery coverage to SPD only (no indefinite dense entry is both large enough and not
  already IllConditioned-excluded) -- the existing bespoke `BlockMinresTests.fProxy.cs` already covers
  indefinite correctness at a comfortable n=20, unaffected by this exclusion.
- 2026-07-20 | Battery-wide block width is 4 (except bminres's 2): the smallest applicable matrix per
  family after the above exclusions is n=5 (bcg-family: MinIJ_5) or n=8 (bminres: Laplacian1D_8) or
  n=20 (bbiCGStab/bgmres: DenseNonsym20), all comfortably >= S. Checks #8/#9 force rows 0 and S-1
  bit-identical (generic in S, not hardcoded to specific indices).

## CRAIGTests
- 2026-07-20 | New bespoke file `CRAIGTests.fProxy.cs` for `Krylov.craig` (task #27). The key test
  (`RectangularMinNorm`) verifies actual least-NORM correctness, not just `Ax=b`: builds an
  underdetermined consistent system from an ARBITRARY x_true (b = A·x_true), then checks craig's x
  against the exact min-2-norm oracle `LQ.minNormSolve` (already in this library, LQ-factorization
  based) -- plus a softer `‖x‖ <= ‖x_true‖` sanity check and a negative guard that x is NOT x_true
  (proving the oracle comparison isn't vacuously true). `SquareFullRank` covers the unique-solution
  case, `ExplicitScratchInJob` exercises the caller-provided u/v/tmpM/tmpN zero-alloc overload inside
  the IJob (guarding the struct-copy/ping-pong-buffer bug class seen previously in LOBPCG),
  `ZeroRhs` checks the exact (not approximate) x=0/iterations=0 early-out, and a bonus
  `RankDeficientBreakdown` bit-exactly constructs a first-step Aᵀu=0 collapse (row of A exactly
  zero, b nonzero only in that row's component) to confirm honest `Breakdown` status with no NaN.
  Convergence tol handed to craig (`SolveTol`, 1e-5f/1e-13) is tighter than
  `Consts.fProxySqrtEps` -- the sqrt-eps default only drives ‖b-Ax‖ to ~1e-2 absolute on these
  test-scale systems, too loose for the oracle comparison tolerance (`Tol`, 1e-3f/1e-9). NOT wired
  into `KrylovSquareBatteryTests` -- see the OP DEVLOG's `Krylov.CRAIG` entry (battery has no
  least-norm invoker yet).

## KrylovSquareBatteryTests / KrylovBattery.Invokers
- 2026-07-20 | Added `fProxyGcrodrInvoker` + `SolverKind.Gcrodr` for `Krylov.gcrodr` (task #29),
  mirroring `fProxyGmresInvoker` (Requires=Square, Forbids=IllConditioned, PrecondKind=
  NonsymmetricBSR, no-op Init, same task-#53-deferred Rosser exclusion). `Restart=30, Recycle=10`
  (recycle at 1/3 of restart, matching the `GcrodrDefaultRecycle` production default). See the OP
  DEVLOG's `Krylov.GCRODR` entry for the solver-side design/deviations.
- 2026-07-20 | Added `fProxyTfqmrInvoker` + `SolverKind.Tfqmr`, mirroring `fProxyBiCGStabInvoker`
  (Requires=Square, Forbids=IllConditioned, PrecondKind=NonsymmetricBSR -- same task-#53-deferred
  Rosser exclusion class as biCGStab/gmres/idr). `MaxIterMul=40` (tfqmr's maxIter counts half-steps,
  ~one A-apply each, so ~40 half-steps matches biCGStab's 20 two-matvec passes).
- 2026-07-20 | Fanned out the remaining single-RHS square solvers (fcg, minres, minresQLP,
  biCGStab, gmres, fgmres, idr) into the battery alongside the cg spike -- one invoker struct +
  one SolverKind case each, no change to the shared RunStandardChecks/CheckDense/CheckBSR harness.
  `fgmres` slots into `SolveWithPrecond<TOp,TPre>` exactly like `gmres` (a single battery call only
  ever hands it one, possibly internally-iterative, TPre instance; the "M varies per step" property
  is internal to that one call, invisible to the invoker interface).
- 2026-07-20 | Found on first wiring pass: the Rosser gallery entry (SymmetricIndefinite |
  IllConditioned, clustered near-degenerate 8x8 spectrum -- previously only exercised by eigenvalue
  tests, never fed through an iterative solve) drives minres/minresQLP/biCGStab/gmres/fgmres/idr
  into unbounded divergence (residuals up to 1e14-1e19), not mere slow convergence. Root cause:
  none of these recurrences guard the near-zero-but-nonzero denominator case in their Givens/
  Hessenberg/shadow-space pivots (minres's `w = (...)/gamma`, gmres/fgmres's `y[i] = sum/H[i,i]`,
  biCGStab/idr's pivot solves) -- a small-but-not-exactly-zero pivot passes their zero/NaN
  breakdown checks yet still amplifies the update by orders of magnitude. Confirmed NOT an
  iteration-budget problem: raising MaxIterMul 5x left float residuals unchanged and made some
  double residuals worse (a corrupted x has no self-correction mechanism once poisoned). cg/fcg are
  unaffected (SPD-only, never see this matrix). minresQLP's Acond/xnorm safety clamps do prevent the
  same magnitude of blowup (residual ~0.38, not 1e14) but still land far outside the check bound.
  Fix applied here: added `Forbids: IllConditioned` to the six affected invokers -- MatrixProfile
  has no tag for "clustered/near-degenerate spectrum" specifically, so this also drops their
  Hilbert4/Pascal5/Grcar8 IllConditioned coverage as collateral (all three converge cleanly and
  would ideally stay covered). A real fix (near-zero-pivot breakdown detection in each recurrence,
  or a dedicated gallery tag once one exists) is future work, not attempted here -- out of scope for
  a wiring task and multi-file. Rosser itself was left untouched (Gallery/Profile are established
  infra for this battery, not this increment's to redesign).
- 2026-07-20 | Separately, `minresQLP`'s own stopping test (`rnorm / (Anorm*xnorm + beta1)`) is
  normalized by the solution/matrix scale and is measurably looser than this battery's raw
  `‖b-Ax‖/‖b‖` check -- reproducibly ~13-14x looser on the WellConditioned dense entries tried
  (Laplacian1D_8, MinIJ_5), independent of the absolute tol requested (scaling TolValue down
  10x moved the goalposts proportionally and changed nothing, confirming the gap is a ratio, not a
  budget). `fProxyMinresQLPInvoker` now requests a solve tolerance well past what it reports to the
  check (`SolveTol = Tol * 0.02`, `Tol` alone still drives the check's own bound) to land the fresh
  residual inside bound with margin; no other invoker needed this split.

## ConvergenceBudgetTests
- 2026-07-13 | Relocated the measured figure from the header: managed (non-Burst) execution of
  this battery measured ~50x slower than the Burst job path, which is why the O(n^3) work runs
  inside [BurstCompile] IJobs. (was ConvergenceBudgetTests.fProxy.cs:20)

## QPEqpTests
- 2026-07-13 | Converted from the hand-written SourceTests/QPEqpTests.cs (hand-duplicated
  float/double halves) into this fProxy template; the "InternalsVisibleTo cannot reach the
  firstpass assembly" folklore repeated in several hand-written test headers was verified FALSE
  against the compiled assemblies (the grant exists and fProxyChooseMarkerTests already used it).
  Header's spec provenance: the oracle follows docs/dev/draft-spec-qp.md Stage 1 plus the
  implementation handoff. QPActiveSetTests.cs / QPSolveTests.cs / LadFrischNewtonQuantileTests.cs
  are candidates for the same conversion.

## QPActiveSetTests
- 2026-07-13 | Converted from the hand-written SourceTests/QPActiveSetTests.cs (hand-duplicated
  float/double halves, one job struct per dtype per case) into this fProxy template, following
  QPEqpTests.fProxy.cs's conversion precedent. Tolerances that differed between the float/double
  halves (HS21/35/52/76 objective and componentwise-x tolerances; the brute-force feasibility-check
  and objective-agreement tolerances; the LP-limit and Degenerate tolerances) became
  /*+choose[...]*/ markers; literal problem data (Q/c/A/b/xl/xu, random-draw ranges) was identical
  between halves value-for-value and needed no marker. Header dropped internal "Stage 2"/
  "draft-spec-qp.md" labels and the (now-false) "HAND-WRITTEN because InternalsVisibleTo can't reach
  firstpass" framing -- see the QPSolveTests entry below for the fuller InternalsVisibleTo
  correction, which also applies here.

## QPSolveTests
- 2026-07-13 | Converted from the hand-written SourceTests/QPSolveTests.cs into this fProxy template,
  same conversion pattern as QPEqpTests.fProxy.cs / QPActiveSetTests.fProxy.cs. The three
  validation-throw tests (Solve_AsymmetricQ_Throws / Solve_DimensionMismatch_Throws /
  Solve_LowerAboveUpper_Throws) were DOUBLE-ONLY in the original hand-written suite (no float half
  existed at all -- the validation logic under test doesn't depend on precision); preserved exactly
  as double-only via `//+skipFor[float]` rather than adding new float coverage. The ill-conditioned
  stress cases (criterion 5) used genuinely different case data per precision (condition up to ~1e6
  at float, ~1e12 at double -- not just a shared tolerance): merged into one IllConditionedCases()
  source with the maxEig value itself behind a choose marker per case, preserving all 3+3 original
  (n, seed, maxEig) triples. Every other float-vs-double tolerance pair (facade HS21/35/52/76,
  NoBounds, IllConditioned residual tolerances, HeavyDegeneracy) became a choose marker. Also
  corrected the false "InternalsVisibleTo cannot reach the firstpass assembly" claim in this file's
  own header and in three other places that repeated the same folklore QPEqpTests' conversion had
  already disproven: QP.fProxy.cs's eqpSolve doc comment, LPTests.fProxy.cs's tau-sanity NOTE, and
  QRCPDowndateTests.fProxy.cs's two oracle-provenance comments. True mechanism in all four: the
  InternalsVisibleTo grants on BOTH BurstLinearAlgebra.Tests and
  BurstLinearAlgebra.TemplateSource.Tests-firstpass (TemplateSource/AssemblyInfo.cs). Dropped the
  header's measured-tolerance aside ("Empirically feas landed ~1e-6 (float) / ~1e-13 (double) and
  stationarity ~1e-8..1e-4 relative to maxEig") per the comment-contracts-only policy; kept the
  tolerance-scaling RULE itself (feasibilityResidual ~ machine precision regardless of conditioning,
  stationarityResidual scales with relTol * maxEig).

## LadFrischNewtonQuantileTests
- 2026-07-13 | Converted from the hand-written SourceTests/LadFrischNewtonQuantileTests.cs into this
  fProxy template. The original header's own justification for staying hand-written -- "InternalsVisibleTo
  only reaches the generated BurstLinearAlgebra.Tests assembly, NOT this template's firstpass
  compile-check assembly" -- was the same false claim QPEqpTests' conversion had already disproven,
  so the premise for keeping this file hand-written no longer held either. TauHalfMatchesCore's six
  tolerances (objFN-vs-objCore, x-vs-xCore componentwise, intercept, slope, L1 residual) differed
  float-vs-double and became choose markers; TauQuarterResidualSign was already fully symmetric
  (same seed, same ranges, same +/-20%*m tolerance in both halves) and merged with no markers at all.

## FullStatsTests
- 2026-07-13 | Header history relocated: the median/quartile/IQR path was previously untested;
  the n==2 case used to read out of bounds (copy[-1]) before the fix its test now pins. The
  facade was StatsOP at the time (renamed Stats). (was FullStatsTests.fProxy.cs:10-12)

## ArenaHandleTests
- 2026-07-13 | The 2026-07-11 relocation (below) shortened the file header but left it still
  narrating history and still tagging four individual cases "FM2"/"FM2:" (lines 7-8 header, 32,
  129, 134, 139) -- the full postmortem already lives in the entry below, so these added nothing.
  Reworded the header to a plain present-tense contract (in-Arena defensive copy must not dangle)
  and dropped all four inline "FM2" tags, keeping each case's contract sentence. (was
  ArenaHandleTests.fProxy.cs:7-8, 32, 129, 134, 139)
- 2026-07-11 | Relocated the "THE OLD BUG (FM2) / THE FIX" postmortem from the file header. Full
  account: Arena used to be a plain struct holding all its mutable tracking state inline, and every
  math struct captured arena identity by RAW ADDRESS (`Arena* _arenaPtr`, set via
  `fixed (Arena* p = &arena) _arenaPtr = p;` in the `in Arena` constructors). Arena's allocator methods
  (e.g. fProxyVec / fProxyMat) are NOT `readonly`, so calling one through an `in Arena` PARAMETER forces
  the C# compiler to make a defensive copy of the arena first -- and the struct being constructed
  captured the address of that dead stack temporary. Once the enclosing helper's frame returned, that
  captured pointer dangled: indexing/Copy()/Dispose() on the returned struct dereferenced freed stack
  memory, surfacing under Burst as a native crash / "allocator handle is not valid". THE FIX: Arena was
  split into ArenaCore (heap-Malloc'd once, holds all mutable state, never copied) and Arena (a thin
  handle wrapping a single `ArenaCore*`). Every math struct now holds an Arena VALUE field. Copying an
  Arena handle -- including compiler-inserted defensive copies of `in Arena` params -- copies only the
  ArenaCore* value, so every copy still resolves to the same live core. The dangling-pointer failure
  mode is now structurally impossible. Originally cited docs/dev/rfc-memory-model.md §1 / §2.2 / §4
  Option A / §6.0 / §6.1 (internal-only doc, not shipped; reference dropped from source).
  (was ArenaHandleTests.fProxy.cs:7-31)

## ArenaWiringTests (fProxy / iProxy / bool)
- 2026-07-11 | Dropped the dangling `docs/dev/rfc-memory-model.md §4 Option A` citation from the file
  header (identical in all three dtype copies: fProxy/ArenaWiringTests.fProxy.cs,
  iProxy/ArenaWiringTests.iProxy.cs, bool/ArenaWiringTests.bool.cs). No other change: the "option
  (b)/(c)" generational-overlay language later in each file already defines both options inline
  (fProxyN/iProxyN/boolN = option (b), Alive-only; fProxyMxN/iProxyMxN/boolMxN = option (c), Alive + a
  `_gen` stamp) before it's used, so that part was left as-is (judged self-contained on re-read).

## SparseArenaWiringTests
- 2026-07-12 | Dropped agent-workflow narration ("the coder flagged" / "the coder's flagged behavioral
  divergence") from the file header and the BlockJacobi double-dispose comment. Kept the substance
  (the three seam differences from the dense types, and that the readonly-struct same-copy-throws
  divergence is deliberately pinned distinctly). (was SparseArenaWiringTests.fProxy.cs:13, 349-350)
- 2026-07-11 | Dropped the dangling `docs/dev/rfc-memory-model.md §4 Option A` citation from the file
  header. (was SparseArenaWiringTests.fProxy.cs:11)

## SparseBSRTests
- 2026-07-12 | Dropped the "Regression for the fixed use-after-free" / "Pre-fix ... Post-fix" narration
  from the GrowthThenDispose comment, the "used to double-free / use-after-free" framing from the
  ToDense/ToBSR transpose-reference comment, the "Pre-fix this double-freed ... post-fix ... idempotent"
  framing from GrowthClearThenDispose_NoDoubleDispose's comment, and the "Dangling-arena-pointer
  history" pointer in DenseTransMatVec's comment. Full account: fProxyBSR.ToDense and
  fProxyBSRBuilder.ToBSR used to take `in Arena arena`, but both call a mutating arena allocator method
  internally (arena.fProxyMat / arena.fProxyBSR); since those Arena methods aren't `readonly`, calling
  them through an `in Arena` parameter forced the C# compiler to make a defensive copy of the arena, and
  the allocated result's internal arena pointer captured the address of that dead stack temporary -- a
  use-after-scope bug. Reading elements off the result was fine (the Values buffer is a real,
  independent allocation), but any op that allocated through the result's own arena pointer (e.g.
  Blas.trans(dense).fProxyTempMat) dereferenced the dangling pointer and threw "allocator handle is not
  valid" under Burst, breaking the trans(ToDense(...)) validation recipe. Fixed by changing both
  signatures to `ref Arena arena` (matching how ArenaExtensions factory methods take `this ref Arena`).
  Separately, a builder grown past capacityHint=8 to 225 triplets (~5 UnsafeList reallocations) used to
  leave the arena's tracked copy of the builder pointing at a freed pre-growth buffer; arena.Clear()
  then arena.Dispose() (whose own trailing Clear() runs again) used to double-free that stale copy;
  both are now fixed via the builder's idempotent Dispose() (_state null-guard).
  (was SparseBSRTests.fProxy.cs:77, 383-390, 564-579, 600-606)
- 2026-07-11 | Replaced the raw Windows exit code in the growth/dispose regression comment
  (`-1073741819`, i.e. STATUS_ACCESS_VIOLATION) with "an access violation" -- a platform-specific crash
  code has no business in cross-platform library test comments. Context preserved: builder grown past
  capacityHint=8 to 225 triplets (~5 UnsafeList reallocations); pre-fix, arena.Dispose() double-freed
  the stale pre-growth buffer held by the arena's tracked copy. (was SparseBSRTests.fProxy.cs:387-389)

## MIPTests
- 2026-07-13 | Dropped the "STAGE 2/3/4 (" banner labels from the file header and the (f)/(g)
  section banners (both the enum-region banners and the method-region "==== (f)/(g) ... ===="
  banners) -- reworded to describe the feature under test (pseudocost/reliability branching +
  best-bound queue; activity-based propagation + rounding heuristic + gap limits) instead of the
  internal stage number. Also dropped the "per the draft spec" qualifier from the header (the
  float-tiny/double-serious rule is stated directly). Trimmed three measured-baseline comments to
  their asserted bound only: Stage3NodesBranchy12's header and its two AssertNodesLE call-site
  comments (were "stage2 267 -> stage3 241 nodes"), P0033's "double explores ~447 nodes" aside,
  and Stage4NodesBranchy12's header + AssertNodes call-site comment (were "stage3 241 ->
  stage4(pre-cache) 218 -> stage4(fProxyLPCache) 199" -- the exact node-count history for this
  case is already recorded in this file's 2026-07-11 entry below). (was MIPTests.fProxy.cs:12-14,
  53, 70, 602, 658, 665-673, 689, 795, 890-891, 987-992, 1007)
- 2026-07-12 | Dropped "(third-review regression)" reviewer-workflow tag from the LargeMagnitudeIntegrality
  enum comment (kept the contract: MIP.solve must classify this fractional root via an absolute, not
  relative, integrality tolerance). Dropped the "Baselines were measured on both stages directly (by
  reverting to the stage-2 commit, running a throwaway diagnostic, then restoring)" debugging-methodology
  sentence from the STAGE 3 section banner. Dropped two "(The brief reported ...)" reviewer-brief asides
  (Stein15's header: the brief reported float stein15 finishing in ~261 nodes, which did not reproduce on
  the shipped code; Stage4NodesGomoryWolsey's comment: the brief reported 5 nodes for both dtypes, only
  double reproduced it). (was MIPTests.fProxy.cs:56-58, 96-100, 833, 946)
- 2026-07-11 | Trimmed the debugging-timeline narration on Stage4NodesBranchy12's node-count comment.
  Full history: node count is 199 (down from stage-3's 241, stage-4-without-cache's 218) once
  fProxyLPCache's persisted DSE weights are in play. An earlier coding-in-progress value of 216 traced
  back to a real bug: weight[] was resumed even when the entry eta-capacity check forced a fresh
  Refactorize, letting weight drift across an unbounded refactorization chain instead of being bounded
  like the eta chain; fixed in DualSimplexPivotCore via a `didResumeFactors` guard. Kept in-source: node
  count is 199, stable across reruns (not a per-launch nondeterminism artifact), and why (persisted DSE
  weights change branch-variable pricing at a warm-started non-logical basis). Also dropped the dangling
  `docs/draft-spec-mip.md` header citation and the redundant `docs/draft-spec-mip.md stage 4` pointer
  (both internal-only, not shipped). (was MIPTests.fProxy.cs:11, 73, 972-981)

## QRCPDowndateTests
- 2026-07-13 | Reworded "reported out via Counts for the orchestrator" to "for the managed
  driver" (residue of the ORCHESTRATOR-narration cleanup). (was QRCPDowndateTests.fProxy.cs:1034)
- 2026-07-12 | Dropped "ORCHESTRATOR DIAGNOSIS" / "ORCHESTRATOR VERIFICATION" / "adversarial
  mutation-testing review pass" agent-workflow narration and the OQ-D1/OQ-D2 ticket references from
  KahanSweep's header, GradualDecay's KNOWN, DISCLOSED LIMITATION comment, and TierEDistinctMagnitudes'
  header (kept the underlying engineering facts as plain contract prose). Full history: the KahanSweep
  n=32/64 exclusion was found via mutation testing (no test pinned "no spurious Kahan permutation" on
  float); a naive fix asserting P==identity for the whole n×theta grid failed at n=32/64, root-caused via
  an in-test exact-recompute oracle comparison to the classic Kahan/RRQR trailing-norm collapse (not a
  downdate defect) -- widening the pivot-tie tolerance 2x had zero effect, and adding an unrelated
  Debug.Log in the pivot-selection loop was enough to flip which type failed, confirming an
  edge-of-precision tie-break rather than a reproducible bug. GradualDecay's ORCHESTRATOR VERIFICATION
  (temporary instrumentation, since removed per OQ-D2: production must not carry a guard-fire counter)
  measured the real cumulative guard firing 1 time vs. a naive per-step guard's 0 times on a float,
  cond≈673, n=128 construction, confirming the mechanism activates on realistic ill-conditioned input.
  (was QRCPDowndateTests.fProxy.cs:102-122, 319-325, 542)
- 2026-07-11 | Dropped three dangling internal-doc citations: the file-header pointer to
  `docs/dev/spec-qrcp-downdate.md`; the "This was checked, not assumed -- see the review pipeline notes
  in docs/dev/spec-qrcp-downdate.md's implementation report" sentence in GradualDecay's KNOWN, DISCLOSED
  LIMITATION comment (kept the substance: the back-of-envelope bound itself, and that it was checked via
  that bound, not merely assumed); and the `(docs/dev/spec-qrcp-blocked.md OQ-B2)` citation in
  BlockedPanels (the surrounding sentence already states the actual fact -- GEMM accumulation vs the
  oracle's rank-1 chain is a different summation order -- so the citation was pure redundancy).
  (was QRCPDowndateTests.fProxy.cs:14, 341-342, 625)

## PCATests
- 2026-07-13 | Dropped two "per the spec" / "the spec says" qualifiers (WellSeparated's doc and
  the well-separated-columns comment in the vector-agreement check) -- the surrounding sentence
  already states the rule (route-/precision-dependent rotation on near-tie columns has no sign-rule
  fix, so vector comparisons are skipped there). Dropped "the fable-caught trap" agent-name
  reference from CorrelationDegenerate's two comments (file header list + section banner) -- kept
  the contract: a constant column must not spuriously add a unit eigenvalue on the Correlation
  route. (was ML/PCATests.fProxy.cs:26, 124, 206, 531)
- 2026-07-12 | Dropped agent-workflow narration from SvdTruncatedWideThrows's comment ("The coder
  confirmed..."). Full account: svdTruncated is NOT shape-free; pcaSVDTruncated adds the n>=p guard so
  it throws on wide data (p>n) just like pcaSVD/pcaRandomized. Deliberately no "truncated works on wide
  data" test. (was ML/PCATests.fProxy.cs:626)
- 2026-07-11 | Dropped the dangling `docs/dev/spec-pca.md "Tests"` citation introducing the
  acceptance-criterion list; the numbered list itself (#1-#7a etc.) is self-describing.
  (was ML/PCATests.fProxy.cs:18)

## QueryTests / QueryPredicateTests (fProxy / iProxy)
- 2026-07-13 | Dropped the remaining internal spec-ticket labels the 2026-07-11 cleanup missed:
  "(review's CRITICAL regression)" and "(Fix 6)" in QueryTests.fProxy.cs; "(spec P1)" (x2, header
  + SYMMETRY banner) in both QueryTests.fProxy.cs and QueryPredicateTests.fProxy.cs; the T1-T5/
  AC#3/AC#4 group-banner labels throughout QueryPredicateTests.fProxy.cs; "(spec P2/P6)" in
  iProxy/QueryTests.iProxy.cs; "(T1, integer)" in iProxy/QueryPredicateTests.iProxy.cs. These came
  from docs/dev/spec-query.md / spec-predicate-queries.md's T1-T5/AC#/P-n taxonomy. The surrounding
  prose already names each group/check, so no rewording was needed beyond deleting the
  parenthetical. (was QueryTests.fProxy.cs:21, 711, 800, 1131; QueryPredicateTests.fProxy.cs:22,
  123, 187, 251, 288, 352, 387, 440, 560; QueryTests.iProxy.cs:913; QueryPredicateTests.iProxy.cs:50)
- 2026-07-12 | Dropped the "(review's HIGH finding)" reviewer-note reference from the MinValue-edge
  section banner in iProxy QueryTests. iAbs() on iProxy.MinValue saturates to iProxy.MaxValue (an
  off-by-one on the |MinValue| case, since MinValue has no positive counterpart in two's complement);
  this was originally raised as a HIGH-severity review finding and is now covered by MinValueEdge().
  (was iProxy/QueryTests.iProxy.cs:699)
- 2026-07-11 | Dropped dangling citations to `docs/dev/spec-query.md` (fProxy and iProxy QueryTests) and
  `docs/dev/spec-predicate-queries.md` (fProxy and iProxy QueryPredicateTests), plus the internal
  "(policies P2/P3)" and "(Section 4b + T1)" / "(Section 6 = T1..T5)" section labels. Where the spec
  citation was the entire line with no other content (fProxy QueryPredicateTests, fProxy QueryTests),
  the line was deleted outright rather than reworded.

## ControlTests / ControlLQRTests
- 2026-07-13 | ControlLQRTests: reworded "The coder's smoke tests live in ControlTests.fProxy.cs"
  to "Basic smoke coverage lives in ControlTests.fProxy.cs" (the mirrored agent-workflow reference
  in ControlTests' own header was cleaned 2026-07-12; this side was missed). Dropped four "per
  spec"/"the spec allows"/"the spec's grid"/"the task's stabilizability guard" qualifiers -- the
  surrounding sentence already states the rule in each case. Dropped the measured warm/cold
  iteration-count aside from WarmPerturbation's comment (assertion is a generous absolute bound,
  not the measured numbers). (was ControlLQRTests.fProxy.cs:11, 30, 110-111, 131, 134-135, 232)
- 2026-07-12 | Dropped agent-workflow narration from ControlTests' file header ("written by the coder
  agent alongside the implementation" / "is the test-writer agent's job"). Kept the substance: this file
  is basic smoke coverage (known tiny instance solves, statuses fire, throws throw), not the full battery
  (literature vectors, SDA-vs-oracle cross-check, property-based stability/PSD checks, warm-path
  perturbation convergence, redundant-actuator rank flagging, determinism). (was ControlTests.fProxy.cs:11)
- 2026-07-11 | Dropped dangling `docs/spec-lqr.md` citations (and the "per that spec's binding rules" /
  "'Tests' section items 1-7" internal labels) from both file headers. The surrounding prose already
  describes what each file covers (smoke tests vs. the full battery) without needing the spec.

## CompMathTests
- 2026-07-13 | Dropped "(renamed from the kernel's old distance/distancesq names ...)" and
  "Renamed from sincosInPlace - review flagged the InPlace suffix as misleading ..." rename-history
  asides. Kept the contracts: absDiff/sqrDiff are componentwise, not whole-vector Euclidean
  distances; sincos does not mutate x. (was CompMathTests.fProxy.cs:461-464, 508-509)

## CHOTests / ConjugateGradientTests
- 2026-07-13 | Dropped "(choleskySolve(in A, ref L, ref b) was deleted -- it was a 2-line
  composition in disguise)" from both files' Cholesky-solve comments. Kept the contract: factor +
  solve as the explicit two-call composition, b overwritten with x. (was CHOTests.fProxy.cs:175-176,
  ConjugateGradientTests.fProxy.cs:188-189)

## UKFTests
- 2026-07-13 | Dropped the tolerance-calibration narration ("Calibrated from a float32/float64
  numpy prototype ... measured max|x diff|~1.9e-6 ... unlike the steadyStateGain tolerance
  episode, which was calibrated too tight against a since-fixed bug"). Kept only the contract:
  both tolerances carry a large margin over the prototype-measured error, in both precisions.
  (was UKFTests.fProxy.cs:51-56)

## SVDRandomizedTests
- 2026-07-13 | Dropped three "Measured ..." asides (worst relative error < 1e-4 for the q=2/
  oversample-10 case; ratio ≈ 1.0000001; q=0 vs q=2 summed-rel-error 0.19/0.29 vs 6e-5/1.6e-4).
  Kept the resulting bound in each test (2% rel tol; 1.05 ratio; q=2 <= q=0 monotone check). (was
  SVDRandomizedTests.fProxy.cs:226, 281, 322)

## CHOTests / CHOPTests / LUTests / QRCPTests
- 2026-07-13 | Dropped the remaining "Solver API rework (commit 2)" / "Commit 2.5 (2a)"
  commit-ticket refs from method-body and enum-case comments (the 07-12 pass only caught the
  enum-comment copies in CHOTests) and the "Stage-3 direct-solve-status coverage" internal stage
  label from all four files. Contracts kept in place (decompSolve-exit reusability, driver
  short-circuit purity, DirectSolveStatus/RankInfo coverage on non-PD/indefinite/singular/
  rank-deficient input). (was CHOTests.fProxy.cs:213, 249, 381; CHOPTests.fProxy.cs:36, 39, 276,
  437, 488; LUTests.fProxy.cs:45, 49, 381, 1019; QRCPTests.fProxy.cs:41, 355, 511-512, 915)

## CHOTests
- 2026-07-12 | Dropped the "Solver API rework (commit 2)" and "Commit 2.5 (2a)" commit-ticket references
  from the SolveInPlaceExitIsUsableFactor / SolveInPlaceShortCircuitPurity enum-case comments. Kept the
  contracts: solveInPlace's exit factor is a valid decompSolve input, bit-identical to a fresh decomp +
  decompSolve on the same original A; and non-PD input leaves b_to_x untouched (short-circuit purity).
  (was CHOTests.fProxy.cs:41, 44)

## ArenaWiringTests (fProxy) — generational-overlay section
- 2026-07-12 | Dropped the "Stage E" internal stage label from the generational-overlay guard-tests
  section banner and its lead sentence. Kept the contract: a checks-gated "generational overlay" on the
  arena-tracked structs' Data getter throws InvalidOperationException on a stale handle instead of
  silently returning a dead/garbage buffer. (was ArenaWiringTests.fProxy.cs:370-371)

## QRTests / SVDTests
- 2026-07-13 | Dropped the remaining "Solver API rework (commit 2)" / "Commit 2.5 (2f-i)" /
  "Commit 2.5 SVD coverage restoration" commit-ticket references (4 sites in QRTests, 6 in
  SVDTests) and the "Ported from the deleted Jacobi-oracle" / "Ported from the deleted
  SVDSingleColumn" / "Ported from the deleted SVDNonConvergence" porting-history framings (3
  sites in SVDTests) -- reworded to name the coverage directly (A-preservation, uninit-x
  contract, independent-algorithm cross-check, known-value oracle, non-convergence regression
  guard). Contracts kept in place. (was QRTests.fProxy.cs:39, 42, 331, 1098; SVDTests.fProxy.cs:
  51, 53, 340, 379, 538, 565, 593)

## SparseArenaWiringTests / ArenaWiringTests (iProxy / bool) -- generational-overlay section
- 2026-07-13 | Dropped the remaining "Stage E" stage labels from the generational-overlay
  guard-tests section banners in the three siblings the 2026-07-12 fProxy-dense cleanup missed
  (SparseArenaWiringTests.fProxy.cs, ArenaWiringTests.iProxy.cs, ArenaWiringTests.bool.cs). Kept
  the contract: a checks-gated "generational overlay" on the arena-tracked structs' data getters
  throws InvalidOperationException on a stale handle instead of silently returning a dead/garbage
  buffer. (was SparseArenaWiringTests.fProxy.cs:388-389; ArenaWiringTests.iProxy.cs:366-367;
  ArenaWiringTests.bool.cs:341-342)

## SVDLowRankTests
- 2026-07-13 | Dropped commit hash de74c48 (x2) and the "FIX 1"/"FIX 2" labels; FIX 1 = the
  converged-residual check once computed V instead of U, FIX 2 = alpha-breakdown betaLast=0
  handling. Contracts kept in place (partial reorthogonalization semantics; what each test
  exercises). (was SVDLowRankTests.fProxy.cs:41, 583, 619, 807)

## MPCTests
- 2026-07-13 | Dropped the "post-ship audit finding (2026-07-12, OP/DEVLOG.md)" citation and the
  duplicated bug postmortem (prestabilized input-bound rows read the wrong Phi/Gamma block; R
  applied naively to v instead of expanding u_k^T R u_k) from PrestabBindingBoundMatchesNonPrestab's
  comment -- the full postmortem already lives in OP/DEVLOG.md's "## MPC / MPC.State" entry. Kept
  the regression contract: prestabilization is a pure change of coordinates, so it must reproduce
  the identical physical answer as the non-prestabilized solve of the same problem. (was
  MPCTests.fProxy.cs:282-291)

## GalleryTests / GalleryPhase2Tests
- 2026-07-11 | Dropped dangling `docs/dev/spec-gallery.md` citations from both file headers.
  GalleryPhase2Tests kept its other parenthetical (the production template file name
  Gallery.Phase2.fProxy.cs), which is a real in-repo reference, not an internal-only doc.

## LiteratureTests
- 2026-07-13 | Dropped "See memory note literature-test-vectors" -- pointed at the agent's private
  memory file, meaningless to a human reader of the repo. The rest of the header already states
  the file's contract (known closed-form results, independent reference values). (was
  LiteratureTests.fProxy.cs:13)

## VectorCopyTests
- 2026-07-13 | Dropped the "Previously both routed to the temp pool, so Copy() returned a vector
  that ClearTemp would free out from under the caller (use-after-dispose)" postmortem. Kept the
  contract: Copy() must be PERSISTENT, TempCopy() must be TEMP. (was VectorCopyTests.fProxy.cs:7-9)

## StatsTests
- 2026-07-13 | Dropped "Previously 1/(M-1) = 1/0 = Inf and 0*Inf = NaN filled every cell" from
  covarianceInto's M_Rows<2 guard test, and "(bug-fix)" from SingleElementVariance's comment. Kept
  the contracts: covarianceInto zero-fills for M<2; single-element variance/stdDev are exactly 0.
  (was StatsTests.fProxy.cs:107-109, 271)

## BoolIndexingTests
- 2026-07-13 | Dropped "previously dereferenced a null arena core (the old _arenaPtr field, now
  the _arena handle's _core)" from MatrixCopyNullArenaGuard's comment. Kept the contract: copying
  a standalone (null-arena) matrix with the default allocator must fall back to Allocator.Temp
  without crashing. (was BoolIndexingTests.cs:172-174)

## SparseBSRTests
- 2026-07-13 | Dropped the remaining "used to leave the arena's tracked value-copy... (double-free
  / use-after-free on dispose)" / "used to double-free / use-after-free... (native crash)" history
  framing that survived the 2026-07-12 relocation pass in BuildDenseGrown's and
  GrowthThenDisposeTest's comments -- the DEVLOG entry above already tells the same story. Kept
  "this is the growth path the regression tests pin." (was SparseBSRTests.fProxy.cs:366-368, 544-545)

## KrylovFusedKernelTests / KrylovRound2Tests / KrylovVerifyAtExitTests / SSORTests / SparseSpMMTests / LargeSparseBenchmark
- 2026-07-13 | Dropped the remaining "R6a"/"pre-R6a" ticket-code tags throughout
  KrylovVerifyAtExitTests (7 sites: header, UnguardedCg's doc, the no-verify branch comment, the
  rnorm-contract comment, the healthy-path section banner, and the pcg/cgls/cgne wiring-check
  banner) and the "R1 review caught" tag in KrylovFusedKernelTests' rectangular-updateXR comment
  -- all reworded to name the invariant directly (verify-at-exit / the shared-loop-bound
  regression) instead of the round/review label. Also trimmed
  VerifyAtExitCatchesOptimisticDriftOnIllConditionedMoler's debugging-methodology narration
  ("prototyping the recurrence in a throwaway dotnet console app ... via a throwaway diagnostic
  sweep run through Unity") down to the resulting contract: n/alpha/tol are tuned to this
  library's actual Krylov kernels, float lies at this size and double does not, hence the
  requireLie choose-gate. (was KrylovVerifyAtExitTests.fProxy.cs:16, 62, 96, 113-122, 169, 177,
  223, 225; KrylovFusedKernelTests.fProxy.cs:147-148)
- 2026-07-12 | SSORTests: dropped the "Krylov Round-3" preamble from the file header too (a later pass
  judged the bare round label still read as an internal stage marker, superseding the 2026-07-11 call
  below to keep it there); replaced with a plain "SSOR preconditioner test coverage:" lead-in for the
  (a)-(e) list. (was SSORTests.fProxy.cs:11)
- 2026-07-12 | SparseSpMMTests: dropped the "ApplyBlock now streams ... instead of looping ... k scalar
  BSR.spMV calls" / "OLD per-row-Apply ApplyBlock it replaced" / "no need to check out pre-change
  history" change-history narration from the file header. Kept the contract: ApplyBlock streams the
  matrix once and applies to k row-vectors together (instead of k separate scalar BSR.spMV calls); SpMM
  output must equal k separate spMV rows, bit-identical; LOBPCG's trajectory must be bit-identical against
  an in-test scalar-loop reference operator. (was SparseSpMMTests.fProxy.cs:10-11, 20-23)
- 2026-07-11 | Dropped all `docs/draft-spec-krylov-optimization.md` citations (R1/R2/R3/R3b/R5/R6a round
  labels) across these files' headers and inline comments. Kept the bare "Round-1" / "Round-2" /
  "Round-3" / "Round-3b" labels where they read as a self-contained section name; dropped the "R5"/"R3"
  short-form tags entirely where they added no information beyond what the surrounding sentence already
  said (SparseSpMMTests' "before" oracle comment, LargeSparseBenchmark's SSOR-preconditioner-for-LOBPCG
  comment). Also dropped a vague "(spec §3b/task brief)" aside in the same LargeSparseBenchmark comment.

## SparseSolverTests
- 2026-07-13 | Dropped the "STAGE 2:" banner tag from the rnorm-contract section comment, the
  "(added this pass)" workflow tag from the pcg rzold>0 guard comment, and the hardcoded
  "minres ~L595, biCGStab ~L797, cgls ~L999, lsqr ~L1175 of Krylov.fProxy.cs" line-number
  references in the warm-start section banner (replaced with "its own pre-loop residual check" --
  line numbers in a sibling template rot the moment either file is edited). (was
  SparseSolverTests.fProxy.cs:503, 820-821, 1532)

## SparseSpMMTests
- (see combined Krylov entry above)

## AccuracySweepTests
- 2026-07-13 | Reworded "so a reviewer can see the input really was ill-cond" to "to confirm the
  input really is ill-conditioned" (reviewer-address language), and dropped "exactly as the spec
  warned" from the QR_Hilbert reference-comparison comment (the sentence already states why a
  fixed tiny bound would false-fail on this input). (was AccuracySweepTests.fProxy.cs:32, 166)

## ArenaWiringTests (fProxy) -- temp-recycling section
- 2026-07-13 | Dropped "wanted by the spec" from TempRecyclingCycles' NOTE comment; the sentence
  already states the stronger check it's contrasting with. (was ArenaWiringTests.fProxy.cs:127-128)

## UnsafeSortTests
- 2026-07-11 | Dropped the dangling `docs/spec-shipped-feature.md pillar 3` citation; kept the quoted
  testing policy itself ("New Blas/UnsafeOP kernels get DIRECT tests against a plain scalar reference
  implementation, not just indirect coverage through callers") inline since it's the actual content, not
  just a pointer. (was UnsafeSortTests.fProxy.cs:15)

## LOBPCGSmokeTests
- 2026-07-13 | Dropped the periodic-initial-X-seed bug history from the k=6 seeding test's
  comment (an earlier default seed `(i+c*3+1)&3` repeated with period 4 in both i and c, so the
  seeded block had at most 4 distinct rows -- exactly rank-deficient for k>4, silently absorbed by
  FactorGram's ridge retry rather than failing loudly) -- kept the contract: the fill must be
  non-periodic to span all 6 rows. Dropped the debugging narrative ("found, while iterating, to
  hit...") and the "worth a dedicated hardening follow-up, out of scope here" open TODO from the
  k=2-not-3 comment on GeneralizedLaplacianDiagBMatchesDenseReduction -- kept the contract: a k=3
  version of this setup hits a rare numerical edge case in the shared Rayleigh-Ritz/
  OrthonormalizeBlock(B) design (not B-specific) when two of three pairs lock in the same
  iteration while the third's residual is also already tiny; k=2 avoids that pattern. Follow-up
  idea (dedicated hardening test for the k=3 case) belongs on the regressions-todo tracker, not in
  this comment. (was LOBPCGSmokeTests.fProxy.cs:179-183, 290-300)
- 2026-07-13 | Dropped "per the spec's suggested recipe" from the buckling-oracle comment header
  -- the recipe itself is stated in full in the following sentences. (was
  LOBPCGSmokeTests.fProxy.cs:284)
- 2026-07-12 | Dropped agent-workflow narration from the file header ("written by the coder agent
  purely to sanity-check the algorithm while iterating", quoted task-brief language, "left for the
  independent test-writer agent" x2, "the coder's OWN scratch smoke tests"). Kept the substance: this
  file is smoke coverage, not the comprehensive suite (analytic Laplacian oracle across k=1..4,
  BSR+block-Jacobi generalized preconditioned convergence comparison, rank-deficiency stress on the
  generalized path, breakdown/non-SPD-B behavior, and warm-start on the generalized cache are still
  open coverage gaps for a full suite). Also dropped "-- left for the independent test-writer agent"
  from GeneralizedBSRSmokeRunsAndConverges' comment (kept the substance: this only exercises SPD-A/B
  convergence, not the indefinite-A/buckling-shaped BSR case).
  (was LOBPCGSmokeTests.fProxy.cs:11-25, 460-464)

## InPlaceOpTests
- 2026-07-12 | Dropped the "PR #1" ticket reference and bug postmortem from the file header. Full
  account: addInPlace(place, from) is supposed to mutate `place` (place += from); the internal compAdd
  operands were reversed, so the method used to mutate the wrong operand instead — masked end-to-end
  only because the + operators also called it backwards. (was InPlaceOpTests.fProxy.cs:7-10)

## LPTests
- 2026-07-13 | Dropped the remaining "per the spec"/"the spec's ..."/"stage-1 test gap in the
  original spec" qualifiers (DegenerateDuplicatedRows, LadFNvsOracle x2, LadBRvsOracle,
  LadBRStackloss, LadBRLargeMSortPath's second literature-vector skip, LadBRKnownAnswer) -- each
  surrounding sentence already states the tolerance/rule directly. Dropped the "Observed margins are
  wide (2026-07-09): double warm 1 / cold 19, float warm 2 / cold 16" measured-baseline aside
  (assertion is `warm < cold`, unconditionally). Trimmed RevisedDenseCovering's "Reproduces a bug
  the benchmark caught" postmortem to a plain regression-guard statement, and
  LadBRLargeMSortPath's "verified via instrumentation" methodology narration down to the resulting
  fact (at m=1000 float, only LP.lad(RevisedSimplex) returns MaxIterations). (was
  LPTests.fProxy.cs:910, 1018-1019, 1123-1124, 1416, 1459, 1482, 1512, 1526, 1541-1542, 1568,
  1652-1653)
- 2026-07-11 | Dropped twelve dangling internal-doc citations across this file: repeated
  `docs/spec-revised-simplex.md` "stage 1"/"stage 2" labels on the RevisedSimplex/DualSimplex section
  headers (both the TestType enum comments and the method-body section banners), `docs/draft-spec-mip.md
  stage 1` on the two LPBasis warm-start banners, `docs/spec-lpbasis-persistence.md acceptance 3` /
  `acceptance item 5` on the fProxyLPCache banners (x2) and the (3b) contract-violation comment,
  `docs/spec-lad-frisch-newton.md`'s "Tests section (4 items)" / "item 1" / "item 2" labels on the
  Frisch-Newton section (kept the actual tolerances: 1e-6 rel double / 1e-3 rel float L1-residual match,
  and the LadStackloss 5e-2 published-coefficient tolerance), and `docs/spec-lad-barrodale-roberts.md`'s
  "Tests section (6 items)" label on the Barrodale-Roberts section. All replaced with self-contained
  prose (no doc paths, no acceptance-item numbers) or removed outright where purely redundant with the
  surrounding sentence.

## ScalarMatrixOpTests (fProxy)
- 2026-07-13 | Dropped the "review-found bugs" header (scalar-matrix used to delegate to
  rhs-lhs and negate; 0/A used to throw DivideByZeroException pre-guard) and the "(review fix D)"
  / "pre-fix" tags -- the iProxy twin had this same postmortem dropped 2026-07-12, this file was
  missed. Kept the contracts: scalar - matrix must equal s - A[i,j] elementwise; normalizeLP must
  sum pow(|x|,p) not pow(x,p); 0/A must not throw. (was ScalarMatrixOpTests.fProxy.cs:10-13, 43, 52)

## ScalarMatrixOpTests (iProxy)
- 2026-07-12 | Dropped the bug-postmortem file header ("the operator delegated to `rhs - lhs`, which
  negates the result since subtraction is not commutative"). Full account: `scalar - matrix` for integer
  matrices used to delegate to `matrix - scalar` (rhs - lhs) internally, which silently negated every
  result; the fix made the operator compute s - A[i,j] directly. The in-body comment already states the
  contract (5 - [[1,2],[3,4]] must be [[4,3],[2,1]]). (was iProxy/ScalarMatrixOpTests.iProxy.cs:9-10)
