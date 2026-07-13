# Release scan 2026-07-13 - N12: TemplateSourceTests, second half

Narrow scan, all dimensions + narrow-pass addendum patterns (pattern 7 emphasized).
Every line of every file in the partition was read.

## Partition covered (files 69-136 of 136, sorted by full path)

Under `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/`:
fProxy/: QRLeastSquaresResidualTests, QRTests, QRWorkspaceTests, QueryPredicateTests, QueryTests,
RandomMatrixTests, RandomTests, RandomWeightedTests, ResampleTests, RollingWindowTests, SSORTests,
SVDLowRankTests, SVDRandomizedTests, SVDSolverTests, SVDSubspaceTests, SVDTests, SVDWorkspaceTests,
ScalarMatrixOpTests, SelectRefTests, SolverBatteryTests, SolversTests, SparseArenaWiringTests,
SparseBSRTests, SparseCompNormsTests, SparseEigenTests, SparseGalleryTests, SparseIC0Tests,
SparseILU0Tests, SparseSolverTests, SparseSpMMTests, SparseStructuralTests, SparseSymmetricTests,
SparseTransposeTests, SparseUnrollTests, SpecialConstructorsTests, StatsTests, SvdFullWorkspaceTests,
SvdRandomizedWorkspaceTests, SvdThinValuesWorkspaceTests, TransformsTests, TransposeTests, UKFTests,
UnsafeSortTests, VectorCopyTests (all .fProxy.cs);
root: fProxyPivotTests.cs;
iProxy/: AnalysisTests, ArenaWiringTests, BridgeFillTests, ChooseMarkerTests, ClampTests,
CompBitsTests, CompMathTests, CompareTests, DebugExportTests, DotOperationTests, DotRefTests,
HashTests, IndexingTests, InitTest, NormsTests, OperationsTest, QueryPredicateTests, QueryTests,
RandomTests, ScalarMatrixOpTests, SelectRefTests, StatsTests, TransposeTests (all .iProxy.cs).

TemplateConverter.cs and GenUtils.cs read first for token/choose/skipFor/alsoExpand rules;
folder DEVLOG.md read and cross-referenced before flagging anything as leftover.

## HIGH

None.

## MEDIUM

**M1 - fProxy ScalarMatrixOpTests still carries the postmortem/review header its iProxy twin already had relocated (sibling drift).**
fProxy/ScalarMatrixOpTests.fProxy.cs:10-13 ("Regression tests for review-found bugs: scalar - matrix returned matrix - scalar (negated) because ... operator delegated to rhs - lhs"), :43 ("(review fix D)"), :52 ("pre-fix this threw DivideByZeroException").
DEVLOG (## ScalarMatrixOpTests (iProxy), 2026-07-12) records dropping exactly this postmortem from the iProxy twin; the fProxy file was missed.
Fix direction: keep the contracts ("5 - [[1,2],[3,4]] must be [[4,3],[2,1]]"; "0/M must not throw"); move history to DEVLOG:
## ScalarMatrixOpTests (fProxy) / - 2026-07-13 | Dropped "review-found bugs" header (scalar-matrix used to delegate to rhs-lhs and negate; 0/A used to throw DivideByZeroException pre-guard) and the "(review fix D)" / "pre-fix" tags. (was ScalarMatrixOpTests.fProxy.cs:10-13,43,52)

**M2 - "Stage E" internal stage label survives in two sibling files after the dense fProxy cleanup.**
fProxy/SparseArenaWiringTests.fProxy.cs:388-389 and iProxy/ArenaWiringTests.iProxy.cs:366-367 ("Generational-overlay guard tests (Stage E; ...)" / "Stage E added a checks-gated ...").
DEVLOG (## ArenaWiringTests (fProxy) - generational-overlay section, 2026-07-12) dropped the label from the dense fProxy file only.
Fix direction: same edit as the dense twin; DEVLOG entry:
## SparseArenaWiringTests / ArenaWiringTests (iProxy) / - 2026-07-13 | Dropped the remaining "Stage E" stage labels from the generational-overlay banners (dense fProxy twin was cleaned 2026-07-12). (was SparseArenaWiringTests.fProxy.cs:388-389; ArenaWiringTests.iProxy.cs:366-367)

**M3 - commit hash and FIX-n ticket codes in SVDLowRankTests comments.**
fProxy/SVDLowRankTests.fProxy.cs:41 and :807 ("partial reorthogonalization (de74c48)" - a commit hash in a shipped comment), :583 ("Also exercises FIX 2 (alpha-breakdown betaLast=0)"), :619-620 ("Directly exercises FIX 1's residual check (the path that was computing V instead of U)" - inline bug postmortem).
Fix direction: keep the contract sentences (what partialReorth selects; what the residual check must catch); move the hash + FIX-n history to DEVLOG:
## SVDLowRankTests / - 2026-07-13 | Dropped commit hash de74c48 (x2) and FIX 1/FIX 2 labels; FIX 1 = the converged-residual check once computed V instead of U, FIX 2 = alpha-breakdown betaLast=0 handling. (was SVDLowRankTests.fProxy.cs:41,583,619,807)

**M4 - "Solver API rework (commit 2)" / "Commit 2.5" commit-ticket references in QRTests and SVDTests.**
fProxy/QRTests.fProxy.cs:57 ("Solver API rework (commit 2) coverage."), :60 ("Commit 2.5 (2f-i): ..."), :349, :1115; fProxy/SVDTests.fProxy.cs:51 ("Solver API rework (commit 2): uninit-x contract."), :53 ("Commit 2.5 SVD coverage restoration:"), :340, :379; plus the "Ported from the deleted Jacobi-oracle ..." history framings at SVDTests.fProxy.cs:538,565,593.
Same class of reference DEVLOG already relocated for CHOTests (2026-07-12); these files were missed.
Fix direction: keep the contract text (decomp must not modify A; x is OUTPUT ONLY; etc.); one DEVLOG entry per file recording the commit-2/2.5 provenance and the deleted-Jacobi porting note. (was QRTests.fProxy.cs:57,60,349,1115; SVDTests.fProxy.cs:51-53,340,379,538,565,593)

**M5 - "STAGE 2" label, "(added this pass)" workflow tag, and hardcoded Krylov line numbers in SparseSolverTests.**
fProxy/SparseSolverTests.fProxy.cs:1532 ("STAGE 2: the square solvers ... now RETURN an SolveInfo"), :504 (the pcg rzold>0 guard "(added this pass)"), :821-822 ("minres ~L595, biCGStab ~L797, cgls ~L999, lsqr ~L1175 of Krylov.fProxy.cs" - rot-prone file:line references to another template).
Fix direction: state the contract without the stage tag (cg/pcg/minres/biCGStab/cgne return SolveInfo with a free tracked rnorm); drop "(added this pass)"; replace line numbers with "each solver's pre-loop residual check". DEVLOG:
## SparseSolverTests / - 2026-07-13 | Dropped "STAGE 2" banner tag, "(added this pass)" on the pcg rzold>0 guard, and the hardcoded Krylov.fProxy.cs line numbers in the warm-start banner. (was SparseSolverTests.fProxy.cs:504,821-822,1532)

**M6 - review-finding tags and residual spec ticket labels in the Query family.**
fProxy/QueryTests.fProxy.cs:800 ("(review's CRITICAL regression)"), :1131 ("(Fix 6)"); residual internal spec labels T1/T2/T3/T4/T5, AC#3/AC#4, "(spec P1)" in fProxy/QueryPredicateTests.fProxy.cs:15-27,123,187,251,287,352,387,441,561 and fProxy/QueryTests.fProxy.cs:21,711; "(spec P2/P6)" in iProxy/QueryTests.iProxy.cs:913; "(T1, integer)" in iProxy/QueryPredicateTests.iProxy.cs:50.
The 2026-07-11 cleanup (DEVLOG ## QueryTests / QueryPredicateTests) dropped the doc paths and some labels but left these ticket codes.
Fix direction: delete the parenthetical codes (the surrounding prose already names each group); one-line DEVLOG note that the T1-T5/AC#/P-n labels came from docs/dev/spec-query.md / spec-predicate-queries.md.

**M7 - retired class name MatrixMetrics in SolverBatteryTests comments.**
fProxy/SolverBatteryTests.fProxy.cs:764 ("// Condition number (MatrixMetrics.cond).") and :20 (header lists "MatrixMetrics") - the code below calls Analysis.cond; MatrixMetrics no longer exists (naming purge). A reader grepping for the commented API finds nothing.
Fix direction: replace MatrixMetrics.cond with Analysis.cond in both comments.

**M8 - iProxy Query headers claim generation into retired *Query_OP types.**
iProxy/QueryTests.iProxy.cs:13-14 ("One template expands to intQuery_OP / shortQuery_OP / longQuery_OP") and iProxy/QueryPredicateTests.iProxy.cs:15 ("expands to int / short / long QueryOP"). The _OP suffix on non-data types was purged; the generated surface is the shared Query class - the comment misdescribes what codegen produces.
Fix direction: reword to "expands per integer type (int/short/long)"; no DEVLOG needed (pure doc fix).

**M9 - iProxy MatVec dot tests assert only output LENGTH, not values.**
iProxy/DotOperationTests.iProxy.cs:94-109 (MatVecDot) and :183-198 (MatVecDotNonSquare): build a random A and x = ones, then assert only b.N == outVecLen. The mat*vec VALUES are never checked against an oracle in this file (VecMat and MatMat cases do check values). iProxy/DotRefTests.iProxy.cs only proves ref-dest == allocating (same kernel, circular). With x = ones the row-sum oracle is one loop away.
Fix direction: assert b[i] == row-sum of A row i (exact integer) in both cases.

## LOW

- **L1** fProxy/RollingWindowTests.fProxy.cs:181,201 + test name CovarianceMatchesStatsOP - retired StatsOP name in comments and a test-method name (code calls Stats.covariance). Rename comment references; the method name is visible in test runners but renaming is optional.
- **L2** fProxy/TransformsTests.fProxy.cs:13-15 - header names retired StatsOP: / NormsOP: / OP.Component: groupings; code uses Stats/Norms/fProxyComp.
- **L3** fProxy/StatsTests.fProxy.cs:107-109 ("Previously 1/(M-1) ... the guard now zero-fills") and :271 ("variance==0 (bug-fix)") - bug-history wording; keep contract (covarianceInto zero-fills for M<2), move "previously NaN-filled" to DEVLOG. (was StatsTests.fProxy.cs:107-109,271)
- **L4** fProxy/VectorCopyTests.fProxy.cs:7-9 ("Previously both routed to the temp pool ... use-after-dispose") - postmortem header; keep the contract sentence, relocate history to DEVLOG.
- **L5** fProxy/UKFTests.fProxy.cs:51-56 - tolerance-calibration narration ("Calibrated from a float32/float64 numpy prototype ... measured max|x diff|~1.9e-6 ... unlike the steadyStateGain tolerance episode, which was calibrated too tight against a since-fixed bug") - measured baselines + bug-history aside; DEVLOG relocation, keep only "large margin over prototype-measured error, both precisions". (was UKFTests.fProxy.cs:51-56)
- **L6** fProxy/SparseBSRTests.fProxy.cs:366-368 and :544-545 - "used to leave the arena's tracked value-copy ... double-free / use-after-free (native crash)" history framing survived the 2026-07-12 relocation pass (DEVLOG ## SparseBSRTests already tells the same story - the comments can keep only "this is the growth path the regression tests pin").
- **L7** Change-history wording "now"/"pre-change": fProxy/SparseSpMMTests.fProxy.cs:194-196 ("the exact scalar per-row loop ... used before BSR.spMM replaced it", "pre-change reference" - borderline: the A/B oracle needs some identification, but "OldStyle" + one sentence suffices), :89 and fProxy/SparseUnrollTests.fProxy.cs:18 ("ToBSRSymmetric now requires").
- **L8** Internal milestone labels: fProxy/SparseSymmetricTests.fProxy.cs:10,516 ("Milestone A"), fProxy/SparseTransposeTests.fProxy.cs:10 ("Milestone B"), fProxy/SparseEigenTests.fProxy.cs:295,395 ("Milestone C2/C3") - same class as the R-n/round labels the 2026-07-11 pass trimmed elsewhere.
- **L9** fProxy/UnsafeSortTests.fProxy.cs:14-25 - header keeps "before this file the kernel had ZERO direct coverage" (history) and a paragraph justifying template-vs-hand-written placement to a reviewer; the quoted testing policy itself is fine per the existing DEVLOG entry.
- **L10** fProxyPivotTests.cs:278 - stray Print.Log(vec) debug print inside PivotVecTest (logs on every suite run); :284 comment "[1, 0, 0, 0] -> [0, 0, 0, 1]" is misplaced (that transformation only happens after the later Swap(0,3); the first ApplyVec is a no-op swap of two zeros).
- **L11** fProxy/SolversTests.fProxy.cs:22-23 - dead enum members USolveIdentity/LSolveIdentity never dispatched or run; file/class name still says "Solvers" (retired class); the single QRSolve case duplicates coverage QRTests already has. Candidate for deletion/merge.
- **L12** fProxy/ResampleTests.fProxy.cs:14-15 - header claims "Catmull-Rom reproduces cubic polynomials exactly on a uniform grid", contradicting the correct in-body comment (:118-122) that it reproduces only up to degree 2 exactly. Fix the header word "cubic" to "quadratic".
- **L13** fProxy/SVDSolverTests.fProxy.cs:72-73 ("pinvSolve no longer modifies A; copies are kept for clarity") - "no longer" history wording; OPEN QUESTION for the maintainer: if SVD.pinvSolve/pseudoInverse genuinely no longer modify A, the ref A parameter (vs in A) is an API-surface inconsistency worth a production-side ruling (also echoed at SolverBatteryTests.fProxy.cs:470 and SVDWorkspaceTests.fProxy.cs:57).
- **L14** fProxy/SparseSymmetricTests.fProxy.cs:504 - MinresSymMatchesFull computes its dense reference with Krylov.cg, not Krylov.minres (looks like copy-paste from CgSymMatchesFull). Harmless (same SPD solution) but either use minres for symmetry or add a comment that the cross-algorithm check is intentional.
- **L15** iProxy/CompBitsTests.iProxy.cs:180 - "(adversarial-review addition)" reviewer-workflow tag; delete the parenthetical.
- **L16** iProxy/InitTest.iProxy.cs:15-29 - InitVecTestJob creates a Persistent arena and only calls Clear(), never Dispose(); the arena's own record-table memory leaks each run (the matrix twin below disposes correctly).
- **L17** fProxy/SpecialConstructorsTests.fProxy.cs:351-356 - HouseholderMat's trailing 2x2 case builds m = fProxyHouseholderMat(2, v) and asserts nothing (dead tail); also unused "using System.Diagnostics;" (:4, and in fProxyPivotTests.cs:4).
- **L18** fProxy/QRTests.fProxy.cs:509-561,834-841 - PrecisionReconstructTestJob (QRDecompErrorBenchRandom/Diagonal) accumulates avgError and never asserts any bound (only the NaN throw); the "Bench" name half-explains it, but as a test it pins nothing about error magnitude. Either assert a per-precision bound (Consts.fProxySqrtEps-scaled) or note the smoke-only intent.

## Addendum-pattern sweep results

1. Role-swapped InPlace wrappers: no test in this partition calls a kernel with swapped operand roles; mulInPlace/addScaledInPlace usages match kernel semantics (SparseCompNormsTests validates against dense).
2. Rename stragglers: found - M7 (MatrixMetrics), M8 (*Query_OP), L1/L2 (StatsOP/NormsOP), L11 (Solvers). No maxIter-to-maxIterations API stragglers (remaining maxIter identifiers are test locals, allowed).
3. Missing InPlace suffix: Eigen.valuesQR destructiveness (wide HIGH) is correctly worked around in this partition - SolverBatteryTests copies before calling (Fc = F.Copy()); no new instances.
4. [NoAlias] violations: none (tests deliberately alias only to assert guards throw).
5. Sibling-validation gaps: none new; guard tests here are consistent and thorough (Query/BSR/Krylov aliasing + dimension guards all exercised).
6. Literal type keywords surviving substitution: none harmful. fProxy templates use (fProxy)-cast literals or Consts per-type tokens; every //+choose tolerance block checked (SSOR/IC0/ILU0/SparseBSR/SparseSolver/SparseEigen/UKF) has correct float|double values in the correct order. iProxy files with negative literals correctly omit uint (3-way choose) or use //+skipFor[u] / 4-way choose where alsoExpand[uint] is present (AnalysisTests, ArenaWiringTests, CompBitsTests, HashTests - all consistent with the production surface they mirror).
7. Comment-policy debt: the bulk of findings above (M1-M6, L3-L9, L15).

Also checked per the brief: tests assert what their names claim (exceptions: M9, L18, L17); tolerances match the precision under test (house pattern Consts.fProxySqrtEps / per-type choose used throughout; the fixed 1E-4 / 1E-6-style absolute bounds in QR/LQ/SVD/Stats tests are loose-but-valid for double and never false-fail - no reverse leak of a double-scale eps into float found); type gating of tests matches the generated API types everywhere inspected.

## Areas confirmed clean (one line each)

- QRWorkspaceTests, QRLeastSquaresResidualTests, SVDRandomizedTests, SVDSubspaceTests, SVDWorkspaceTests, SvdFull/SvdRandomized/SvdThinValues workspace suites: contracts-only comments, per-type tolerances, correct guards.
- Sparse family (SSORTests, SparseIC0Tests, SparseILU0Tests, SparseCompNormsTests, SparseGalleryTests, SparseStructuralTests, SparseTransposeTests, SparseUnrollTests, SparseSpMMTests numerics): dense-oracle cross-checks sound, choose-tolerances correct, no logic issues.
- RandomTests/RandomMatrixTests/RandomWeightedTests/ResampleTests (numerics), RollingWindowTests (logic), StatsTests/TransformsTests (numerics), UKFTests (logic/numerics), UnsafeSortTests (oracle design), VectorCopyTests, SelectRefTests (both dtypes), TransposeTests (both dtypes), fProxyPivotTests (logic).
- Entire iProxy folder logic/numerics: exact-oracle integer tests are careful about short-width overflow, uint gating, and MinValue edges (NormsTests/StatsTests overflow pins and CompBitsTests width-driven oracles are exemplary); no correctness defects found.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 9     |
| LOW      | 18    |

No wrong-result/crash-level defects in the second half of the test templates. The dominant debt is pattern-7 comment policy: commit hashes, STAGE/Milestone/FIX-n/commit-n ticket codes, review-workflow tags, and "previously/used to/no longer" bug history - several in files whose siblings were already cleaned per DEVLOG (M1, M2), so a sibling-sweep of the 2026-07-11/12 relocation passes would close most of it. The only test-strength gaps are M9 (integer mat*vec values unasserted), L18 (assertion-free "ErrorBench" jobs), and L17 (dead Householder tail).
