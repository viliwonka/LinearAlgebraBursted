# Level-3 (GEMM) Optimization Opportunities

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

Scan of every dense kernel EXCEPT QR/LQ (both now blocked compact-WY, commits 4f04e76 / cc087c9).
Ranked by payoff/effort. Line numbers are from `Assets/LinearAlgebra/CodeGen/TemplateSource/OP`.

> **STATUS 2026-07-01.** SHIPPED since this scan: Cholesky POTRF (#2, d286658), LU GETRF (#4, 921bcc9),
> and `formT` (compact-WY LARFT) consolidated into `Unsafe_OP` (e7b8ff2) so reductions can reuse it.
> Also banked: accuracy sweep (7ef51bd) + small/non-square benchmarks (b7400f0). IN PROGRESS: SYTRD (#1,
> values path first). NEXT per owner ("full send"): SYTRD eigenvector variant → GEBRD (#6, values path
> first). Finding #6's original "off hot path" claim is CORRECTED below.

## Framing (two distinct wins)
- **GEMM-amortization (~3-4×):** even where level-2 already vectorizes (rank-1 axpy in Cholesky/LU),
  the trailing update re-streams the whole trailing matrix each step; blocking amortizes into GEMM.
- **Scalar→SIMD (larger):** where the level-2 core is a *reduction* Burst can't vectorize (the
  symmetric gemv in SYTRD, the `Mv` right-apply in bidiag) → blocking converts to GEMM/SYR2K = biggest
  relative win.
- **Note:** the whole family is capped by `matMatDot`'s own ~70 GFLOP/s untiled ceiling. Register-tiling
  `matMatDot` is a cross-cutting win that would lift EVERY blocked kernel (and is arguably the highest-
  leverage single perf task in the library).

Already level-3 (no action): svdRandomized, StatsOP.covarianceInto, kmeans GEMM-assignment.

## Ranked findings
1. **Symmetric tridiagonalization → blocked SYTRD (SYR2K)** — HIGH payoff / HIGH effort.
   `Eigen.fProxy.cs`: `eigenvaluesSymmetric` loop 446-505 (sym gemv 481-487, rank-2 update 497-502);
   `eigenSymmetric` 685-746 (+ Q accumulation 737-743). This is the DEFAULT fast symmetric eigensolver,
   O(n³), and its hot inner loop is a reduction-gemv (worst case). Reuses the QR/LQ compact-WY
   block-reflector machinery. Biggest intrinsic win; do after QR/LQ (which it now can build on).
2. **Cholesky → blocked POTRF (SYRK)** — HIGH payoff / MEDIUM effort — **best ratio, do first.**
   `Cholesky.fProxy.cs` `choleskyDecomposition` 75-97 (rank-1 trailing update 94-96). Textbook blocked
   right-looking POTRF; `matMatDotTransA` (SYRK) already exists. LEAVE `choleskyDecompositionPivot`
   157-288 unblocked (xPSTRF doesn't block cleanly).
3. **pseudoInverse / lowRankApprox → matMatDot** — MEDIUM payoff / LOW effort — **quick wins.**
   `SVD.Solvers.fProxy.cs` `pseudoInverse` 254-267 & 291-301; `SVD.LowRank.fProxy.cs` `lowRankApprox`
   700-710. Rank-1 outer-product accumulation → scale the k retained columns then one GEMM. Product is
   `X·Yᵀ`; no `transB` primitive exists → either add `matMatDotTransB` or transpose the k-wide factor
   once (O(nk), negligible).
4. **LU → blocked GETRF (TRSM + GEMM)** — HIGH payoff / HIGH effort.
   `LU.fProxy.cs` `luDecomposition` 137-148, `luDecompositionInpl` 213-224. Pivoting blocks fine
   (LAPACK), but deferred row-swaps + panel pivot search must stay bitwise-consistent with the
   solver/determinant paths.
5. **Hessenberg → blocked GEHRD** — MEDIUM payoff / HIGH effort.
   `Eigen.fProxy.cs` `eigenvaluesQR` elmhes 897-942; the strided column update 937-938 (stride-N
   scalar, SIMD-hostile) is a smaller independent cleanup worth doing regardless. Full fix = algorithm
   swap (elmhes → Householder), nonsymmetric-eigenvalue path only.
6. **Bidiag → GEBRD** — HIGH intrinsic / MEDIUM real payoff (ON THE HOT PATH) / HIGH effort.
   `Bidiag.fProxy.cs` `applyHouseholderRight` 106-122 (the `Mv` reduction).
   ⚠️ **CORRECTION 2026-07-01:** the original claim here ("svdThin/svdValues do NOT use Bidiag — wired
   only to tests") is WRONG — it conflated the `[Obsolete]` one-sided-Jacobi `svdDecomposition` with
   `svdThin`. Verified from code: `svdThin` Phase 1 = `Bidiag.bidiagonalize` (SVD.fProxy.cs:401, both
   alloc + workspace overloads); `svdValues` = `Bidiag.bidiagonalizeValues` (SVD.fProxy.cs:254). Both
   are on the SVD hot path, and every full-SVD consumer (Solvers/Subspace/lowRankApprox, and
   svdRandomized's small dense SVD) routes through `svdThin`. Only `svdTruncated` (GKL-Lanczos) genuinely
   does not use Bidiag. So GEBRD (compact-WY blocking of the bidiagonalization, LABRD-style) IS a real
   win: BEST on the VALUES path (bidiag ≈ 90% of `svdValues` runtime → the ~1.3× lands nearly whole ≈
   25-30%, clears a 20% bar); MARGINAL on `svdThin` alone (~10-15% — the un-blockable implicit-shift
   bidiagonal QR is co-dominant) but rides the same machinery and lifts every full-SVD consumer. GEBRD
   is the hardest reduction to block (interleaved left+right reflectors; each panel column needs
   on-the-fly matvec corrections), so ~half the flops recast as GEMM.
7. **powerIteration matvec** — LOW. `Eigen.fProxy.cs` 94-99 — route through `matVecDot` for
   consistency/vectorization (not a level-3 target; single-vector iteration).

## Suggested order
Original: Cholesky POTRF → pseudoInverse/lowRankApprox → SYTRD → LU GETRF → (defer GEHRD/GEBRD).
ACTUAL (2026-07-01): Cholesky ✅ → LU ✅ → formT consolidation ✅ → SYTRD (in progress) → GEBRD
values-path (next, corrected up from "deferred"). Still open/low-priority: pseudoInverse/lowRankApprox
GEMM routing (#3, quick), GEHRD (#5, nonsymmetric path). Register-tiling `matMatDot` remains the
highest-leverage cross-cutting perf task — it raises the ~70 GFLOP/s ceiling for EVERY blocked kernel
(all of the above are capped by it), and would convert the current ~1.3× gains toward the theoretical 2-3×.
