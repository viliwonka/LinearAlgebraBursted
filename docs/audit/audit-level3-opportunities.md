# Level-3 (GEMM) Optimization Opportunities

Scan of every dense kernel EXCEPT QR/LQ (both now blocked compact-WY, commits 4f04e76 / cc087c9).
Ranked by payoff/effort. Line numbers are from `Assets/LinearAlgebra/CodeGen/TemplateSource/OP`.

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
6. **Bidiag → GEBRD** — HIGH intrinsic / LOW real payoff (OFF HOT PATH) / HIGH effort.
   `Bidiag.fProxy.cs` `applyHouseholderRight` 106-122 (the `Mv` reduction). BUT svdThin/svdValues/
   svdTruncated do NOT use `Bidiag.bidiagonalize` — it's wired only to its own tests. Defer until it's
   on a shipping path.
7. **powerIteration matvec** — LOW. `Eigen.fProxy.cs` 94-99 — route through `matVecDot` for
   consistency/vectorization (not a level-3 target; single-vector iteration).

## Suggested order
Cholesky POTRF → pseudoInverse/lowRankApprox GEMM routing → symmetric SYTRD → LU GETRF →
(defer GEHRD/GEBRD). Consider register-tiling `matMatDot` first if a broad perf lift is wanted — it
raises the ceiling for QR/LQ (already blocked) and every kernel above.
