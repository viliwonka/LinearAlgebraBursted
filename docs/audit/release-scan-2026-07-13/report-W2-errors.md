# W2 - Error handling and exceptions (templates only)

Scope: Assets/LinearAlgebra/CodeGen/TemplateSource*. Read-only audit of the error-handling
dimension only (argument/size/allocator validation, numerical-failure surfacing, exception
messages, Burst-legality of throw paths, swallowed failures, test assertions vs live throws).

## House patterns confirmed

- Numerical failure in SOLVERS is surfaced via diagnostic structs carrying shared enums
  (IterativeSolveStatus / DirectSolveStatus, plus domain enums LQRStatus / MIPStatus), NOT by
  throwing. See OP/SolveStatus.cs, OP/SolveInfo.cs. Direct factorizations (LU/CHO/CHOP) return
  DirectSolveStatus.Singular / NotPositiveDefinite / Indefinite instead of throwing.
- Argument/shape validation IS thrown (ArgumentException / ArgumentOutOfRangeException /
  InvalidOperationException) with a "Method:" message prefix, at the top of every public entry
  point BEFORE any destructive work (LU.solveInPlace validates before decompInPlace destroys
  A_to_LU, at LU.fProxy.cs:608).
- Aliasing guards use a raw pointer compare + throw (z.Data.Ptr == r.Data.Ptr).
- Burst-legality: throw paths use string literals; the type-agnostic aliasing guard passes a
  literal who-string (OP/Krylov.Guards.cs). No catch blocks anywhere (grep clean).

## Findings

### 1. MEDIUM - Preconditioner factorizations throw on a NUMERICAL condition, diverging from the DirectSolveStatus convention
- Sparse/fProxyBlockJacobi.cs:132 - throw ArgumentException("fProxyBlockJacobi: diagonal block is singular");
- Sparse/fProxyILU0.cs:81 - throw ArgumentException("fProxyILU0: factorization broke down at every diagonal shift ...");
- Sparse/fProxyIC0.cs:120 - throw ArgumentException("fProxyIC0: factorization broke down at every diagonal shift ...");

The library already HAS a status vocabulary for exactly these conditions: the dense direct
factorizations return DirectSolveStatus.Singular / NotPositiveDefinite / Indefinite rather than
throwing. fProxyBlockJacobi.cs:125-132 even reads the bool status returned by LU.decompInPlace and
then DISCARDS it, converting a first-class singular status into an ArgumentException. Two problems:
(a) a numerically-singular/indefinite operator is a runtime data property, and ArgumentException
semantically claims the caller passed a malformed argument; (b) it is inconsistent with how every
dense solver surfaces the same condition. Concrete scenario: a Krylov user builds an
IC0/BlockJacobi preconditioner from an operator that turns out indefinite - instead of a
recoverable status they get an exception (and if the build ever runs inside a scheduled job, an
aborted job) where the dense CHO path would hand back NotPositiveDefinite.
Fix direction (maintainer judgment): give the preconditioner builders a status-return (mirroring
DirectSolveInfo), or at minimum switch the numerical-breakdown throws to InvalidOperationException
so they are not mislabeled as argument errors - and record the chosen convention in
Sparse/DEVLOG.md (no decision is documented there today).

### 2. LOW - fProxyBlockJacobi throws immediately on a singular diagonal block; sibling ILU0/IC0 retry with diagonal shifts first
- Sparse/fProxyBlockJacobi.cs:127-132 vs Sparse/fProxyILU0.cs:81 / Sparse/fProxyIC0.cs:120.

ILU0 and IC0 apply escalating diagonal shifts and only throw broke-down-at-every-shift as a last
resort; BlockJacobi throws on the first singular diagonal block with no robustness fallback. The
three sibling preconditioners present very different failure behaviour for the same diagonal-factor-
failed event. Fix direction: align the fallback strategy, or note the intentional difference in
Sparse/DEVLOG.md.

### 3. LOW - LU pivot-size guards drop the message-prefix used everywhere else in the file
- OP/LU.fProxy.cs:105, 328, 616, 696, 798, 826 - throw System.ArgumentException("pivot size must equal matrix dimension");

Every other validation message in LU.fProxy.cs (and in sibling CHOP/QR/LQRP files) is prefixed with
the method name, e.g. "decompSolve: pivot.N must equal b_to_x.N". These six pivot-size guards drop
the prefix, so a caught message cannot be traced to the failing entry point. Fix direction: prefix
them (decomp / decompInPlace / solveInPlace / solveInPlaceTransA).

### 4. LOW / informational - fProxyMPCState constructor throws on non-convergent terminal DARE
- OP/MPC.State.fProxy.cs:341-348 - throw ArgumentException("fProxyMPCState: terminal DARE did not converge -- (A,B) must be stabilizable");

Control.lqr surfaces non-convergence as LQRStatus.Diverged (a status), but the MPC state
CONSTRUCTOR must throw (a constructor cannot return a diagnostic struct), so this is defensible and
the disposal-before-throw cleanup (lines 343-347) is correct. Noted only for completeness; not a
release blocker.

## Areas checked and confirmed CLEAN

- Interpolated exception messages (Arena/ChunkedRecordTable.cs:158,250,262, OP/MIP.fProxy.cs:129)
  interpolate only int values (slotIndex, idx, Count, j). MIP.solve IS reachable from a
  [BurstCompile] job (MIPBenchmark.fProxy.cs:22-42), but the Burst string-formatting support handles
  primitive-int interpolation in exceptions, so these compile. NOT a Burst-legality defect.
- Validate-before-destroy ordering in the solveInPlace / decompSolve families across
  LU/CHO/CHOP/QR/QRCP/LQ/LQRP/SVD.Solvers - every one validates all shapes at the top before
  touching the destroyed buffers.
- Aliasing guards symmetric across all four preconditioners (IC0.cs:273, SSOR.cs:102, ILU0.cs:261,
  BlockJacobi.cs:190) and the sparse primitives (SparseOP.fProxy.cs spMV/spMM/spMVT/sweepLower/
  sweepUpper), plus FFT in-place (FFT.fProxy.cs rfft/irfft/dft must-not-alias guards).
- No swallowed failures - grep found no catch blocks, no return-default, and no empty
  return-new-Info in OP/; every status field is set from real control flow.
- Message correctness - spot-checked the full validation-throw surface of LU, CHO, CHOP, QP, MIP,
  LP, LQ, LQRP, QR, SVD.Solvers, SVD.Subspace, Kalman, Kalman.UKF, MPC, NLS, Optimize, FFT,
  Blas.Triangular, Statistics/StatsCore, ML/PCA, ML/KMeans, Gallery.*, Bidiag, Pivot: method-name
  prefixes match the enclosing method and the row/col dimension in each message is correct (no
  swapped rows/cols found).
- Test assertions match live throws - the Assert.Throws sites in the test templates (RandomSharedTests,
  BoolRandomTests, BoolHashTests, Bridge*, PCATests, BidiagWorkspaceTests, ClampTests,
  ArenaWiringTests) correspond to validations the current API still performs; no assertion targets a
  type/message the API no longer throws.
- Asymmetric coverage is intentional, not a missing guard - StatsCore.iProxy.cs exposes only the
  array/vector reductions (with the same empty-array guards as the fProxy sibling); the matrix
  row/col reductions and covariance/correlation are float-only by the integer-surface policy.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0 |
| MEDIUM   | 1 |
| LOW      | 3 (incl. 1 informational) |

The error-handling dimension is in strong shape: validation is uniformly method-prefixed,
dimensionally correct, placed before destructive work, and Burst-legal; numerical failure in the
solver core is consistently status-based; tests track the live throw surface. The only substantive
item is the preconditioner-build family surfacing numerical singular/indefinite breakdown as
ArgumentException (Finding 1) instead of via the status convention used everywhere else.
