# DEVLOG — Interfaces
Code comments state contracts only; history lives here (see CLAUDE.md).

## IfProxyPreconditioner — IsSpd / IsConstant compatibility flags
- 2026-07-22 | After the per-preconditioner overload collapse widened every solver to accept any
  `IfProxyPreconditioner`, added two self-describing flags so solvers reject an incompatible
  preconditioner at entry (uniform runtime check, NOT a marker interface — chosen because AMG is
  only CONDITIONALLY SPD (V-cycle yes, K-cycle no), so a static marker can't express it, and because
  the flags cover the raw-operator path a marker can't). `IsSpd` = symmetric positive-definite M;
  `IsConstant` = fixed operator each iteration (false only for a variable one, e.g. AMG K-cycle).
  Both are compile-time-constant on the static preconditioners, so Burst constant-folds the checks
  to zero cost there (like `IsIdentity`); the real check survives only for AMG. Solver requirements:
  cg/minres/minresQLP/bcg/bcgrq/bfbcg/bminres need SPD∧Constant; fcg + lobpcg need SPD (lobpcg lenient
  on constancy — block iteration tolerates a varying M, enabling K-cycle-AMG-preconditioned LOBPCG);
  gmres/gcrodr/biCGStab/idr/tfqmr + block twins need Constant; fgmres/bfgmres none. AMG's bespoke
  `IsCycleSymmetric` throw was replaced by the uniform mechanism (`IsSpd=IsCycleSpd`,
  `IsConstant=IsCycleConstant`; `IsCycleSymmetric` kept as their AND). This is how PETSc/Eigen handle
  it (runtime + docs) — we just make the check explicit. Retired
  SparseSolverTests.PcgNonSpdPreconditionerBreaksDown (its non-SPD-M → graceful-breakdown scenario is
  superseded by the entry rejection, covered by the new managed fProxyPreconditionerCompatibilityTests).

## ResidualFunction.fProxy.cs
- 2026-07-12 | NEW file: IfProxyResidualFunction/IfProxyResidualJacobian (Optimize.nlsSolve),
  IfProxyRobustLoss + fProxyL2Loss/fProxyHuberLoss/fProxyCauchyLoss/fProxyTukeyLoss (shared,
  standalone -- designed for a FUTURE linear-IRLS facade to reuse, not just NLS), and
  IfProxyCurveModel (Optimize.curveFit). Full provenance/validation history is in OP/DEVLOG.md's NLS
  section (the interfaces themselves carry no algorithm, just the functor shape).
  IfProxyResidualJacobian extends IfProxyResidualFunction (adds Jacobian) mirroring
  IfProxyScalarDerivativeFunction's own relationship to IfProxyScalarFunction just above.

## LinearOperator.fProxy.cs
- 2026-07-19 | `fProxyIdentityPreconditioner.Apply` / `fProxyIdentityOperator.Apply`/`ApplyT` switched
  from `z.Data.CopyFrom(r.Data)` to the length-checked `z.CopyFrom(in r)` (fProxy/DEVLOG.md's
  silent-resize fix) -- same-size by contract (both N-sized), no observable bug, defense-in-depth.
- 2026-07-11 | `IfProxyLinearOperator.ApplyDot`'s doc comment used to spend most of its length on a
  fusion-attempt post-mortem (Krylov R2, docs/draft-spec-krylov-optimization.md): an earlier
  version genuinely fused the dot-product reduction into the dense/BSR Apply kernels, but that was
  measurably SLOWER -- see `LinearAlgebra.Sparse.BSR.spMVDot`'s doc comment for the A/B numbers and
  root cause. Root cause: the fused kernel's cross-row dot fold couldn't reuse `vecDot`'s SIMD
  accumulator pattern, and lost to just calling vecDot separately. So ApplyDot exists purely for a
  clean single call site in cg/pcg (and any future solver), not because fusion pays for itself.
  (was Interfaces\LinearOperator.fProxy.cs:27-40)
- 2026-07-11 | `fProxyDenseOperator.ApplyDot`'s implementation comment repeated the same
  fusion-was-tried-and-slower story a second time as an inline comment; same history as above.
  (was Interfaces\LinearOperator.fProxy.cs:95-97)
- 2026-07-11 | `fProxyColScaledOperator`'s doc comment used to include a longer tutorial-style
  explanation of column-equilibration preconditioning: with d[j] = 1/‖A_:,j‖₂ (an AᵀA-Jacobi /
  column-equilibration preconditioner, built via `Blas.columnNormsSquared` +
  `Blas.buildJacobiScale`), the scaled operator A·D has a unit-diagonal normal matrix, so an
  ill-conditioned least-squares problem converges in fewer iterations. It also spelled out that
  this composes with every generic solver with no solver change (already generic over `TOp`), so
  passing `fProxyColScaledOperator<fProxyDenseOperator>` (or the BSR operator) turns cgls/lsqr/lsmr
  into their column-preconditioned variants. (was Interfaces\LinearOperator.fProxy.cs:168-185)
- 2026-07-11 | `fProxyIdentityOperator`'s doc comment used to argue in detail why an exact bit-copy
  Apply/ApplyT reproduces the Euclidean-only LOBPCG formula bit-for-bit when B is this operator,
  pointing at LOBPCG's own "B=I strategy" doc note for the full argument. (was
  Interfaces\LinearOperator.fProxy.cs:123-131)
