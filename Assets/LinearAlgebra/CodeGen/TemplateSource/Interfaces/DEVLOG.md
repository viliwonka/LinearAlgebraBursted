# DEVLOG — Interfaces
Code comments state contracts only; history lives here (see CLAUDE.md).

## LinearOperator.fProxy.cs
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
