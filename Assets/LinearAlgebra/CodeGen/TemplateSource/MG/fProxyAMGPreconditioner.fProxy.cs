using LinearAlgebra.Sparse;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// An <see cref="fProxyAMG"/> hierarchy used as a preconditioner: Apply(r, z) runs exactly one
    /// symmetric V-cycle solving A z = r from a zero initial guess. With pre == post smoothing sweeps
    /// and the symmetric Chebyshev smoother the cycle is a fixed SPD linear operator, so this is valid
    /// for <see cref="LinearAlgebra.Krylov"/>.pcg. The wrapped hierarchy must outlive the
    /// preconditioner; its buffers are shared (not copied), so a single instance is not safe for
    /// concurrent Apply.
    /// </summary>
    public readonly struct fProxyAMGPreconditioner : IfProxyPreconditioner
    {
        readonly fProxyAMG _amg;

        public fProxyAMGPreconditioner(in fProxyAMG amg) { _amg = amg; }

        public int Rows => _amg.Rows;

        /// <summary>z = one V-cycle on A z = r, zero initial guess. z must not alias r.</summary>
        public void Apply(in fProxyN r, ref fProxyN z) => _amg.ApplyCycleFromZero(in r, ref z);
    }
}

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// Preconditioned Conjugate Gradient over a BSR SPD matrix with an AMG-V-cycle preconditioner.
        /// Same three-rung BSR convenience pattern as the block-Jacobi / SSOR / IC0 / FSAI / Chebyshev
        /// / additive-Schwarz overloads. Valid only when the hierarchy's cycle is symmetric (pre ==
        /// post) — the standard SPD-preconditioner requirement for pcg.
        /// </summary>
        public static SolveInfo pcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pcg(new Sparse.fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// AMG-preconditioned CG over a BSR SPD matrix — allocates four scratch vectors from the arena
        /// and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// AMG-preconditioned CG over a BSR SPD matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x)
        {
            return pcg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
