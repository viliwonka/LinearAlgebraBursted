using System;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// An <see cref="fProxyAMG"/> hierarchy used as a preconditioner: Apply(r, z) runs exactly one
    /// cycle solving A z = r from a zero initial guess. A symmetric V-cycle (pre == post) is a fixed
    /// SPD operator — pair it with <see cref="LinearAlgebra.Krylov"/>.pcg. A K-cycle is a VARIABLE
    /// operator (inner Krylov acceleration) — pair it with <see cref="LinearAlgebra.Krylov"/>.fcg; pcg
    /// is invalid for it (and the pcg overloads reject it). The wrapped hierarchy must outlive the
    /// preconditioner; its buffers are shared (not copied), so a single instance is not safe for
    /// concurrent Apply.
    /// </summary>
    public readonly struct fProxyAMGPreconditioner : IfProxyPreconditioner
    {
        readonly fProxyAMG _amg;

        public fProxyAMGPreconditioner(in fProxyAMG amg) { _amg = amg; }

        public int Rows => _amg.Rows;
        /// <summary>True iff the cycle is a fixed SPD operator (symmetric V-cycle) valid for pcg.</summary>
        public bool IsCycleSymmetric => _amg.IsCycleSymmetric;

        /// <summary>z = one cycle on A z = r, zero initial guess. z must not alias r.</summary>
        public bool IsIdentity => false;

        public void Apply(in fProxyN r, ref fProxyN z) => _amg.ApplyCycleFromZero(in r, ref z);
    }
}

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// Preconditioned Conjugate Gradient over a BSR SPD matrix with an AMG preconditioner. Same
        /// three-rung BSR convenience pattern as the block-Jacobi / SSOR / IC0 / FSAI / Chebyshev /
        /// additive-Schwarz overloads. Requires a fixed SPD preconditioner — a SYMMETRIC V-cycle
        /// (AMGOptions.pre == post, cycle == V); a K-cycle is variable and rejected here (use
        /// <see cref="fcg(in Sparse.fProxyBSR, in Sparse.fProxyAMGPreconditioner, in fProxyN, ref fProxyN, int, fProxy)"/>).
        /// </summary>
        public static SolveInfo pcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            if (!M.IsCycleSymmetric)
                throw new ArgumentException("Krylov.pcg: the AMG preconditioner is not a fixed SPD operator (needs a symmetric V-cycle, pre == post); use Krylov.fcg for a K-cycle or asymmetric cycle");
            return pcg(new Sparse.fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>AMG-preconditioned CG over a BSR SPD matrix — allocates four scratch vectors.</summary>
        public static SolveInfo pcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            if (!M.IsCycleSymmetric)
                throw new ArgumentException("Krylov.pcg: the AMG preconditioner is not a fixed SPD operator (needs a symmetric V-cycle, pre == post); use Krylov.fcg for a K-cycle or asymmetric cycle");
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>AMG-preconditioned CG with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo pcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x)
        {
            return pcg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Flexible-CG-accelerated AMG over a BSR SPD matrix — the correct pairing for a K-cycle
        /// preconditioner (variable operator). Also valid for a V-cycle. Same convenience-rung pattern
        /// as the pcg overloads (fcg's fifth scratch vector rOld holds the previous residual).
        /// </summary>
        public static SolveInfo fcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z, ref fProxyN rOld,
                               int maxIter, fProxy tol)
        {
            return fcg(new Sparse.fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, ref rOld, maxIter, tol);
        }

        /// <summary>AMG-preconditioned Flexible CG over a BSR SPD matrix — allocates five scratch vectors.</summary>
        public static SolveInfo fcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            return fcg(new Sparse.fProxyBSROperator(in A), in M, in b, ref x, maxIter, tol);
        }

        /// <summary>AMG-preconditioned Flexible CG with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo fcg(in Sparse.fProxyBSR A, in Sparse.fProxyAMGPreconditioner M, in fProxyN b, ref fProxyN x)
        {
            return fcg(new Sparse.fProxyBSROperator(in A), in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
