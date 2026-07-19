using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// ILU(0)-preconditioned BiCGSTAB over a block-sparse (BSR) matrix — forwards into
        /// <see cref="biCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            fProxyN pHat  = b.fProxyTempVec(A.M_Rows);
            fProxyN sHat  = b.fProxyTempVec(A.M_Rows);
            return biCGStab(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat,
                             maxIter, tol);
        }

        /// <summary>ILU(0) BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// SPAI-preconditioned BiCGSTAB over a block-sparse (BSR) matrix -- forwards into
        /// <see cref="biCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>. SPAI is NOT symmetric
        /// (even for symmetric A) and is not a valid CG/MINRES preconditioner -- biCGStab is its
        /// only Krylov rung, mirroring ILU0's placement.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxySPAI M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            fProxyN pHat  = b.fProxyTempVec(A.M_Rows);
            fProxyN sHat  = b.fProxyTempVec(A.M_Rows);
            return biCGStab(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat,
                             maxIter, tol);
        }

        /// <summary>SPAI BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxySPAI M, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Restricted additive-Schwarz (RAS) preconditioned BiCGSTAB over a block-sparse (BSR)
        /// matrix -- forwards into <see cref="biCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// RAS is NOT symmetric (even for symmetric A) and is not a valid CG/MINRES preconditioner --
        /// biCGStab is its only Krylov rung, mirroring ILU0's and SPAI's placement.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyRestrictedSchwarz M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            fProxyN pHat  = b.fProxyTempVec(A.M_Rows);
            fProxyN sHat  = b.fProxyTempVec(A.M_Rows);
            return biCGStab(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat,
                             maxIter, tol);
        }

        /// <summary>RAS BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyRestrictedSchwarz M, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
