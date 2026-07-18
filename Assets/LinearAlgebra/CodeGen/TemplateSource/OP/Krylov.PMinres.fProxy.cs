using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// Preconditioned MINRES solver -- allocates eight scratch vectors from the arena and
        /// calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN y  = b.fProxyTempVec(A.Rows);
            fProxyN r1 = b.fProxyTempVec(A.Rows);
            fProxyN r2 = b.fProxyTempVec(A.Rows);
            fProxyN v  = b.fProxyTempVec(A.Rows);
            fProxyN w  = b.fProxyTempVec(A.Rows);
            fProxyN w1 = b.fProxyTempVec(A.Rows);
            fProxyN w2 = b.fProxyTempVec(A.Rows);
            fProxyN z  = b.fProxyTempVec(A.Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned MINRES solver with default maxIter (A.Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            return minres(in A, in M, in b, ref x, A.Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching block-Jacobi
        /// preconditioner. Forwards into <see cref="minres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned MINRES over a BSR matrix -- allocates eight scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows)
        /// and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching SSOR
        /// preconditioner. Forwards into <see cref="minres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// overloads above.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// SSOR Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors from
        /// the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// SSOR Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching block IC(0)
        /// preconditioner. Forwards into <see cref="minres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and SSOR overloads above.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// IC(0) Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors from
        /// the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// IC(0) Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching FSAI
        /// preconditioner. Forwards into <see cref="minres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and IC0 overloads above. FSAI's local SPD solves need A[J,J] SPD; on an indefinite A
        /// build may fall back to shifted rows (same practical caveat IC0 already carries on
        /// minres).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// FSAI Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors from
        /// the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// FSAI Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching Chebyshev
        /// preconditioner. Forwards into <see cref="minres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi,
        /// SSOR, and IC0 overloads above.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Chebyshev Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors
        /// from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Chebyshev Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and
        /// tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching symmetric
        /// additive-Schwarz preconditioner. Forwards into <see cref="minres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and IC0 overloads above. AS is SPD whenever its build reports Success, so it is a valid
        /// MINRES preconditioner; restricted Schwarz (RAS) is NOT symmetric and has no minres rung.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Additive-Schwarz Preconditioned MINRES over a BSR matrix -- allocates eight scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Additive-Schwarz Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows)
        /// and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
