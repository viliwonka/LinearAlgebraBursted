using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// Right-preconditioned BiCGSTAB (van der Vorst): solves A x = b for a general square A
        /// with preconditioner M ≈ A (M⁻¹ applied via <typeparamref name="TPre"/>). Same contract
        /// as <see cref="biCGStab{TOp}"/> — warm-startable x, true-residual convergence test,
        /// breakdown statuses — plus two extra scratch vectors pHat/sHat for the preconditioned
        /// directions.
        /// </summary>
        public static SolveInfo pbiCGStab<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                         ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                         ref fProxyN pHat, ref fProxyN sHat,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("pbiCGStab: A must be square");
            if (b.N != A.Rows || x.N != A.Rows || r.N != A.Rows || rHat0.N != A.Rows ||
                p.N != A.Rows || v.N != A.Rows || t.N != A.Rows || pHat.N != A.Rows || sHat.N != A.Rows)
                throw new ArgumentException("pbiCGStab: all vectors must have length A.Rows");
            if (maxIter < 1)
                throw new ArgumentException("pbiCGStab: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)rHat0.Data.Ptr; ptrs[2] = (long)p.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr; ptrs[4] = (long)t.Data.Ptr; ptrs[5] = (long)x.Data.Ptr;
                ptrs[6] = (long)b.Data.Ptr; ptrs[7] = (long)pHat.Data.Ptr; ptrs[8] = (long)sHat.Data.Ptr;
                RequireDistinctBuffers("pbiCGStab: r/rHat0/p/v/t/x/b/pHat/sHat must be distinct", ptrs, 9);
            }

            fProxy bb = Blas.dot(b, b);
            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            A.Apply(in x, ref v);
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((fProxy)(-1), v);

            fProxy threshold = tol * tol * bb;
            fProxy rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            rHat0.Data.CopyFrom(r.Data);
            for (int i = 0; i < A.Rows; i++) { p[i] = (fProxy)0; v[i] = (fProxy)0; }

            fProxy rho = (fProxy)1, alpha = (fProxy)1, omega = (fProxy)1;

            for (int k = 0; k < maxIter; k++)
            {
                fProxy rhoNew = Blas.dot(rHat0, r);
                if (rhoNew == (fProxy)0 || math.isnan(rhoNew))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                fProxy beta = (rhoNew / rho) * (alpha / omega);
                p.addScaledInPlace(-omega, v);
                p.scaleAddInPlace(beta, r);

                M.Apply(in p, ref pHat);                     // pHat = M^-1 p
                A.Apply(in pHat, ref v);                     // v = A pHat

                fProxy rv = Blas.dot(rHat0, v);
                if (rv == (fProxy)0 || math.isnan(rv))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                alpha = rhoNew / rv;

                fProxy ss = Blas.axpyNormSq(-alpha, v, ref r);   // r := s
                if (ss <= threshold)
                {
                    x.addScaledInPlace(alpha, pHat);
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(ss));
                }

                M.Apply(in r, ref sHat);                     // sHat = M^-1 s
                A.Apply(in sHat, ref t);                     // t = A sHat

                fProxy tt = Blas.dot(t, t);
                if (!(tt > (fProxy)0))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                omega = Blas.dot(t, r) / tt;
                if (omega == (fProxy)0 || math.isnan(omega))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                x.addScaledInPlace(alpha, pHat);
                x.addScaledInPlace(omega, sHat);

                rr = Blas.axpyNormSq(-omega, t, ref r);      // r := s - omega t
                if (rr <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));

                rho = rhoNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(rr));
        }

        /// <summary>
        /// ILU(0)-preconditioned BiCGSTAB over a block-sparse (BSR) matrix — forwards into
        /// <see cref="pbiCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo pbiCGStab(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            fProxyN pHat  = b.fProxyTempVec(A.M_Rows);
            fProxyN sHat  = b.fProxyTempVec(A.M_Rows);
            return pbiCGStab(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat,
                             maxIter, tol);
        }

        /// <summary>ILU(0) BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo pbiCGStab(in fProxyBSR A, in fProxyILU0 M, in fProxyN b, ref fProxyN x)
        {
            return pbiCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// SPAI-preconditioned BiCGSTAB over a block-sparse (BSR) matrix -- forwards into
        /// <see cref="pbiCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>. SPAI is NOT symmetric
        /// (even for symmetric A) and is not a valid CG/MINRES preconditioner -- pbiCGStab is its
        /// only Krylov rung, mirroring ILU0's placement.
        /// </summary>
        public static SolveInfo pbiCGStab(in fProxyBSR A, in fProxySPAI M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            fProxyN pHat  = b.fProxyTempVec(A.M_Rows);
            fProxyN sHat  = b.fProxyTempVec(A.M_Rows);
            return pbiCGStab(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat,
                             maxIter, tol);
        }

        /// <summary>SPAI BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo pbiCGStab(in fProxyBSR A, in fProxySPAI M, in fProxyN b, ref fProxyN x)
        {
            return pbiCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Restricted additive-Schwarz (RAS) preconditioned BiCGSTAB over a block-sparse (BSR)
        /// matrix -- forwards into <see cref="pbiCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// RAS is NOT symmetric (even for symmetric A) and is not a valid CG/MINRES preconditioner --
        /// pbiCGStab is its only Krylov rung, mirroring ILU0's and SPAI's placement.
        /// </summary>
        public static SolveInfo pbiCGStab(in fProxyBSR A, in fProxyRestrictedSchwarz M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            fProxyN pHat  = b.fProxyTempVec(A.M_Rows);
            fProxyN sHat  = b.fProxyTempVec(A.M_Rows);
            return pbiCGStab(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat,
                             maxIter, tol);
        }

        /// <summary>RAS BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo pbiCGStab(in fProxyBSR A, in fProxyRestrictedSchwarz M, in fProxyN b, ref fProxyN x)
        {
            return pbiCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
