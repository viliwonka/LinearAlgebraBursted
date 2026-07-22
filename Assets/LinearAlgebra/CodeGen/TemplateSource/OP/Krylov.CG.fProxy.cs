using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b,
        /// generic over any <see cref="IfProxyLinearOperator"/>. This is the single implementation of
        /// the CG loop — the concrete dense (<c>cg(in fProxyMxN, ...)</c>) and BSR
        /// (<c>cg(in fProxyBSR, ...)</c>) overloads below are thin forwarders that
        /// wrap their matrix in <see cref="fProxyDenseOperator"/> / <c>fProxyBSROperator</c> and
        /// call this method.
        ///
        /// Caller provides x (initial guess, overwritten with solution — WARM-STARTABLE: seed x
        /// with a previous solution to resume/refine) and three scratch vectors r, p, Ap (all
        /// length A.Rows). Returns a <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖) — see that struct
        /// for the implicit-bool/status/undefined-x contract. Breakdown on non-positive curvature
        /// p·Ap &lt;= 0 (A not SPD or numerical breakdown).
        /// </summary>
        public static SolveInfo cg<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                   ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                   int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            // Unpreconditioned CG == the single cg<TOp,TPre> body with the identity preconditioner:
            // its IsIdentity fold strips every z access, so this compiles to plain CG and needs no z.
            fProxyN z = default;
            return cg(in A, default(fProxyIdentityPreconditioner), in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// CG over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="cg{TOp}"/> via <see cref="fProxyDenseOperator"/>. See that method for the
        /// actual loop and buffer semantics.
        /// </summary>
        public static SolveInfo cg(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                             ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                             int maxIter, fProxy tol)
        {
            return cg(new fProxyDenseOperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIter, tol);
        }

        /// <summary>
        /// Conjugate Gradient solver — allocates three scratch vectors from the arena and calls
        /// the zero-alloc primitive. x is overwritten with the solution on convergence.
        /// </summary>
        public static SolveInfo cg(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                             int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in b, ref x, ref r, ref p, ref Ap, maxIter, tol);
        }

        /// <summary>
        /// Conjugate Gradient solver with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps). x is overwritten with the solution on convergence.
        /// </summary>
        public static SolveInfo cg(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix. Same semantics as
        /// the dense overload — see <see cref="cg(in fProxyMxN, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>.
        /// Forwards into <see cref="cg{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                             ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                             int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIter, tol);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix — allocates three
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                             int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in b, ref x, ref r, ref p, ref Ap, maxIter, tol);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc Conjugate Gradient for SPD systems A x = b, generic over BOTH the operator
        /// (<see cref="IfProxyLinearOperator"/>) and the preconditioner
        /// (<see cref="IfProxyPreconditioner"/>) -- the SINGLE body behind the plain and the
        /// preconditioned entry points. Every z access (its scratch, size/aliasing guards, M.Apply,
        /// and the ⟨r,z⟩ dot) sits behind <c>if (!M.IsIdentity)</c>, a compile-time-literal branch
        /// that constant-folds per Burst specialization -- so with
        /// <see cref="fProxyIdentityPreconditioner"/> this compiles to, and is bit-identical to,
        /// plain CG (no Apply, no z traffic), and z may be passed as <c>default</c>.
        ///
        /// Caller provides x (initial guess, overwritten -- warm-startable) and scratch r, p, Ap, z
        /// (length A.Rows; z UNUSED under the identity). Convergence tests the true residual ‖r‖²
        /// against tol²·‖b‖². Returns a <see cref="SolveInfo"/>. Breakdown on non-positive curvature
        /// p·Ap ≤ 0 (or a non-SPD preconditioner's non-positive ⟨r,z⟩).
        /// </summary>
        public static SolveInfo cg<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                                          int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("cg: A must be square");
            if (b.N != A.Rows)
                throw new ArgumentException("cg: b.N must equal A.Rows");
            if (x.N != A.Rows)
                throw new ArgumentException("cg: x.N must equal A.Rows");
            if (r.N != A.Rows)
                throw new ArgumentException("cg: r.N must equal A.Rows");
            if (p.N != A.Rows)
                throw new ArgumentException("cg: p.N must equal A.Rows");
            if (Ap.N != A.Rows)
                throw new ArgumentException("cg: Ap.N must equal A.Rows");
            if (!M.IsIdentity && z.N != A.Rows)
                throw new ArgumentException("cg: z.N must equal A.Rows");
            if (maxIter < 1)
                throw new ArgumentException("cg: maxIter must be >= 1");

            // Aliasing guard -- see cg<TOp>. z joins the checked set only for a real preconditioner
            // (the identity path never dereferences z, and may pass default).
            unsafe
            {
                int n = M.IsIdentity ? 5 : 6;
                long* ptrs = stackalloc long[6];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)p.Data.Ptr; ptrs[2] = (long)Ap.Data.Ptr;
                ptrs[3] = (long)x.Data.Ptr; ptrs[4] = (long)b.Data.Ptr;
                if (!M.IsIdentity) ptrs[5] = (long)z.Data.Ptr;
                RequireDistinctBuffers("cg: r/p/Ap/z/x/b must be distinct", ptrs, n);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref Ap);
            r.CopyFrom(in b);
            r.addScaledInPlace((fProxy)(-1), Ap);

            fProxy threshold = tol * tol * bb;

            fProxy rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            // p = z = M⁻¹r ; rzold = ⟨r,z⟩. Identity folds to p = r, rzold = ⟨r,r⟩ (== plain CG).
            fProxy rzold;
            if (M.IsIdentity)
            {
                p.CopyFrom(in r);
                rzold = rr;
            }
            else
            {
                M.Apply(in r, ref z);
                p.CopyFrom(in z);
                rzold = Blas.dot(r, z);

                // Non-SPD preconditioner guard (identity's ⟨r,r⟩ is always > 0 here).
                if (!(rzold > (fProxy)0))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, 0, math.sqrt(rr));
            }

            for (int k = 0; k < maxIter; k++)
            {
                fProxy pAp = A.ApplyDot(in p, ref Ap);

                if (!(pAp > (fProxy)0))                  // NaN-safe: also catches breakdown
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                fProxy alpha = rzold / pAp;

                // x += alpha p ; r -= alpha Ap ; rr = ‖r‖² folded into the r-update pass.
                rr = Blas.updateXR(alpha, p, ref x, Ap, ref r);
                if (rr <= threshold)
                {
                    // Verify-at-exit -- see cg<TOp>'s matching block for the rationale.
                    rr = VerifyTrueResidual(in A, in b, in x, ref Ap, ref r);

                    if (rr <= threshold)
                        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));
                }

                fProxy rznew;
                if (M.IsIdentity)
                {
                    rznew = rr;                          // ⟨r,z⟩ = ⟨r,r⟩, already in hand
                }
                else
                {
                    M.Apply(in r, ref z);                 // z = M⁻¹r
                    rznew = Blas.dot(r, z);
                    if (!(rznew > (fProxy)0))             // NaN-safe: same breakdown guard, fresh ⟨r,z⟩
                        return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(rr));
                }

                fProxy beta = rznew / rzold;

                if (M.IsIdentity)
                    p.scaleAddInPlace(beta, r);              // p = beta p + r
                else
                    p.scaleAddInPlace(beta, z);              // p = beta p + z

                rzold = rznew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(rr));
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient — allocates four scratch vectors from the arena and
        /// calls the merged zero-alloc <see cref="cg{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>.
        /// </summary>
        public static SolveInfo cg<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN r  = b.fProxyTempVec(A.Rows);
            fProxyN p  = b.fProxyTempVec(A.Rows);
            fProxyN Ap = b.fProxyTempVec(A.Rows);
            fProxyN z  = b.fProxyTempVec(A.Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver with default maxIter (A.Rows) and
        /// tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            return cg(in A, in M, in b, ref x, A.Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with ANY
        /// <see cref="IfProxyPreconditioner"/> (block-Jacobi/SSOR/IC0/FSAI/Chebyshev/additive-Schwarz).
        /// Forwards into <see cref="cg{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo cg<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a BSR SPD matrix with ANY
        /// <see cref="IfProxyPreconditioner"/> (block-Jacobi/SSOR/IC0/FSAI/Chebyshev/additive-Schwarz)
        /// -- allocates four scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a BSR SPD matrix with ANY
        /// <see cref="IfProxyPreconditioner"/> (block-Jacobi/SSOR/IC0/FSAI/Chebyshev/additive-Schwarz),
        /// with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
