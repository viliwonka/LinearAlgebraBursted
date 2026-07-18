using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // Shared factory for the square-solver diagnostics struct (cg/minres/pminres/
        // biCGStab). rnorm is normally a value the solver already holds -- a tracked
        // residual norm, or a single dot on its live residual r -- never a fresh A*x. EXCEPTION:
        // cg verify a claimed Converged exit with one fresh r = b-Ax before trusting it,
        // so rnorm on that path is the verified value. pminres does the same, PLUS one fresh r on
        // a MaxIterations exit -- its recursively tracked phibar is the M⁻¹-weighted residual
        // once preconditioned, not ‖b-Ax‖. minres/biCGStab do not do this verification.
        static SolveInfo MakeSolveInfo(IterativeSolveStatus status, int iterations, fProxy rnorm)
            => new SolveInfo { rnorm = rnorm, iterations = iterations, status = status };

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
                fProxy* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == xPtr || ApPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("cg: r/p/Ap/x/b must be distinct");

                if (!M.IsIdentity)
                {
                    fProxy* zPtr = z.Data.Ptr;
                    if (zPtr == rPtr || zPtr == pPtr || zPtr == ApPtr || zPtr == xPtr || zPtr == bPtr)
                        throw new ArgumentException("cg: z must be distinct from r/p/Ap/x/b");
                }
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref Ap);
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((fProxy)(-1), Ap);

            fProxy threshold = tol * tol * bb;

            fProxy rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            // p = z = M⁻¹r ; rzold = ⟨r,z⟩. Identity folds to p = r, rzold = ⟨r,r⟩ (== plain CG).
            fProxy rzold;
            if (M.IsIdentity)
            {
                p.Data.CopyFrom(r.Data);
                rzold = rr;
            }
            else
            {
                M.Apply(in r, ref z);
                p.Data.CopyFrom(z.Data);
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
                    A.Apply(in x, ref Ap);
                    r.Data.CopyFrom(b.Data);
                    r.addScaledInPlace((fProxy)(-1), Ap);
                    rr = Blas.dot(r, r);

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
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching block-Jacobi preconditioner. Forwards into <see cref="cg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix — allocates four
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching SSOR preconditioner. Forwards into <see cref="cg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// overloads above.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// SSOR Preconditioned Conjugate Gradient over a BSR SPD matrix -- allocates four scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// SSOR Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching block IC(0) preconditioner. Forwards into <see cref="cg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and SSOR overloads above.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// IC(0) Preconditioned Conjugate Gradient over a BSR SPD matrix -- allocates four scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// IC(0) Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching FSAI preconditioner. Forwards into <see cref="cg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi,
        /// SSOR, and IC0 overloads above.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// FSAI Preconditioned Conjugate Gradient over a BSR SPD matrix -- allocates four scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// FSAI Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching Chebyshev preconditioner. Forwards into <see cref="cg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi,
        /// SSOR, and IC0 overloads above.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Chebyshev Preconditioned Conjugate Gradient over a BSR SPD matrix -- allocates four
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Chebyshev Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its matching
        /// symmetric additive-Schwarz preconditioner. Forwards into <see cref="cg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi,
        /// SSOR, IC0, FSAI, and Chebyshev overloads above. Restricted Schwarz (RAS) is NOT symmetric
        /// and has no cg rung (pbiCGStab only) -- that absence is the CG-safety guard.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return cg(new fProxyBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Additive-Schwarz Preconditioned Conjugate Gradient over a BSR SPD matrix -- allocates four
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN r  = b.fProxyTempVec(A.M_Rows);
            fProxyN p  = b.fProxyTempVec(A.M_Rows);
            fProxyN Ap = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIter, tol);
        }

        /// <summary>
        /// Additive-Schwarz Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo cg(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x)
        {
            return cg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        // MINRES (symmetric indefinite), BiCGSTAB (non-symmetric), LSQR/LSMR (rectangular
        // least-squares). Same generic-operator pattern as cg&lt;TOp&gt;/cg&lt;TOp,TPre&gt; above --
        // see cg&lt;TOp&gt;'s doc comment for the shared "why an up-front aliasing guard" rationale.
        // These solvers carry more scratch vectors than cg (6-9 vs 3-4), so their guards
        // use RequireDistinctBuffers (a small loop-based helper) instead of a hand-expanded OR chain.

        /// <summary>
        /// Zero-alloc MINRES (Paige-Saunders) solver for symmetric systems A x = b, generic over
        /// <see cref="IfProxyLinearOperator"/>. Unlike <see cref="cg{TOp}"/>, A need NOT be positive
        /// definite -- MINRES minimizes the 2-norm residual ‖b-Ax‖ over the Krylov subspace, so it
        /// converges cleanly on symmetric INDEFINITE (and singular/semidefinite) systems where CG's
        /// p·Ap&gt;0 curvature requirement breaks down. A MUST be symmetric -- caller precondition,
        /// not verified at runtime.
        ///
        /// Caller provides x (initial guess, overwritten with solution -- WARM-STARTABLE) and seven
        /// scratch vectors (y, r1, r2, v, w, w1, w2, all length A.Rows), matching the classic MINRES
        /// variable names (Paige &amp; Saunders 1975).
        ///
        /// Returns a <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖) — see that struct for the
        /// implicit-bool/status/undefined-x contract. Breakdown if the Lanczos recurrence exactly
        /// exhausts the Krylov subspace short of tolerance.
        /// </summary>
        public static SolveInfo minres<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                       ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                       ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                       int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("minres: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("minres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("minres: x.N must equal A.Rows");
            if (y.N != A.Rows) throw new ArgumentException("minres: y.N must equal A.Rows");
            if (r1.N != A.Rows) throw new ArgumentException("minres: r1.N must equal A.Rows");
            if (r2.N != A.Rows) throw new ArgumentException("minres: r2.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("minres: v.N must equal A.Rows");
            if (w.N != A.Rows) throw new ArgumentException("minres: w.N must equal A.Rows");
            if (w1.N != A.Rows) throw new ArgumentException("minres: w1.N must equal A.Rows");
            if (w2.N != A.Rows) throw new ArgumentException("minres: w2.N must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("minres: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)y.Data.Ptr;  ptrs[1] = (long)r1.Data.Ptr; ptrs[2] = (long)r2.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr;  ptrs[4] = (long)w.Data.Ptr;  ptrs[5] = (long)w1.Data.Ptr;
                ptrs[6] = (long)w2.Data.Ptr; ptrs[7] = (long)x.Data.Ptr;  ptrs[8] = (long)b.Data.Ptr;
                RequireDistinctBuffers("minres: y/r1/r2/v/w/w1/w2/x/b must be distinct", ptrs, 9);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r1 = b - A x
            A.Apply(in x, ref y);                       // y = A x (temp use of y)
            r1.Data.CopyFrom(b.Data);
            r1.addScaledInPlace((fProxy)(-1), y);           // r1 = b - A x

            fProxy beta1 = math.sqrt(Blas.dot(r1, r1));
            fProxy threshold = tol * tol * bb;

            if (beta1 * beta1 <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, beta1);

            r2.Data.CopyFrom(r1.Data);

            // Zero the 3-term search-direction history (w/w1/w2 start at 0 in exact MINRES).
            for (int i = 0; i < A.Rows; i++) { w[i] = (fProxy)0; w1[i] = (fProxy)0; w2[i] = (fProxy)0; }

            fProxy oldb = (fProxy)0;
            fProxy beta = beta1;
            fProxy dbar = (fProxy)0;
            fProxy epsln = (fProxy)0;
            fProxy phibar = beta1;
            fProxy cs = (fProxy)(-1);
            fProxy sn = (fProxy)0;
            fProxy gammaFloor = Consts.fProxyEpsilon;

            for (int k = 0; k < maxIter; k++)
            {
                // ---- Lanczos step: extend the tridiagonalization by one vector ----
                // v = r2 / beta, one pass (Blas.scaledCopy with a = 1/beta, i.e. reciprocal-multiply
                // instead of a per-element divide).
                Blas.scaledCopy(1 / beta, r2, ref v);

                A.Apply(in v, ref y);                      // y = A v

                if (k >= 1)
                    y.addScaledInPlace(-(beta / oldb), r1);   // y -= (beta/oldb) r1

                fProxy alfa = Blas.dot(v, y);
                y.addScaledInPlace(-(alfa / beta), r2);       // y -= (alfa/beta) r2

                // Buffer rotation (r1,r2,y) -> (r2,y,r1): swap the local fProxyN handles instead of
                // Data.CopyFrom. r1's old buffer is fully consumed above (last read this iteration)
                // and is recycled as next iteration's y, which A.Apply fully overwrites regardless of
                // its incoming contents.
                { fProxyN tmp = r1; r1 = r2; r2 = y; y = tmp; }

                oldb = beta;
                beta = math.sqrt(Blas.dot(r2, r2));

                // ---- apply the PREVIOUS Givens rotation (cs,sn) to the new tridiagonal column ----
                fProxy oldeps = epsln;
                fProxy delta = cs * dbar + sn * alfa;
                fProxy gbar = sn * dbar - cs * alfa;
                epsln = sn * beta;
                dbar = -cs * beta;

                // ---- compute the NEW Givens rotation that zeros the subdiagonal entry ----
                fProxy gamma = math.sqrt(gbar * gbar + beta * beta);
                gamma = math.max(gamma, gammaFloor);
                cs = gbar / gamma;
                sn = beta / gamma;
                fProxy phi = cs * phibar;
                phibar = sn * phibar;

                // ---- update the 3-term search direction, then the solution ----
                // Buffer rotation (w1,w2,w) -> (w2,w,w1), mirroring the r-rotation above.
                { fProxyN tmp = w1; w1 = w2; w2 = w; w = tmp; }

                // w = (v - oldeps*w1 - delta*w2) / gamma, one pass (Blas.combine3 with s = 1/gamma,
                // i.e. reciprocal-multiply instead of a per-element divide at the end).
                Blas.combine3(ref w, v, -oldeps, w1, -delta, w2, 1 / gamma);

                x.addScaledInPlace(phi, w);

                // phibar IS the true residual norm ‖b-Ax‖ at this step (MINRES identity) --
                // no extra dot product needed, so rnorm = phibar is free.
                if (phibar * phibar <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, phibar);

                if (!(beta > (fProxy)0))
                    // Lanczos breakdown: invariant subspace exhausted, no further progress possible.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, phibar);
        }

        /// <summary>
        /// MINRES over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="minres{TOp}"/> via <see cref="fProxyDenseOperator"/>. See that method for
        /// the actual loop and buffer semantics.
        /// </summary>
        public static SolveInfo minres(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                  ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                  ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                  int maxIter, fProxy tol)
        {
            return minres(new fProxyDenseOperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIter, tol);
        }

        /// <summary>MINRES over a dense matrix -- allocates seven scratch vectors from the arena.</summary>
        public static SolveInfo minres(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIter, tol);
        }

        /// <summary>MINRES over a dense matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo minres(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// MINRES over a symmetric block-sparse (BSR) matrix -- zero-alloc primitive. Forwards
        /// into <see cref="minres{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                  ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                  ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                  int maxIter, fProxy tol)
        {
            return minres(new fProxyBSROperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIter, tol);
        }

        /// <summary>MINRES over a BSR matrix -- allocates seven scratch vectors from the arena.</summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIter, tol);
        }

        /// <summary>MINRES over a BSR matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo minres(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc BiCGSTAB (van der Vorst 1992, stabilized Bi-Conjugate Gradient) solver for
        /// NON-symmetric (general) square systems A x = b, generic over
        /// <see cref="IfProxyLinearOperator"/>. Short two-sided recurrence, flat O(n) memory (no
        /// growing Krylov basis like GMRES) -- the non-symmetric counterpart to CG/MINRES, for
        /// e.g. frictional-LCP or MNA-circuit operators.
        ///
        /// Caller provides x (initial guess, overwritten with solution -- WARM-STARTABLE) and five
        /// scratch vectors r, rHat0, p, v, t (all length A.Rows). rHat0 is the fixed "shadow"
        /// residual, chosen once at the start and never mutated after.
        ///
        /// Returns a <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖) — see that struct for the
        /// implicit-bool/status/undefined-x contract. Breakdown on one of the standard BiCGSTAB
        /// breakdowns (rho == 0, rHat0·v == 0, or omega == 0 -- A not amenable to BiCGSTAB from
        /// this shadow residual, or numerical breakdown).
        /// </summary>
        public static SolveInfo biCGStab<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                         ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("biCGStab: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("biCGStab: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("biCGStab: x.N must equal A.Rows");
            if (r.N != A.Rows) throw new ArgumentException("biCGStab: r.N must equal A.Rows");
            if (rHat0.N != A.Rows) throw new ArgumentException("biCGStab: rHat0.N must equal A.Rows");
            if (p.N != A.Rows) throw new ArgumentException("biCGStab: p.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("biCGStab: v.N must equal A.Rows");
            if (t.N != A.Rows) throw new ArgumentException("biCGStab: t.N must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("biCGStab: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[7];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)rHat0.Data.Ptr; ptrs[2] = (long)p.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr; ptrs[4] = (long)t.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                RequireDistinctBuffers("biCGStab: r/rHat0/p/v/t/x/b must be distinct", ptrs, 7);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref v);                          // v = A x (temp use, overwritten below)
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((fProxy)(-1), v);

            fProxy threshold = tol * tol * bb;

            // rr tracks ‖current residual‖²; ss the ‖half-step residual s‖². Both are already
            // computed for the convergence tests, so every exit reports rnorm from a held value.
            fProxy rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            rHat0.Data.CopyFrom(r.Data);

            // p_0 = v_0 = 0 (standard BiCGSTAB init).
            for (int i = 0; i < A.Rows; i++) { p[i] = (fProxy)0; v[i] = (fProxy)0; }

            fProxy rho = (fProxy)1, alpha = (fProxy)1, omega = (fProxy)1;

            for (int k = 0; k < maxIter; k++)
            {
                fProxy rhoNew = Blas.dot(rHat0, r);

                if (rhoNew == (fProxy)0 || math.isnan(rhoNew))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // r orthogonal to shadow residual

                fProxy beta = (rhoNew / rho) * (alpha / omega);

                p.addScaledInPlace(-omega, v);                // p -= omega v      (still old p, old v)
                p.scaleAddInPlace(beta, r);                    // p = beta p + r

                A.Apply(in p, ref v);                       // v = A p

                fProxy rv = Blas.dot(rHat0, v);

                if (rv == (fProxy)0 || math.isnan(rv))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // breakdown: alpha undefined

                alpha = rhoNew / rv;

                // r := s = r - alpha v ; ss = ||s||^2, fused into one pass (Blas.axpyNormSq).
                fProxy ss = Blas.axpyNormSq(-alpha, v, ref r);

                if (ss <= threshold)
                {
                    // Early exit: the half-step residual s is already small enough -- finish
                    // with x += alpha p (skipping the t = A s stabilization matvec entirely).
                    x.addScaledInPlace(alpha, p);
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(ss));
                }

                A.Apply(in r, ref t);                       // t = A s   (r currently holds s)

                fProxy tt = Blas.dot(t, t);

                if (!(tt > (fProxy)0))                       // NaN-safe: tt is a norm^2, nonnegative
                    // breakdown: omega undefined. x is still x_old here (the alpha·p / omega·r
                    // updates are below), so its residual is rr -- NOT ss (ss = ‖b - A(x_old+alpha·p)‖,
                    // an iterate this path never commits to x).
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                omega = Blas.dot(t, r) / tt;

                if (omega == (fProxy)0 || math.isnan(omega))
                    // breakdown: beta would divide by zero. x is still x_old (see above) -> report rr.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                x.addScaledInPlace(alpha, p);
                x.addScaledInPlace(omega, r);                  // r still holds s here

                // r := s - omega t (new residual) ; rr = ||r||^2, fused into one pass (Blas.axpyNormSq).
                rr = Blas.axpyNormSq(-omega, t, ref r);

                if (rr <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));

                rho = rhoNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(rr));
        }

        /// <summary>
        /// BiCGSTAB over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                    int maxIter, fProxy tol)
        {
            return biCGStab(new fProxyDenseOperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIter, tol);
        }

        /// <summary>BiCGSTAB over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static SolveInfo biCGStab(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIter, tol);
        }

        /// <summary>BiCGSTAB over a dense matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// BiCGSTAB over a block-sparse (BSR) matrix -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                    ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                    int maxIter, fProxy tol)
        {
            return biCGStab(new fProxyBSROperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIter, tol);
        }

        /// <summary>BiCGSTAB over a BSR matrix -- allocates five scratch vectors from the arena.</summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN r     = b.fProxyTempVec(A.M_Rows);
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN p     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN t     = b.fProxyTempVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIter, tol);
        }

        /// <summary>BiCGSTAB over a BSR matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Certified-exact least-squares residual diagnostics: given any solution x, recomputes
        /// rnorm = ‖b - A x‖, Arnorm = ‖Aᵀr - damp²x‖ (the damped normal-equation residual; damp==0
        /// -> ‖Aᵀr‖), and xnorm = ‖x‖ FRESH from x with one extra Apply + one ApplyT. This is the
        /// independent, matvec-paying counterpart to the FREE tracked norms the solvers put in their
        /// returned <see cref="LstsqInfo"/>: use it to audit a solution (or a warm-start seed)
        /// to full accuracy when the couple-extra-matvec cost is acceptable. Reuses two caller
        /// scratch buffers -- <paramref name="rScratch"/> (length A.Rows) and
        /// <paramref name="sScratch"/> (length A.Cols) -- so it allocates nothing. The returned
        /// struct carries only the three norms (iterations = 0, status = Converged as placeholders):
        /// it describes x, not a solve.
        ///
        /// DAMPED CONVENTION: Arnorm uses the ‖x‖-Tikhonov gradient ‖Aᵀr - damp²x‖ -- the optimality
        /// residual for a cold-start (x₀=0) lsqr/lsmr. Auditing a WARM-STARTED
        /// lsqr/lsmr result with damp!=0 will report a nonzero Arnorm even at that solver's optimum,
        /// because it minimizes the CORRECTION-penalized ‖Aᵀr - damp²(x-x₀)‖ instead; rnorm = ‖b-Ax‖
        /// is unaffected by the convention and is always exact.
        /// </summary>
        public static LstsqInfo lstsqResidual<TOp>(in TOp A, in fProxyN b, in fProxyN x, fProxy damp,
                                                         ref fProxyN rScratch, ref fProxyN sScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            // r = b - A x
            A.Apply(in x, ref rScratch);
            rScratch.scaleAddInPlace((fProxy)(-1), b);          // rScratch = -A x + b = b - A x
            fProxy rnorm = math.sqrt(Blas.dot(rScratch, rScratch));

            // s = Aᵀr - damp²x  (the ‖x‖-Tikhonov optimality residual)
            A.ApplyT(in rScratch, ref sScratch);
            if (damp != (fProxy)0) sScratch.addScaledInPlace(-(damp * damp), x);
            fProxy arnorm = math.sqrt(Blas.dot(sScratch, sScratch));

            fProxy xnorm = math.sqrt(Blas.dot(x, x));

            return new LstsqInfo
            {
                rnorm = rnorm,
                Arnorm = arnorm,
                xnorm = xnorm,
                iterations = 0,
                status = IterativeSolveStatus.Converged,
            };
        }

        /// <summary>
        /// Zero-alloc LSQR (Paige-Saunders 1982) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfProxyLinearOperator"/>. Builds an implicit bidiagonalization of A via the
        /// Golub-Kahan process and folds it through an incremental Givens-rotated QR factorization.
        /// Robust on ill-conditioned A (never squares A's condition number the way the normal
        /// equations implicitly do), at O(n+m) memory and per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// Caller provides x (initial guess, length A.Cols -- overwritten with solution, WARM-
        /// STARTABLE) and five scratch vectors: u, tmpM (length A.Rows) and v, w, tmpN (length
        /// A.Cols). Converges when ‖Aᵀr‖ &lt;= tol*‖Aᵀb‖ (the least-squares optimality condition).
        ///
        /// <paramref name="damp"/> (&gt;= 0) applies Tikhonov regularization: minimizes
        /// ‖Ax-b‖² + damp²‖x‖² (damp == 0 is BIT-IDENTICAL to the plain solve).
        ///
        /// WARM START + DAMPING GOTCHA: lsqr bidiagonalizes the residual b - A·x₀, so a NONZERO
        /// initial x₀ makes it minimize ‖Ax-b‖² + damp²‖x - x₀‖² (regularizing the CORRECTION), not
        /// ‖x‖. Start from x = 0 for the ‖x‖-regularized minimizer.
        ///
        /// Returns an <see cref="LstsqInfo"/> — see that struct for the implicit-bool/status/
        /// undefined-x contract. Breakdown on a total bidiagonalization breakdown (alpha and beta
        /// both collapse to zero in the same step).
        /// </summary>
        public static LstsqInfo lsqr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("lsqr: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("lsqr: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("lsqr: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("lsqr: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("lsqr: v.N must equal A.Cols");
            if (w.N != A.Cols) throw new ArgumentException("lsqr: w.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("lsqr: tmpN.N must equal A.Cols");

            if (maxIter < 1)
                throw new ArgumentException("lsqr: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[7];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr; ptrs[2] = (long)w.Data.Ptr;
                ptrs[3] = (long)tmpM.Data.Ptr; ptrs[4] = (long)tmpN.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                RequireDistinctBuffers("lsqr: u/v/w/tmpM/tmpN/x/b must be distinct", ptrs, 7);
            }

            // Fixed scale reference for the relative tolerance: ‖Aᵀb‖².
            A.ApplyT(in b, ref tmpN);
            fProxy atbSq = Blas.dot(tmpN, tmpN);

            if (atbSq == (fProxy)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, math.sqrt(Blas.dot(b, b)), (fProxy)0, (fProxy)0, in x);
            }

            fProxy threshold = tol * tol * atbSq;

            // u = b - A x ; beta = ||u||
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInPlace((fProxy)(-1), tmpM);

            fProxy beta = math.sqrt(Blas.dot(u, u));

            if (beta == (fProxy)0)
                // x already exact (r = 0): rnorm = 0, Aᵀr = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (fProxy)0, (fProxy)0, (fProxy)0, in x);

            u.divInPlace(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            fProxy alpha = math.sqrt(Blas.dot(v, v));

            if (alpha == (fProxy)0)
                // x already least-squares-stationary (A^T r = 0). ‖r‖ = beta.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, (fProxy)0, (fProxy)0, in x);

            v.divInPlace(alpha);

            // phibar tracks ‖r‖ (LSQR identity); arnorm tracks ‖Aᵀr‖ = alpha*beta pre-loop.
            fProxy phibar = beta;
            fProxy rhobar = alpha;
            fProxy arnorm = alpha * beta;

            // Σψ²: energy the damping rotations peel off phibar into the residual. With damp>0 the
            // residual LSQR actually reduces is the AUGMENTED one ‖[b-Ax; -damp·x]‖, whose square is
            // sumPsiSq + phibar² -- phibar ALONE is neither the plain nor the augmented residual once
            // damping folds in. LstsqInfoTracked recovers the plain ‖b-Ax‖ from the augmented norm.
            // damp==0 -> sumPsiSq stays 0, so the undamped path reports rnorm = phibar unchanged.
            fProxy sumPsiSq = (fProxy)0;

            if (arnorm * arnorm <= threshold)
                // already within tolerance before the first bidiagonalization step
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, phibar, arnorm, (fProxy)0, in x);

            w.Data.CopyFrom(v.Data);

            for (int k = 0; k < maxIter; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                // u = A v - alpha u ; beta = ||u||, fused (Blas.xpayNormSq) into one pass over u.
                A.Apply(in v, ref tmpM);
                beta = math.sqrt(Blas.xpayNormSq(-alpha, tmpM, ref u));
                if (beta > (fProxy)0) u.divInPlace(beta);

                // v = A^T u - beta v ; alpha = ||v||, same fusion.
                A.ApplyT(in u, ref tmpN);
                alpha = math.sqrt(Blas.xpayNormSq(-beta, tmpN, ref v));
                if (alpha > (fProxy)0) v.divInPlace(alpha);

                // ---- fold Tikhonov damping into rhobar: rotate (rhobar, damp) -> (rhobar1, 0),
                // scaling phibar by the rotation cosine. damp==0 -> rhobar1==rhobar and phibar is
                // untouched, so the undamped path is bit-identical. ----
                fProxy rhobar1 = rhobar;
                if (damp != (fProxy)0)
                {
                    rhobar1 = math.sqrt(rhobar * rhobar + damp * damp);
                    fProxy psi = (damp / rhobar1) * phibar;  // sn1 * phibar: residual rotated out by damping
                    sumPsiSq += psi * psi;
                    phibar = (rhobar / rhobar1) * phibar;   // cs1 * phibar
                }

                // ---- Givens rotation folding (rhobar1, beta) -> (rho, 0) ----
                fProxy rho = math.sqrt(rhobar1 * rhobar1 + beta * beta);

                if (!(rho > (fProxy)0))
                    // total breakdown: rhobar1 and beta both zero. phibar/arnorm carry the last
                    // pre-rotation values (arnorm from the previous completed step).
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                fProxy c = rhobar1 / rho;
                fProxy sn = beta / rho;
                fProxy theta = sn * alpha;
                rhobar = -c * alpha;
                fProxy phi = c * phibar;
                phibar = sn * phibar;

                // ---- update x using the OLD w, then update w ----
                x.addScaledInPlace(phi / rho, w);
                w.scaleAddInPlace(-theta / rho, v);             // w = -(theta/rho)*w + v

                arnorm = phibar * alpha * math.abs(c);        // ‖Aᵀr‖ for the just-updated x (free)

                if (arnorm * arnorm <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                if (!(beta > (fProxy)0) || !(alpha > (fProxy)0)) // NaN-safe: both are norms, nonnegative
                    // bidiagonalization breakdown: Krylov space exhausted, no further progress
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);
        }

        /// <summary>Assemble an <see cref="LstsqInfo"/> from a solver's tracked residual and
        /// ‖Aᵀr‖ scalars, filling xnorm = ‖x‖ with one dot on x. Shared by lsqr/lsmr.
        ///
        /// Recovers the plain ‖b-Ax‖ from the (possibly damping-augmented) tracked residual; exact
        /// only for damp == 0 or a cold start (x₀ = 0). Under a NONZERO warm start with damping, this
        /// does NOT return ‖b-Ax‖ -- start damped solves from x=0, or read ‖b-Ax‖ from
        /// <see cref="lstsqResidual{TOp}"/> on the returned x.</summary>
        static LstsqInfo LstsqInfoTracked(IterativeSolveStatus status, int iterations, fProxy resNorm, fProxy Arnorm, fProxy dampAug, in fProxyN x)
        {
            fProxy xnorm = math.sqrt(Blas.dot(x, x));
            fProxy rr = resNorm * resNorm - dampAug * dampAug * xnorm * xnorm;
            fProxy rnorm = rr > (fProxy)0 ? math.sqrt(rr) : (fProxy)0;   // guard estimate noise when ‖b-Ax‖≈0
            return new LstsqInfo
            {
                rnorm = rnorm,
                Arnorm = Arnorm,
                xnorm = xnorm,
                iterations = iterations,
                status = status,
            };
        }

        /// <summary>Undamped LSQR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo lsqr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            => lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol, (fProxy)0);

        /// <summary>
        /// LSQR over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsqr{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return lsqr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>LSQR over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN w    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// five scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN w    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return lsqr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol, damp);
        }

        /// <summary>LSQR over a dense matrix with default maxIter (A.N_Cols) and tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsqr{TOp}"/> via <c>fProxyBSROperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA, with better ill-conditioned behavior than the normal
        /// equations.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return lsqr(new fProxyBSROperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.fProxyBSRTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSROperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="lsqr(in fProxyBSR, in fProxyN, ref fProxyN, int, fProxy)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSR A, in fProxyBSR AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return lsqr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// LSQR over a BSR matrix -- allocates five scratch vectors AND materializes A^T ONCE via
        /// <c>arena.fProxyBSRTranspose</c>, then drives LSQR with the two-arg
        /// <see cref="fProxyBSROperator"/> so every ApplyT call routes through a cache-friendly
        /// forward spMV(A^T, x) instead of scatter-heavy spMVT(A, x) every iteration. For a
        /// build-free zero-alloc path, build A^T yourself once and call the zero-alloc
        /// <see cref="lsqr(in fProxyBSR, in fProxyBSR, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// overload above with your own scratch vectors.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN w    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return lsqr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates five
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN w    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return lsqr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol, damp);
        }

        /// <summary>LSQR over a BSR matrix with default maxIter (A.N_Cols) and tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqr(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSMR (Fong-Saunders 2011) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfProxyLinearOperator"/>. Built on the same Golub-Kahan bidiagonalization as
        /// <see cref="lsqr{TOp}"/>, but folds it through a rotation sequence equivalent to applying
        /// MINRES to the normal equations AᵀA x = Aᵀb, so the normal-equation residual ‖Aᵀrₖ‖
        /// decreases MONOTONICALLY. That makes LSMR's stopping test cleaner and its early
        /// termination safer than LSQR's on ill-conditioned problems, at the same O(n+m) memory and
        /// per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// Caller provides x (initial guess, length A.Cols -- overwritten with solution, WARM-
        /// STARTABLE) and six scratch vectors: u, tmpM (length A.Rows) and v, h, hbar, tmpN (length
        /// A.Cols) -- one more than LSQR, since LSMR carries both the Golub-Kahan search direction h
        /// and the MINRES-folded direction hbar. Converges when ‖Aᵀr‖ &lt;= tol*‖Aᵀb‖ (same
        /// contract as lsqr).
        ///
        /// <paramref name="damp"/> (&gt;= 0) applies Tikhonov regularization: minimizes
        /// ‖Ax-b‖² + damp²‖x‖² (damp == 0 is BIT-IDENTICAL to the plain solve).
        ///
        /// WARM START + DAMPING GOTCHA: same as <see cref="lsqr{TOp}"/> -- lsmr bidiagonalizes the
        /// residual b - A·x₀, so a NONZERO initial x₀ makes it minimize ‖Ax-b‖² + damp²‖x - x₀‖²
        /// (regularizing the CORRECTION), not ‖x‖. Start from x = 0 for the ‖x‖-regularized
        /// minimizer.
        ///
        /// Returns an <see cref="LstsqInfo"/> — see that struct for the implicit-bool/status/
        /// undefined-x contract. Breakdown on a bidiagonalization breakdown (a rotation radius
        /// collapses to zero).
        /// </summary>
        public static LstsqInfo lsmr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                     ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("lsmr: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("lsmr: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("lsmr: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("lsmr: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("lsmr: v.N must equal A.Cols");
            if (h.N != A.Cols) throw new ArgumentException("lsmr: h.N must equal A.Cols");
            if (hbar.N != A.Cols) throw new ArgumentException("lsmr: hbar.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("lsmr: tmpN.N must equal A.Cols");

            if (maxIter < 1)
                throw new ArgumentException("lsmr: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[8];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr; ptrs[2] = (long)h.Data.Ptr;
                ptrs[3] = (long)hbar.Data.Ptr; ptrs[4] = (long)tmpM.Data.Ptr; ptrs[5] = (long)tmpN.Data.Ptr;
                ptrs[6] = (long)x.Data.Ptr; ptrs[7] = (long)b.Data.Ptr;
                RequireDistinctBuffers("lsmr: u/v/h/hbar/tmpM/tmpN/x/b must be distinct", ptrs, 8);
            }

            // Fixed scale reference for the relative tolerance (identical contract to lsqr).
            A.ApplyT(in b, ref tmpN);
            fProxy atbSq = Blas.dot(tmpN, tmpN);

            if (atbSq == (fProxy)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, math.sqrt(Blas.dot(b, b)), (fProxy)0, (fProxy)0, in x);
            }

            fProxy threshold = tol * tol * atbSq;

            // u = b - A x ; beta = ||u||   (warm-startable: bidiagonalization of the residual)
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInPlace((fProxy)(-1), tmpM);

            fProxy beta = math.sqrt(Blas.dot(u, u));

            if (beta == (fProxy)0)
                // x already exact (r = 0).
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (fProxy)0, (fProxy)0, (fProxy)0, in x);

            u.divInPlace(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            fProxy alpha = math.sqrt(Blas.dot(v, v));

            if (alpha == (fProxy)0)
                // x already least-squares-stationary (A^T r = 0). ‖r‖ = beta.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, (fProxy)0, (fProxy)0, in x);

            v.divInPlace(alpha);

            // ||A^T r_0|| = alpha*beta = |zetabar_1|; matches lsqr's pre-loop early-out.
            if ((alpha * beta) * (alpha * beta) <= threshold)
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, alpha * beta, (fProxy)0, in x);

            // h = v ; hbar = 0
            h.Data.CopyFrom(v.Data);
            for (int i = 0; i < hbar.N; i++) hbar[i] = (fProxy)0;

            // MINRES-on-normal-equations rotation state.
            fProxy alphabar = alpha;
            fProxy zetabar  = alpha * beta;
            fProxy rho = (fProxy)1, rhobar = (fProxy)1, cbar = (fProxy)1, sbar = (fProxy)0;

            // ---- ‖r‖ estimate state (Fong & Saunders 2011, "LSMR" §5.4 / SciPy lsmr). LSMR does
            // not hold the residual r = b - A x, but ‖r‖ falls out of a short scalar recurrence
            // over the SAME rotations at O(1)/iteration -- no extra matvec/dot. beta here is
            // beta1 = ‖b - A x0‖; undamped (damp==0) -> chat==1, shat==0 -> betacheck==0. ----
            fProxy betadd = beta;
            fProxy betad = (fProxy)0;
            fProxy rhodold = (fProxy)1;
            fProxy tautildeold = (fProxy)0;
            fProxy thetatilde = (fProxy)0;
            fProxy zeta = (fProxy)0;
            fProxy dnorm = (fProxy)0;   // accumulates betacheck^2
            fProxy normr = beta;

            for (int k = 0; k < maxIter; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                // u = A v - alpha u ; beta = ||u||, fused (Blas.xpayNormSq) into one pass over u.
                A.Apply(in v, ref tmpM);
                beta = math.sqrt(Blas.xpayNormSq(-alpha, tmpM, ref u));
                if (beta > (fProxy)0)
                {
                    u.divInPlace(beta);
                    // v = A^T u - beta v ; alpha = ||v||, same fusion.
                    A.ApplyT(in u, ref tmpN);
                    alpha = math.sqrt(Blas.xpayNormSq(-beta, tmpN, ref v));
                    if (alpha > (fProxy)0) v.divInPlace(alpha);
                }

                // ---- rotation P_k : (alphahat, beta) -> (rho, 0) ----
                // alphahat folds in the Tikhonov damping: alphahat = sqrt(alphabar^2 + damp^2).
                // damp==0 -> alphahat==alphabar exactly, so the undamped path is bit-identical.
                // (chat, shat) is the rotation folding damp -- needed by the ‖r‖ recurrence.
                fProxy rhoold = rho;
                fProxy alphahat = damp != (fProxy)0 ? math.sqrt(alphabar * alphabar + damp * damp) : alphabar;
                fProxy chat, shat;
                if (alphahat > (fProxy)0) { chat = alphabar / alphahat; shat = damp / alphahat; }
                else { chat = (fProxy)1; shat = (fProxy)0; }
                rho = math.sqrt(alphahat * alphahat + beta * beta);
                if (!(rho > (fProxy)0))
                    // breakdown: alphahat and beta both zero. normr/zetabar carry the prior step's values.
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
                fProxy c = alphahat / rho;
                fProxy s = beta / rho;
                fProxy thetanew = s * alpha;
                alphabar = c * alpha;

                // ---- rotation Pbar_k : fold R^T into Rbar (the MINRES layer) ----
                fProxy rhobarold = rhobar;
                fProxy thetabar = sbar * rho;
                fProxy cbarrho = cbar * rho;
                rhobar = math.sqrt(cbarrho * cbarrho + thetanew * thetanew);
                if (!(rhobar > (fProxy)0))
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
                cbar = cbarrho / rhobar;
                sbar = thetanew / rhobar;
                fProxy zetaold = zeta;
                zeta = cbar * zetabar;
                zetabar = -sbar * zetabar;

                // ---- updates: hbar, x, h ----
                // hbar = h - (thetabar*rho / (rhoold*rhobarold)) * hbar
                fProxy coefHbar = thetabar * rho / (rhoold * rhobarold);
                hbar.scaleAddInPlace(-coefHbar, h);           // hbar = -coefHbar*hbar + h
                // x = x + (zeta / (rho*rhobar)) * hbar
                x.addScaledInPlace(zeta / (rho * rhobar), hbar);
                // h = v - (thetanew/rho) * h
                h.scaleAddInPlace(-thetanew / rho, v);         // h = -(thetanew/rho)*h + v

                // ---- ‖r‖ recurrence for the just-updated x (this step's rotations; no matvec) ----
                fProxy betaacute = chat * betadd;
                fProxy betacheck = -shat * betadd;
                fProxy betahat = c * betaacute;
                betadd = -s * betaacute;

                fProxy thetatildeold = thetatilde;
                fProxy rhotildeold = math.sqrt(rhodold * rhodold + thetabar * thetabar);
                fProxy ctildeold = rhotildeold > (fProxy)0 ? rhodold / rhotildeold : (fProxy)1;
                fProxy stildeold = rhotildeold > (fProxy)0 ? thetabar / rhotildeold : (fProxy)0;
                thetatilde = stildeold * rhobar;
                rhodold = ctildeold * rhobar;
                betad = -stildeold * betad + ctildeold * betahat;

                tautildeold = rhotildeold > (fProxy)0 ? (zetaold - thetatildeold * tautildeold) / rhotildeold : (fProxy)0;
                fProxy taud = rhodold > (fProxy)0 ? (zeta - thetatilde * tautildeold) / rhodold : (fProxy)0;
                dnorm = dnorm + betacheck * betacheck;
                normr = math.sqrt(dnorm + (betad - taud) * (betad - taud) + betadd * betadd);

                // ‖A^T r‖ for the just-updated x = |zetabar| (falls out for free, decreases
                // monotonically). With damping this is the DAMPED normal-equation residual
                // ‖AᵀA x + damp² x − Aᵀb‖ = ‖Aᵀr − damp² x‖.
                if (zetabar * zetabar <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, normr, math.abs(zetabar), damp, in x);

                if (!(beta > (fProxy)0) || !(alpha > (fProxy)0)) // NaN-safe: both are norms, nonnegative
                    // bidiagonalization breakdown: Krylov space exhausted, no further progress
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIter, normr, math.abs(zetabar), damp, in x);
        }

        /// <summary>Undamped LSMR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo lsmr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                     ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            => lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol, (fProxy)0);

        /// <summary>
        /// LSMR over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsmr{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return lsmr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>LSMR over a dense matrix -- allocates six scratch vectors from the arena.</summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN h    = b.fProxyTempVec(A.N_Cols);
            fProxyN hbar = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// six scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN h    = b.fProxyTempVec(A.N_Cols);
            fProxyN hbar = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return lsmr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol, damp);
        }

        /// <summary>LSMR over a dense matrix with default maxIter (A.N_Cols) and tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsmr{TOp}"/> via <c>fProxyBSROperator</c>. Matrix-free least
        /// squares over a sparse Jacobian-like operator, never forming AᵀA, with LSMR's monotone
        /// ‖Aᵀr‖ decrease (see the generic overload).
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return lsmr(new fProxyBSROperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) BSR matrix -- zero-alloc primitive that takes a
        /// CALLER-PROVIDED precomputed transpose AT (e.g. <c>arena.fProxyBSRTranspose(in A)</c>
        /// built once outside a hot loop) and routes every ApplyT through the cache-friendly
        /// forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSROperator"/>'s two-arg ctor. Caller is responsible for AT being A's
        /// transpose; this overload does not verify it.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSR A, in fProxyBSR AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return lsmr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// LSMR over a BSR matrix -- allocates six scratch vectors AND materializes A^T ONCE via
        /// <c>arena.fProxyBSRTranspose</c>, then drives LSMR with the two-arg
        /// <see cref="fProxyBSROperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// zero-alloc AT overload above with your own scratch vectors.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN h    = b.fProxyTempVec(A.N_Cols);
            fProxyN hbar = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return lsmr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates six
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, fProxy damp)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN h    = b.fProxyTempVec(A.N_Cols);
            fProxyN hbar = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return lsmr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol, damp);
        }

        /// <summary>LSMR over a BSR matrix with default maxIter (A.N_Cols) and tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmr(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        // AᵀA-Jacobi (column-equilibration) convenience overloads.
        // lsqrJacobi / lsmrJacobi build the column scale d[j] = 1/||A_:,j|| from
        // columnNormsSquared, wrap A in a fProxyColScaledOperator, solve the equilibrated system
        // (A*D) y = b with the underlying solver (COLD start -- x is zeroed internally; column
        // scaling is a change of variable, so a warm start would need pre-mapping y0 = D^-1 x0), and
        // unscale x = D*y in place. On an ill-conditioned least-squares problem this converges in
        // fewer iterations than the un-preconditioned solve to the SAME solution. Everything is
        // temp-pool allocated from b. BSR forms materialize A^T once (ApplyT-heavy). For explicit
        // control (custom d, warm start, damping semantics, zero-alloc) use the composable path
        // directly: Blas.columnNormsSquared + buildJacobiScale + fProxyColScaledOperator + the
        // generic solver overload.
        //
        // DIAGNOSTICS: the returned LstsqInfo is reported in ORIGINAL coordinates. The
        // equilibrated solve tracks rnorm/Arnorm/xnorm in scaled y-space (Arnorm = ‖D·Aᵀr‖ can be
        // wildly off), so JacobiFinish recomputes all three exactly on the unscaled A via
        // lstsqResidual (one Apply + ApplyT) and keeps the solve's iteration count + status. So
        // info.Arnorm here is the true ‖Aᵀr‖ and info.Solved still reflects the equilibrated solve.

        /// <summary>Shared tail for the *Jacobi convenience wrappers: unscale the solution
        /// (x = D·y) back to the ORIGINAL variables, then report diagnostics in original coordinates.
        /// The equilibrated solve of (A·D)y=b tracks rnorm/Arnorm/xnorm in the SCALED y-space --
        /// rnorm happens to coincide (‖b-(AD)y‖ = ‖b-Ax‖) but Arnorm = ‖(AD)ᵀr‖ = ‖D·Aᵀr‖ is off by
        /// the column scaling (badly so on exactly the ill-scaled systems the preconditioner targets).
        /// So we recompute all three exactly on the UNSCALED operator via <see cref="lstsqResidual"/>
        /// (one Apply + one ApplyT -- negligible next to the solve and the Aᵀ build these wrappers
        /// already pay), keeping the solve's iteration count and status. <paramref name="Aop"/> is the
        /// UNSCALED operator; <paramref name="mScratch"/>/<paramref name="nScratch"/> are Rows-/Cols-
        /// length scratch (the solver's own buffers, free to reuse post-solve).</summary>
        static LstsqInfo JacobiFinish<TOp>(in TOp Aop, in fProxyN b, ref fProxyN x, in fProxyN d,
                                                 int iterations, IterativeSolveStatus status,
                                                 ref fProxyN mScratch, ref fProxyN nScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            for (int j = 0; j < d.N; j++) x[j] *= d[j];      // unscale x = D y
            var info = lstsqResidual(in Aop, in b, in x, (fProxy)0, ref mScratch, ref nScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        // ---- LSQR + Jacobi ----
        /// <summary>LSQR with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo lsqrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.fProxyTempVec(n), d2 = b.fProxyTempVec(n), scratch = b.fProxyTempVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.fProxyTempVec(m), v = b.fProxyTempVec(n), w = b.fProxyTempVec(n), tmpM = b.fProxyTempVec(m), tmpN = b.fProxyTempVec(n);
            var solveInfo = lsqr(op, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
            return JacobiFinish(new fProxyDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSQR + Jacobi (dense), default maxIter (A.N_Cols) / tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => lsqrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        /// <summary>LSQR with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo lsqrJacobi(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.fProxyTempVec(n), d2 = b.fProxyTempVec(n), scratch = b.fProxyTempVec(n);
            BSR.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            var op = new fProxyColScaledOperator<fProxyBSROperator>(new fProxyBSROperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.fProxyTempVec(m), v = b.fProxyTempVec(n), w = b.fProxyTempVec(n), tmpM = b.fProxyTempVec(m), tmpN = b.fProxyTempVec(n);
            var solveInfo = lsqr(op, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol);
            return JacobiFinish(new fProxyBSROperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSQR + Jacobi (BSR), default maxIter (A.N_Cols) / tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqrJacobi(in fProxyBSR A, in fProxyN b, ref fProxyN x)
            => lsqrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        // ---- LSMR + Jacobi ----
        /// <summary>LSMR with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo lsmrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.fProxyTempVec(n), d2 = b.fProxyTempVec(n), scratch = b.fProxyTempVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.fProxyTempVec(m), v = b.fProxyTempVec(n), h = b.fProxyTempVec(n), hbar = b.fProxyTempVec(n), tmpM = b.fProxyTempVec(m), tmpN = b.fProxyTempVec(n);
            var solveInfo = lsmr(op, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
            return JacobiFinish(new fProxyDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSMR + Jacobi (dense), default maxIter (A.N_Cols) / tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => lsmrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        /// <summary>LSMR with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo lsmrJacobi(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.fProxyTempVec(n), d2 = b.fProxyTempVec(n), scratch = b.fProxyTempVec(n);
            BSR.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            var op = new fProxyColScaledOperator<fProxyBSROperator>(new fProxyBSROperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.fProxyTempVec(m), v = b.fProxyTempVec(n), h = b.fProxyTempVec(n), hbar = b.fProxyTempVec(n), tmpM = b.fProxyTempVec(m), tmpN = b.fProxyTempVec(n);
            var solveInfo = lsmr(op, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIter, tol);
            return JacobiFinish(new fProxyBSROperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSMR + Jacobi (BSR), default maxIter (A.N_Cols) / tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmrJacobi(in fProxyBSR A, in fProxyN b, ref fProxyN x)
            => lsmrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

    }

}
