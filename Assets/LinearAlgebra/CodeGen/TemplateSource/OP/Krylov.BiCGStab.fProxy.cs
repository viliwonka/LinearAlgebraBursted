using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// Zero-alloc BiCGSTAB (van der Vorst 1992) for NON-symmetric (general) square systems
        /// A x = b, generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>) and the
        /// preconditioner (<see cref="IfProxyPreconditioner"/>) -- the SINGLE body behind the plain
        /// and the right-preconditioned entry points. Short two-sided recurrence, flat O(n) memory
        /// (no growing Krylov basis like GMRES) -- the non-symmetric counterpart to CG/MINRES.
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes pHat = p and
        /// sHat = s, so this compiles to, and is bit-identical to, plain BiCGSTAB -- pHat/sHat are
        /// untouched and may be passed as <c>default</c>. With a real M this is right-preconditioned
        /// BiCGSTAB (M ≈ A applied as M⁻¹): pHat = M⁻¹p, sHat = M⁻¹s drive the A-applies and the x
        /// update.
        ///
        /// Caller provides x (initial guess, overwritten -- WARM-STARTABLE) and seven scratch vectors
        /// r, rHat0, p, v, t, pHat, sHat (all length A.Rows; pHat/sHat unused under the identity).
        /// rHat0 is the fixed "shadow" residual. Returns a <see cref="SolveInfo"/>. Breakdown on one
        /// of the standard BiCGSTAB breakdowns (rho == 0, rHat0·v == 0, or omega == 0).
        /// </summary>
        public static SolveInfo biCGStab<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                         ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                         ref fProxyN pHat, ref fProxyN sHat,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
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
            if (!M.IsIdentity && (pHat.N != A.Rows || sHat.N != A.Rows))
                throw new ArgumentException("biCGStab: pHat/sHat must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("biCGStab: maxIter must be >= 1");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.biCGStab: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            unsafe
            {
                // pHat/sHat join the checked set only for a real preconditioner.
                int n = M.IsIdentity ? 7 : 9;
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)rHat0.Data.Ptr; ptrs[2] = (long)p.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr; ptrs[4] = (long)t.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                if (!M.IsIdentity) { ptrs[7] = (long)pHat.Data.Ptr; ptrs[8] = (long)sHat.Data.Ptr; }
                RequireDistinctBuffers("biCGStab: r/rHat0/p/v/t/pHat/sHat/x/b must be distinct", ptrs, n);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref v);                          // v = A x (temp use, overwritten below)
            r.CopyFrom(in b);
            r.addScaledInPlace((fProxy)(-1), v);

            fProxy threshold = tol * tol * bb;

            // rr tracks ‖current residual‖²; ss the ‖half-step residual s‖². Both are already
            // computed for the convergence tests, so every exit reports rnorm from a held value.
            fProxy rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            rHat0.CopyFrom(in r);

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

                // v = A (M⁻¹ p). Identity: pHat = p, so v = A p directly (pHat untouched).
                if (M.IsIdentity)
                {
                    A.Apply(in p, ref v);
                }
                else
                {
                    M.Apply(in p, ref pHat);                  // pHat = M⁻¹ p
                    A.Apply(in pHat, ref v);                  // v = A pHat
                }

                fProxy rv = Blas.dot(rHat0, v);

                if (rv == (fProxy)0 || math.isnan(rv))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // breakdown: alpha undefined

                alpha = rhoNew / rv;

                // r := s = r - alpha v ; ss = ||s||^2, fused into one pass (Blas.axpyNormSq).
                fProxy ss = Blas.axpyNormSq(-alpha, v, ref r);

                if (ss <= threshold)
                {
                    // Verify-at-exit on a TRIAL x (not yet committed): t/v are both idle here (t: not
                    // yet written this iteration; v: fully consumed forming rv above). On a failed
                    // verify, x is left untouched, so the standard stabilization step below applies
                    // alpha*p exactly once (no double-apply).
                    if (M.IsIdentity) { t.CopyFrom(in x); t.addScaledInPlace(alpha, p); }
                    else              { t.CopyFrom(in x); t.addScaledInPlace(alpha, pHat); }
                    A.Apply(in t, ref v);                     // v = A * (trial x)
                    v.addScaledInPlace((fProxy)(-1), b);      // v = A*(trial x) - b; sign irrelevant, only dot(v,v) is used
                    fProxy trialRR = Blas.dot(v, v);
                    if (trialRR <= threshold)
                    {
                        x.CopyFrom(in t);
                        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(trialRR));
                    }
                    // Failed trial: this block overwrote v (= A M⁻¹p), which the NEXT iteration's
                    // p-recurrence (p -= omega v) reads. t is overwritten below, but v is not — restore it.
                    if (M.IsIdentity) A.Apply(in p, ref v);
                    else              A.Apply(in pHat, ref v);
                }

                // t = A (M⁻¹ s). Identity: sHat = s (r holds s), so t = A r directly (sHat untouched).
                if (M.IsIdentity)
                {
                    A.Apply(in r, ref t);                     // t = A s   (r currently holds s)
                }
                else
                {
                    M.Apply(in r, ref sHat);                  // sHat = M⁻¹ s
                    A.Apply(in sHat, ref t);                  // t = A sHat
                }

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

                // x += alpha (M⁻¹ p) + omega (M⁻¹ s). Identity: pHat = p, sHat = s (r holds s).
                if (M.IsIdentity)
                {
                    x.addScaledInPlace(alpha, p);
                    x.addScaledInPlace(omega, r);              // r still holds s here
                }
                else
                {
                    x.addScaledInPlace(alpha, pHat);
                    x.addScaledInPlace(omega, sHat);
                }

                // r := s - omega t (new residual) ; rr = ||r||^2, fused into one pass (Blas.axpyNormSq).
                rr = Blas.axpyNormSq(-omega, t, ref r);

                if (rr <= threshold)
                {
                    // Verify-at-exit. Scratch must be t (idle: last read at the rr update above, next
                    // written next iteration) — NOT v, which the next iteration's p-recurrence still
                    // reads. On a failed verify, r is left holding the FRESH residual so the next
                    // iteration continues from a corrected state.
                    rr = VerifyTrueResidual(in A, in b, in x, ref t, ref r);

                    if (rr <= threshold)
                        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));
                }

                rho = rhoNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(rr));
        }

        /// <summary>
        /// Unpreconditioned BiCGSTAB -- forwards into the merged
        /// <see cref="biCGStab{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips the pHat/sHat traffic), so
        /// this needs no pHat/sHat buffers.
        /// </summary>
        public static SolveInfo biCGStab<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                         ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyN pHat = default, sHat = default;
            return biCGStab(in A, default(fProxyIdentityPreconditioner), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, ref pHat, ref sHat, maxIter, tol);
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

        /// <summary>BiCGSTAB over a dense matrix -- allocates five scratch vectors from Allocator.Temp.</summary>
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

        /// <summary>BiCGSTAB over a BSR matrix -- allocates five scratch vectors from Allocator.Temp.</summary>
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
        /// Preconditioned BiCGSTAB over a block-sparse (BSR) matrix with ANY
        /// <see cref="IfProxyPreconditioner"/> (ILU0/SPAI/restricted-Schwarz) -- forwards into
        /// <see cref="biCGStab{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo biCGStab<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
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

        /// <summary>Preconditioned BiCGSTAB over BSR with default maxIter (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
        {
            return biCGStab(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
