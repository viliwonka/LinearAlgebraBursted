using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// Zero-alloc TFQMR (Transpose-Free QMR, Freund 1993) for NON-symmetric (general) square
        /// systems A x = b, generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>)
        /// and the preconditioner (<see cref="IfProxyPreconditioner"/>) -- the SINGLE body behind
        /// the plain and the RIGHT-preconditioned entry points. Derived from CGS but replaces its
        /// erratic residual with a quasi-minimized quantity (a smoothed, monotonically-shrinking
        /// upper bound on the true residual) at the same cost of one A-apply per HALF-step, two
        /// half-steps per full Lanczos-style step -- so <paramref name="maxIter"/> bounds
        /// half-steps (~one A-apply each), unlike biCGStab's two-matvec-per-pass convention.
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes uHat = u, so
        /// this compiles to, and is bit-identical to, plain TFQMR -- uHat is untouched and may be
        /// passed as <c>default</c>. With a real M this is right-preconditioned TFQMR (solving
        /// A M⁻¹ y = b, x = M⁻¹ y implicitly): every direction fed to A is first passed through M⁻¹
        /// (uHat = M⁻¹ u), and that SAME uHat feeds the solution-update accumulator d directly, so
        /// eta·d folds straight into x with no separate M⁻¹ apply at the end.
        ///
        /// Caller provides x (initial guess, overwritten -- WARM-STARTABLE) and seven scratch
        /// vectors rHat0, u, w, v, au, d, uHat (all length A.Rows; uHat unused under the identity).
        /// rHat0 is the fixed initial residual, doubling as the shadow vector for every
        /// bi-orthogonality dot product (never mutated after setup). Returns a
        /// <see cref="SolveInfo"/>. TFQMR's own quasi-residual bound (Freund's tau·sqrt(half-steps
        /// taken)) is a rigorous upper bound on the true ‖b−Ax‖ only in exact arithmetic; when the
        /// bound first crosses tol·‖b‖ this is verified against a freshly recomputed true residual
        /// before being reported as Converged (falling back to MaxIterations, not a false Converged,
        /// if the estimate turned out optimistic) -- rnorm on that path is the verified value, not
        /// the raw bound. Breakdown on a zero/NaN bi-orthogonality dot (vtrstar), a zero/NaN/
        /// degenerate alpha, a zero/NaN rhoLast (beta undefined), or tau collapsing to zero/NaN
        /// before the bound-based convergence check catches it.
        /// </summary>
        public static SolveInfo tfqmr<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                         ref fProxyN rHat0, ref fProxyN u, ref fProxyN w, ref fProxyN v,
                                         ref fProxyN au, ref fProxyN d, ref fProxyN uHat,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("tfqmr: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("tfqmr: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("tfqmr: x.N must equal A.Rows");
            if (rHat0.N != A.Rows) throw new ArgumentException("tfqmr: rHat0.N must equal A.Rows");
            if (u.N != A.Rows) throw new ArgumentException("tfqmr: u.N must equal A.Rows");
            if (w.N != A.Rows) throw new ArgumentException("tfqmr: w.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("tfqmr: v.N must equal A.Rows");
            if (au.N != A.Rows) throw new ArgumentException("tfqmr: au.N must equal A.Rows");
            if (d.N != A.Rows) throw new ArgumentException("tfqmr: d.N must equal A.Rows");
            if (!M.IsIdentity && uHat.N != A.Rows)
                throw new ArgumentException("tfqmr: uHat must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("tfqmr: maxIter must be >= 1");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.tfqmr: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            unsafe
            {
                // uHat joins the checked set only for a real preconditioner.
                int n = M.IsIdentity ? 8 : 9;
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)rHat0.Data.Ptr; ptrs[1] = (long)u.Data.Ptr; ptrs[2] = (long)w.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr; ptrs[4] = (long)au.Data.Ptr; ptrs[5] = (long)d.Data.Ptr;
                ptrs[6] = (long)x.Data.Ptr; ptrs[7] = (long)b.Data.Ptr;
                if (!M.IsIdentity) ptrs[8] = (long)uHat.Data.Ptr;
                RequireDistinctBuffers("tfqmr: rHat0/u/w/v/au/d/uHat/x/b must be distinct", ptrs, n);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            fProxy thresh = tol * math.sqrt(bb);

            // rHat0 = b - A x -- the fixed initial residual AND shadow vector for the rest of the
            // solve (never mutated again).
            A.Apply(in x, ref v);                          // v = A x   (temp use, overwritten below)
            rHat0.CopyFrom(in b);
            rHat0.addScaledInPlace((fProxy)(-1), v);

            fProxy tau = math.sqrt(Blas.dot(rHat0, rHat0));
            if (tau <= thresh)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, tau);

            u.CopyFrom(in rHat0);
            w.CopyFrom(in rHat0);

            // au = A (M⁻¹ u). Identity: uHat = u, so au = A u directly (uHat untouched).
            if (M.IsIdentity) A.Apply(in u, ref au);
            else { M.Apply(in u, ref uHat); A.Apply(in uHat, ref au); }

            v.CopyFrom(in au);
            for (int i = 0; i < A.Rows; i++) d[i] = (fProxy)0;

            fProxy theta = (fProxy)0, eta = (fProxy)0, alpha = (fProxy)0;
            fProxy rho = tau * tau;                        // <rHat0, rHat0> == tau^2 (shadow == r0)
            fProxy rhoLast = rho;
            fProxy bound = tau;

            for (int k = 0; k < maxIter; k++)
            {
                bool even = (k & 1) == 0;

                if (even)
                {
                    fProxy vtrstar = Blas.dot(rHat0, v);
                    if (vtrstar == (fProxy)0 || math.isnan(vtrstar))
                        return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, bound); // r orthogonal to shadow

                    alpha = rho / vtrstar;
                    if (alpha == (fProxy)0 || math.isnan(alpha))
                        return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, bound); // degenerate alpha
                }

                // w -= alpha au -- au already holds A(M⁻¹ u) for the CURRENT u (computed at the
                // tail of the previous half-step, or just above for k == 0).
                w.addScaledInPlace(-alpha, au);

                // d = uHat + (theta^2/alpha) eta d -- uHat is M⁻¹ of the CURRENT u (identity: u
                // itself), already right-preconditioned so eta*d below folds straight into x.
                fProxy dCoeff = (theta * theta / alpha) * eta;
                if (M.IsIdentity) d.scaleAddInPlace(dCoeff, u);
                else              d.scaleAddInPlace(dCoeff, uHat);

                theta = math.sqrt(Blas.dot(w, w)) / tau;
                fProxy c = (fProxy)1 / math.sqrt((fProxy)1 + theta * theta);
                tau = tau * theta * c;
                eta = c * c * alpha;

                x.addScaledInPlace(eta, d);

                bound = tau * math.sqrt((fProxy)(k + 1));
                if (bound <= thresh)
                {
                    // Verify-at-exit: Freund's bound is rigorous only in exact arithmetic (see the
                    // header doc). No scratch vector is idle here across both loop parities under
                    // the identity preconditioner (au/v/d/u/w all feed the next half-step), so this
                    // commits to a final return either way instead of falling through -- au and v
                    // are safe to clobber since both outcomes return immediately.
                    fProxy trueRnorm = math.sqrt(VerifyTrueResidual(in A, in b, in x, ref au, ref v));
                    return MakeSolveInfo(
                        trueRnorm <= thresh ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations,
                        k + 1, trueRnorm);
                }
                if (tau == (fProxy)0 || math.isnan(tau))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, bound); // tau collapsed

                if (even)
                {
                    // Advance u = u - alpha v, then refresh au (and uHat) from the new u.
                    u.addScaledInPlace(-alpha, v);
                    if (M.IsIdentity) A.Apply(in u, ref au);
                    else { M.Apply(in u, ref uHat); A.Apply(in uHat, ref au); }
                    rhoLast = rho;
                }
                else
                {
                    fProxy rhoNew = Blas.dot(rHat0, w);
                    if (rhoLast == (fProxy)0 || math.isnan(rhoLast))
                        return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, bound); // beta undefined
                    fProxy beta = rhoNew / rhoLast;
                    if (math.isnan(beta))
                        return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, bound);
                    rho = rhoNew;

                    u.scaleAddInPlace(beta, w);                 // u = beta u + w
                    v.mulInPlace(beta * beta);                  // v = beta^2 v ...
                    v.addScaledInPlace(beta, au);                // ... + beta au   (au still OLD here)
                    if (M.IsIdentity) A.Apply(in u, ref au);
                    else { M.Apply(in u, ref uHat); A.Apply(in uHat, ref au); }
                    v.addInPlace(au);                            // v += au   (au now refreshed)
                }
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, bound);
        }

        /// <summary>
        /// Unpreconditioned TFQMR -- forwards into the merged
        /// <see cref="tfqmr{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips the uHat traffic), so
        /// this needs no uHat buffer.
        /// </summary>
        public static SolveInfo tfqmr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                         ref fProxyN rHat0, ref fProxyN u, ref fProxyN w, ref fProxyN v,
                                         ref fProxyN au, ref fProxyN d,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyN uHat = default;
            return tfqmr(in A, default(fProxyIdentityPreconditioner), in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, ref uHat, maxIter, tol);
        }

        /// <summary>
        /// TFQMR over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="tfqmr{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static SolveInfo tfqmr(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    ref fProxyN rHat0, ref fProxyN u, ref fProxyN w, ref fProxyN v,
                                    ref fProxyN au, ref fProxyN d,
                                    int maxIter, fProxy tol)
        {
            return tfqmr(new fProxyDenseOperator(in A), in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, maxIter, tol);
        }

        /// <summary>TFQMR over a dense matrix -- allocates six scratch vectors from Allocator.Temp.</summary>
        public static SolveInfo tfqmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN u     = b.fProxyTempVec(A.M_Rows);
            fProxyN w     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN au    = b.fProxyTempVec(A.M_Rows);
            fProxyN d     = b.fProxyTempVec(A.M_Rows);
            return tfqmr(in A, in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, maxIter, tol);
        }

        /// <summary>TFQMR over a dense matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo tfqmr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return tfqmr(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// TFQMR over a block-sparse (BSR) matrix -- zero-alloc primitive. Forwards into
        /// <see cref="tfqmr{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo tfqmr(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                    ref fProxyN rHat0, ref fProxyN u, ref fProxyN w, ref fProxyN v,
                                    ref fProxyN au, ref fProxyN d,
                                    int maxIter, fProxy tol)
        {
            return tfqmr(new fProxyBSROperator(in A), in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, maxIter, tol);
        }

        /// <summary>TFQMR over a BSR matrix -- allocates six scratch vectors from Allocator.Temp.</summary>
        public static SolveInfo tfqmr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN u     = b.fProxyTempVec(A.M_Rows);
            fProxyN w     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN au    = b.fProxyTempVec(A.M_Rows);
            fProxyN d     = b.fProxyTempVec(A.M_Rows);
            return tfqmr(in A, in b, ref x, ref rHat0, ref u, ref w, ref v, ref au, ref d, maxIter, tol);
        }

        /// <summary>TFQMR over a BSR matrix with default maxIter (A.M_Rows) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo tfqmr(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return tfqmr(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Right-preconditioned TFQMR over a block-sparse (BSR) matrix with ANY
        /// <see cref="IfProxyPreconditioner"/> (ILU0) -- forwards into <see cref="tfqmr{TOp,TPre}"/>
        /// via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo tfqmr<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN rHat0 = b.fProxyTempVec(A.M_Rows);
            fProxyN u     = b.fProxyTempVec(A.M_Rows);
            fProxyN w     = b.fProxyTempVec(A.M_Rows);
            fProxyN v     = b.fProxyTempVec(A.M_Rows);
            fProxyN au    = b.fProxyTempVec(A.M_Rows);
            fProxyN d     = b.fProxyTempVec(A.M_Rows);
            fProxyN uHat  = b.fProxyTempVec(A.M_Rows);
            return tfqmr(new fProxyBSROperator(in A), in M, in b, ref x,
                             ref rHat0, ref u, ref w, ref v, ref au, ref d, ref uHat,
                             maxIter, tol);
        }

        /// <summary>Preconditioned TFQMR over BSR with ANY <see cref="IfProxyPreconditioner"/> (ILU0),
        /// with default maxIter (A.M_Rows) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo tfqmr<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
        {
            return tfqmr(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
