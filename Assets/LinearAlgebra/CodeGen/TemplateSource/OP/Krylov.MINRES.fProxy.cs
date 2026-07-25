using System;
using Unity.Mathematics;
using BULA.Sparse;

namespace BULA
{
    public static partial class Krylov {

        // MINRES (symmetric indefinite), BiCGSTAB (non-symmetric), LSQR/LSMR (rectangular
        // least-squares). Same generic-operator pattern as cg&lt;TOp&gt;/cg&lt;TOp,TPre&gt; above --
        // see cg&lt;TOp&gt;'s doc comment for the shared "why an up-front aliasing guard" rationale.
        // These solvers carry more scratch vectors than cg (6-9 vs 3-4), so their guards
        // use RequireDistinctBuffers (a small loop-based helper) instead of a hand-expanded OR chain.

        /// <summary>
        /// Zero-alloc MINRES (Paige-Saunders) for symmetric (possibly indefinite) systems A x = b,
        /// generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>) and the preconditioner
        /// (<see cref="IfProxyPreconditioner"/>) -- the SINGLE body behind the plain and the
        /// preconditioned entry points. Unlike <see cref="cg{TOp}"/>, A need NOT be positive definite;
        /// MINRES minimizes ‖b-Ax‖ over the Krylov subspace, converging on symmetric INDEFINITE (and
        /// singular/semidefinite) systems. A MUST be symmetric; a real preconditioner M MUST be SPD --
        /// caller precondition, not verified beyond the NaN-safe breakdown guards.
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold reduces this to plain
        /// MINRES bit-for-bit: z is untouched (may be <c>default</c>), and the initial residual check
        /// matches the classic recurrence. With a real M the Lanczos recurrence runs in the
        /// M⁻¹-inner-product and phibar is the M⁻¹-weighted residual. Either way, a claimed Converged
        /// exit is reported against one freshly recomputed ‖b-Ax‖, not the raw phibar; the
        /// preconditioned MaxIterations exit is verified the same way, while the identity
        /// MaxIterations exit reports the unverified phibar (still an honest non-convergence signal).
        /// Breakdown always reports the unverified phibar.
        ///
        /// Caller provides x (initial guess, overwritten -- WARM-STARTABLE) and eight scratch vectors
        /// (y, r1, r2, v, w, w1, w2, z; all length A.Rows; z unused under the identity). Returns a
        /// <see cref="SolveInfo"/> — see that struct for the implicit-bool/status/undefined-x contract.
        /// </summary>
        public static SolveInfo minres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                       ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                       ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                                       int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
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
            if (!M.IsIdentity && z.N != A.Rows) throw new ArgumentException("minres: z.N must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("minres: maxIter must be >= 1");
            if (!M.IsSpd)
                throw new ArgumentException("Krylov.minres: requires an SPD preconditioner (M.IsSpd == false — e.g. ILU0/SPAI/restricted-Schwarz). Use a non-symmetric solver (gmres/biCGStab) for a general preconditioner.");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.minres: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            unsafe
            {
                // z joins the checked set only for a real preconditioner (identity never touches it).
                int n = M.IsIdentity ? 9 : 10;
                long* ptrs = stackalloc long[10];
                ptrs[0] = (long)y.Data.Ptr;  ptrs[1] = (long)r1.Data.Ptr; ptrs[2] = (long)r2.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr;  ptrs[4] = (long)w.Data.Ptr;  ptrs[5] = (long)w1.Data.Ptr;
                ptrs[6] = (long)w2.Data.Ptr; ptrs[7] = (long)x.Data.Ptr;  ptrs[8] = (long)b.Data.Ptr;
                if (!M.IsIdentity) ptrs[9] = (long)z.Data.Ptr;
                RequireDistinctBuffers("minres: y/r1/r2/v/w/w1/w2/z/x/b must be distinct", ptrs, n);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.CopyFrom(in b);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r1 = b - A x
            A.Apply(in x, ref y);                       // y = A x (temp use of y)
            r1.CopyFrom(in b);
            r1.addScaledInPlace((fProxy)(-1), y);           // r1 = b - A x

            fProxy threshold = tol * tol * bb;
            fProxy trueRR0 = Blas.dot(r1, r1);

            // Lanczos normalization beta. Identity: beta = ‖r1‖, and phibar tracks the TRUE residual
            // (so the checks below match plain MINRES exactly). Preconditioned: beta = sqrt⟨r1, M⁻¹r1⟩
            // in the M-inner-product, and phibar tracks the M⁻¹-weighted residual (verified at exit).
            fProxy beta;
            if (M.IsIdentity)
            {
                beta = math.sqrt(trueRR0);
                if (beta * beta <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, 0, beta);
            }
            else
            {
                if (trueRR0 <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(trueRR0));

                M.Apply(in r1, ref z);                       // z = M⁻¹ r1
                fProxy betaSq = Blas.dot(r1, z);
                // Non-SPD preconditioner (non-positive ⟨r1, M⁻¹r1⟩): mirrors cg's breakdown guard.
                if (!(betaSq > (fProxy)0))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, 0, math.sqrt(trueRR0));
                beta = math.sqrt(betaSq);
            }

            r2.CopyFrom(in r1);

            // Zero the 3-term search-direction history (w/w1/w2 start at 0 in exact MINRES).
            for (int i = 0; i < A.Rows; i++) { w[i] = (fProxy)0; w1[i] = (fProxy)0; w2[i] = (fProxy)0; }

            fProxy oldb = (fProxy)0;
            fProxy dbar = (fProxy)0;
            fProxy epsln = (fProxy)0;
            fProxy phibar = beta;
            fProxy cs = (fProxy)(-1);
            fProxy sn = (fProxy)0;
            fProxy gammaFloor = Consts.fProxyEpsilon;

            for (int k = 0; k < maxIter; k++)
            {
                // ---- (preconditioned) Lanczos step: extend the tridiagonalization by one vector ----
                // v = (M⁻¹ of the current Lanczos vector) / beta. Identity: that is just r2/beta.
                if (M.IsIdentity) Blas.scaledCopy(1 / beta, r2, ref v);
                else              Blas.scaledCopy(1 / beta, z, ref v);

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

                // beta = sqrt of the (M-weighted) norm of the new unpreconditioned Lanczos vector.
                fProxy betaNewSq;
                if (M.IsIdentity)
                {
                    betaNewSq = Blas.dot(r2, r2);
                }
                else
                {
                    M.Apply(in r2, ref z);                    // z = M⁻¹ r2 (feeds next v, and beta now)
                    betaNewSq = Blas.dot(r2, z);
                    // Non-SPD preconditioner: ⟨r2, M⁻¹r2⟩ < 0 -> sqrt = NaN. Bail before the Givens/x
                    // update poisons a warm-started x; x left untouched.
                    if (!(betaNewSq >= (fProxy)0))
                        return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);
                }
                beta = math.sqrt(betaNewSq);

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

                if (phibar * phibar <= threshold)
                {
                    // Verify-at-exit (identity AND preconditioned): phibar can drift from the true
                    // ‖b-Ax‖ once gamma has been floored (gammaFloor guard above breaks the Givens
                    // rotation's unitarity) -- true under the identity path too. y and v are both
                    // idle here (y: recycled garbage; v: consumed by combine3 above), reused as
                    // scratch. Fall through and keep iterating on a failed verify.
                    fProxy trueRR = VerifyTrueResidual(in A, in b, in x, ref y, ref v);

                    if (trueRR <= threshold)
                        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(trueRR));
                }

                if (!(beta > (fProxy)0))
                    // Lanczos breakdown: invariant subspace exhausted, no further progress possible.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);
            }

            if (M.IsIdentity)
                return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, phibar);

            // Preconditioned MaxIterations: report the TRUE residual (one fresh Apply), not phibar.
            fProxy finalRR = VerifyTrueResidual(in A, in b, in x, ref y, ref v);
            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(finalRR));
        }

        /// <summary>
        /// Unpreconditioned MINRES -- forwards into the merged
        /// <see cref="minres{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// with the identity preconditioner (whose IsIdentity fold strips all z traffic), so this
        /// needs no z buffer.
        /// </summary>
        public static SolveInfo minres<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                       ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                       ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                       int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyN z = default;
            return minres(in A, default(fProxyIdentityPreconditioner), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
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

        /// <summary>MINRES over a dense matrix -- allocates seven scratch vectors from Allocator.Temp.</summary>
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

        /// <summary>MINRES over a BSR matrix -- allocates seven scratch vectors from Allocator.Temp.</summary>
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
        /// Preconditioned MINRES solver -- allocates eight scratch vectors from Allocator.Temp and
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
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with ANY
        /// <see cref="IfProxyPreconditioner"/> (block-Jacobi/SSOR/IC0/FSAI/Chebyshev/additive-Schwarz).
        /// Forwards into <see cref="minres{TOp,TPre}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo minres<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            return minres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned MINRES over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/>
        /// (block-Jacobi/SSOR/IC0/FSAI/Chebyshev/additive-Schwarz) -- allocates eight scratch
        /// vectors from Allocator.Temp and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo minres<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
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
        /// Preconditioned MINRES over a BSR matrix with ANY <see cref="IfProxyPreconditioner"/>
        /// (block-Jacobi/SSOR/IC0/FSAI/Chebyshev/additive-Schwarz), with default maxIter (A.M_Rows)
        /// and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo minres<TPre>(in fProxyBSR A, in TPre M, in fProxyN b, ref fProxyN x)
            where TPre : struct, IfProxyPreconditioner
        {
            return minres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
