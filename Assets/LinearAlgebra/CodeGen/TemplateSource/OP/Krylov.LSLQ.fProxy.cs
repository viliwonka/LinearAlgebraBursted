using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// Zero-alloc LSLQ (Estrin, Orban &amp; Saunders 2019) solver for RECTANGULAR least-squares
        /// systems: minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfProxyLinearOperator"/>. Builds the same Golub-Kahan bidiagonalization as
        /// <see cref="lsqr{TOp}"/> but folds it through an LQ factorization -- the SYMMLQ-equivalent
        /// formulation on the normal equations AᵀAx = Aᵀb. Returns the LQ point xᴸ, whose Euclidean
        /// forward error ‖xᴸ - x*‖ decreases monotonically (LSLQ's error-minimization property), NOT
        /// the residual-minimizing LSQR point. O(n+m) memory and per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// No warm start: the error-minimization characterization only holds from x₀ = 0, so x is
        /// zeroed internally. Converges when the certified optimality residual ‖Aᵀ(b-Ax)‖ &lt;=
        /// tol*‖Aᵀb‖ (a cheap forward-error trigger arms a certified audit; a claimed Converged is
        /// always reconciled against that audit before being reported). Returns an
        /// <see cref="LslqInfo"/>; rnorm/Arnorm/xnorm are certified-exact. Breakdown when the
        /// bidiagonalization collapses before reaching optimality (A lacks full column rank).
        ///
        /// <paramref name="sigmaMinEst"/> is a strict UNDERESTIMATE of σ_min(A): when &gt; 0 the
        /// returned <see cref="LslqInfo.xErrBound"/> is the constant-time Gauss-Radau bound |ζ̃|
        /// (EOS2019) on ‖x* - xᴸ‖. In the DOUBLE build it is a certified upper bound (given
        /// <paramref name="sigmaMinEst"/> ≤ σ_min(A)); in the FLOAT build it is a tight ESTIMATE (~1-3%,
        /// may marginally under-report) -- it reads the bidiagonalization scalars, so it inherits the
        /// solve's precision floor and single precision cannot certify it. Too large a
        /// <paramref name="sigmaMinEst"/> also makes it under-report (the caller owns that contract).
        /// When ≤ 0 the bound machinery is skipped and xErrBound is NaN.
        /// </summary>
        public static LslqInfo lslq<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN wbar,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol, double sigmaMinEst = 0)
            where TOp : struct, IfProxyLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("lslq: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("lslq: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("lslq: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("lslq: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("lslq: v.N must equal A.Cols");
            if (wbar.N != A.Cols) throw new ArgumentException("lslq: wbar.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("lslq: tmpN.N must equal A.Cols");

            if (maxIter < 1)
                throw new ArgumentException("lslq: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[7];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr; ptrs[2] = (long)wbar.Data.Ptr;
                ptrs[3] = (long)tmpM.Data.Ptr; ptrs[4] = (long)tmpN.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                RequireDistinctBuffers("lslq: u/v/wbar/tmpM/tmpN/x/b must be distinct", ptrs, 7);
            }

            // No warm start: the error-minimization characterization requires x₀ = 0.
            for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;

            // x = 0 is EXACT on the early-out paths below, so ‖x*-x‖ = 0: report a 0 bound when a
            // σ estimate was supplied (only NaN it when the bound was not requested).
            double exactBound = sigmaMinEst > 0 ? 0.0 : double.NaN;

            fProxy bnorm = math.sqrt(Blas.dot(b, b));
            if (bnorm == (fProxy)0)
                // b = 0: x = 0 is the exact least-squares solution.
                return LslqInfoFrom(IterativeSolveStatus.Converged, 0, in A, in b, ref x, ref tmpM, ref tmpN, exactBound);

            // β₁ u₁ = b  (Golub-Kahan line 1).
            u.CopyFrom(in b);
            u.divInPlace(bnorm);

            // α₁ v₁ = Aᵀ u₁.
            A.ApplyT(in u, ref tmpN);
            v.CopyFrom(in tmpN);
            fProxy alpha = math.sqrt(Blas.dot(v, v));

            if (alpha == (fProxy)0)
                // Aᵀb = 0: x = 0 is already least-squares-stationary.
                return LslqInfoFrom(IterativeSolveStatus.Converged, 0, in A, in b, ref x, ref tmpM, ref tmpN, exactBound);

            v.divInPlace(alpha);

            // ‖Aᵀb‖ = α₁·β₁ (since u₁ = b/β₁): the fixed optimality scale for the stopping test.
            fProxy atbNorm = alpha * bnorm;

            wbar.CopyFrom(in v);                 // w̄₁ = v₁

            // LQ-recurrence state (EOS2019 Algorithm 2; c₀ = δ₁ = -1, γ̄₁ = α₁, τ₀ = α₁β₁).
            fProxy gbar = alpha;
            fProxy c = (fProxy)(-1);
            fProxy s = (fProxy)0;
            fProxy delta = (fProxy)(-1);
            fProxy tau = alpha * bnorm;
            fProxy zeta = (fProxy)0;
            double xlqNorm2 = 0;

            // ---- Gauss-Radau forward-error bound sidecar (active only for a positive σ estimate).
            // Runs at the solve's precision: it reads the bidiagonalization scalars (gamma/delta/tau/
            // zeta/c/s), so a double sidecar can't recover accuracy the float solve already lost --
            // hence a certified upper bound in the double build, a ~1-3% estimate in the float build.
            fProxy se = (fProxy)sigmaMinEst;
            bool boundOn = se > (fProxy)0;
            bool complexBnd = false;
            fProxy csig = (fProxy)(-1);
            fProxy rhoBar = -se;
            fProxy omega = (fProxy)0;
            double xErrBound = double.NaN;

            for (int k = 0; k < maxIter; k++)
            {
                // ---- Golub-Kahan: β_{k+1} u = A v - α u ; α_{k+1} v = Aᵀ u - β v ----
                fProxy beta = GolubKahanUStep(in A, in v, alpha, ref tmpM, ref u);
                if (beta > (fProxy)0) u.divInPlace(beta);

                alpha = GolubKahanVStep(in A, in u, beta, ref tmpN, ref v);
                if (alpha > (fProxy)0) v.divInPlace(alpha);

                // ---- QR of the lower bidiagonal: rotate (γ̄, β) -> (γ, 0) ----
                SymGivens(gbar, beta, out fProxy cp, out fProxy sp, out fProxy gamma);
                if (gamma == (fProxy)0)
                    // γ̄ = β = 0: the bidiagonalization has no further direction.
                    return LslqInfoFrom(IterativeSolveStatus.Breakdown, k + 1, in A, in b, ref x, ref tmpM, ref tmpN, xErrBound);

                tau = -tau * delta / gamma;         // forward substitution (uses OLD δ)
                delta = sp * alpha;
                gbar = -cp * alpha;

                // ---- Gauss-Radau σ-QR: advance ω for the error bound (uses γ and the NEW δ) ----
                if (boundOn && !complexBnd)
                {
                    fProxy muBar = -csig * gamma;
                    SymGivens(rhoBar, gamma, out csig, out fProxy ssig, out _);
                    rhoBar = ssig * muBar + csig * se;
                    muBar = -csig * delta;
                    fProxy hh = delta * csig / rhoBar;
                    fProxy disc = se * (se - delta * hh);
                    if (disc < (fProxy)0) complexBnd = true;
                    else omega = math.sqrt(disc);
                    SymGivens(rhoBar, delta, out csig, out ssig, out _);
                    rhoBar = ssig * muBar + csig * se;
                }

                // ---- LQ of the upper bidiagonal: rotate (ε̄, δ) -> (ε, 0) ----
                fProxy ebar = -gamma * c;           // uses OLD c
                fProxy eta = gamma * s;             // uses OLD s
                SymGivens(ebar, delta, out c, out s, out fProxy eps);
                if (eps == (fProxy)0)
                    return LslqInfoFrom(IterativeSolveStatus.Breakdown, k + 1, in A, in b, ref x, ref tmpM, ref tmpN, xErrBound);

                fProxy zetaOld = zeta;
                zeta = (tau - zetaOld * eta) / eps;

                // ---- advance xᴸ and the LQ direction w̄: xᴸ += cζ·w̄ + sζ·v ; w̄ = s·w̄ - c·v ----
                x.addScaledInPlace(c * zeta, wbar);
                x.addScaledInPlace(s * zeta, v);
                wbar.mulInPlace(s);
                wbar.addScaledInPlace(-c, v);
                xlqNorm2 += (double)zeta * (double)zeta;

                // ---- LQ forward-error bound |ζ̃| for the just-updated xᴸ (uses NEW c,s and ω) ----
                if (boundOn)
                {
                    if (!complexBnd)
                    {
                        fProxy etaT = omega * s;
                        fProxy epsT = -omega * c;
                        fProxy tauT = -tau * delta / omega;
                        fProxy zetaT = (tauT - zeta * etaT) / epsT;
                        xErrBound = (double)math.abs(zetaT);
                    }
                    else
                    {
                        xErrBound = double.NaN;
                    }
                }

                // ---- convergence: cheap forward-error trigger arms a certified optimality audit ----
                fProxy xlqNorm = (fProxy)math.sqrt(xlqNorm2);
                bool collapse = !(beta > (fProxy)0) || !(alpha > (fProxy)0);   // NaN-safe
                bool trigger = collapse || (xlqNorm > (fProxy)0 && math.abs(zeta) <= tol * xlqNorm);

                if (trigger)
                {
                    var info = LslqInfoFrom(IterativeSolveStatus.Converged, k + 1, in A, in b, ref x, ref tmpM, ref tmpN, xErrBound);
                    if (info.Arnorm <= tol * atbNorm)
                        return info;
                }

                if (collapse)
                    // bidiagonalization exhausted without reaching the optimality tolerance.
                    return LslqInfoFrom(IterativeSolveStatus.Breakdown, k + 1, in A, in b, ref x, ref tmpM, ref tmpN, xErrBound);
            }

            return LslqInfoFrom(IterativeSolveStatus.MaxIterations, maxIter, in A, in b, ref x, ref tmpM, ref tmpN, xErrBound);
        }

        /// <summary>Assembles the returned <see cref="LslqInfo"/> from a certified-exact least-squares
        /// residual audit (<see cref="lstsqResidual{TOp}"/>: one Apply + one ApplyT) plus the caller's
        /// iteration count, status, and (already-computed) forward-error bound.</summary>
        static LslqInfo LslqInfoFrom<TOp>(IterativeSolveStatus status, int iterations, in TOp A, in fProxyN b,
                                          ref fProxyN x, ref fProxyN rScratch, ref fProxyN sScratch, double xErrBound)
            where TOp : struct, IfProxyLinearOperator
        {
            var r = lstsqResidual(in A, in b, in x, (fProxy)0, ref rScratch, ref sScratch);
            return new LslqInfo
            {
                rnorm = r.rnorm,
                Arnorm = r.Arnorm,
                xnorm = r.xnorm,
                xErrBound = xErrBound,
                iterations = iterations,
                status = status,
            };
        }

        /// <summary>
        /// LSLQ over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc primitive.
        /// Forwards into <see cref="lslq{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LslqInfo lslq(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN wbar,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol, double sigmaMinEst = 0)
        {
            return lslq(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref wbar, ref tmpM, ref tmpN, maxIter, tol, sigmaMinEst);
        }

        /// <summary>LSLQ over a dense matrix -- allocates five scratch vectors from Allocator.Temp.</summary>
        public static LslqInfo lslq(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, double sigmaMinEst = 0)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN wbar = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return lslq(in A, in b, ref x, ref u, ref v, ref wbar, ref tmpM, ref tmpN, maxIter, tol, sigmaMinEst);
        }

        /// <summary>LSLQ over a dense matrix with default maxIter (A.N_Cols -- the bidiagonalization on
        /// a full-column-rank A terminates within n = Cols steps in exact arithmetic) and tol
        /// (Consts.fProxySqrtEps).</summary>
        public static LslqInfo lslq(in fProxyMxN A, in fProxyN b, ref fProxyN x, double sigmaMinEst = 0)
        {
            return lslq(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps, sigmaMinEst);
        }

        /// <summary>
        /// LSLQ over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lslq{TOp}"/> via <c>fProxyBSROperator</c>: matrix-free least
        /// squares over a sparse operator, never forming AᵀA.
        /// </summary>
        public static LslqInfo lslq(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN wbar,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol, double sigmaMinEst = 0)
        {
            return lslq(new fProxyBSROperator(in A), in b, ref x, ref u, ref v, ref wbar, ref tmpM, ref tmpN, maxIter, tol, sigmaMinEst);
        }

        /// <summary>
        /// LSLQ over a BSR matrix -- zero-alloc primitive taking a CALLER-PROVIDED precomputed
        /// transpose AT (e.g. <c>A.Transpose(allocator)</c> built once outside a hot loop) so
        /// every ApplyT routes through the cache-friendly forward spMV(AT, x). Caller owns AT actually
        /// being A's transpose; this overload does not verify it.
        /// </summary>
        public static LslqInfo lslq(in fProxyBSR A, in fProxyBSR AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN wbar,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol, double sigmaMinEst = 0)
        {
            return lslq(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref wbar, ref tmpM, ref tmpN, maxIter, tol, sigmaMinEst);
        }

        /// <summary>
        /// LSLQ over a BSR matrix -- allocates five scratch vectors AND materializes Aᵀ ONCE via
        /// <c>A.Transpose(allocator)</c>, then drives LSLQ with the two-arg <c>fProxyBSROperator</c>.
        /// For a build-free zero-alloc path, build Aᵀ yourself once and call the zero-alloc AT overload.
        /// </summary>
        public static LslqInfo lslq(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol, double sigmaMinEst = 0)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN wbar = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return lslq(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref wbar, ref tmpM, ref tmpN, maxIter, tol, sigmaMinEst);
        }

        /// <summary>LSLQ over a BSR matrix with default maxIter (A.N_Cols) and tol
        /// (Consts.fProxySqrtEps).</summary>
        public static LslqInfo lslq(in fProxyBSR A, in fProxyN b, ref fProxyN x, double sigmaMinEst = 0)
        {
            return lslq(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps, sigmaMinEst);
        }
    }
}
