using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// Zero-alloc CRAIGMR (MINRES-flavored CRAIG; monotonic residual) solver for UNDERDETERMINED
        /// consistent systems: among all x with A x = b, finds the one of minimum Euclidean norm,
        /// generic over <see cref="IfProxyLinearOperator"/>. Requires A (Rows &lt;= Cols) to have
        /// full row rank and b in range(A) (a consistent system) -- on a square nonsingular A it
        /// recovers the unique solution. CRAIGMR is to <see cref="craig{TOp}"/> what
        /// <see cref="lsmr{TOp}"/> is to <see cref="lsqr{TOp}"/>: the same Golub-Kahan
        /// bidiagonalization, but the lower-bidiagonal system is solved via a running QR
        /// factorization (one Givens rotation per step) instead of craig's direct forward
        /// substitution, so ‖b-Ax‖ decreases MONOTONICALLY -- craig's error decreases monotonically
        /// but its residual can bounce. x is accumulated as x = Σ ζᵢ dᵢ where each dᵢ is a running
        /// combination of Aᵀ-images (v₁..vᵢ), so x stays in row(A) (hence row(A)-minimal) at every
        /// iteration by construction. O(n+m) memory and per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// No warm start (same as craig): the min-norm characterization only holds starting from
        /// x₀ = 0, so x is zeroed internally regardless of its incoming contents.
        ///
        /// Converges when ‖b-Ax‖ &lt;= tol*‖b‖. Returns an <see cref="LstsqInfo"/> -- see that
        /// struct for the implicit-bool/status/undefined-x contract. Unlike craig, rnorm/Arnorm fall
        /// out of the same QR recurrence that drives x for FREE (no extra Apply/ApplyT audit).
        /// Breakdown when the bidiagonalization or the QR rotation radius collapses before ‖b-Ax‖
        /// reaches tolerance -- A lacks full row rank (or, on the very first step, b lies outside its
        /// row space).
        /// </summary>
        public static LstsqInfo craigmr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN d,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows > A.Cols) throw new ArgumentException("craigmr: A must be square or underdetermined (Rows <= Cols)");
            if (b.N != A.Rows) throw new ArgumentException("craigmr: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("craigmr: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("craigmr: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("craigmr: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("craigmr: v.N must equal A.Cols");
            if (d.N != A.Cols) throw new ArgumentException("craigmr: d.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("craigmr: tmpN.N must equal A.Cols");

            if (maxIter < 1)
                throw new ArgumentException("craigmr: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[7];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr; ptrs[2] = (long)d.Data.Ptr;
                ptrs[3] = (long)tmpM.Data.Ptr; ptrs[4] = (long)tmpN.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                RequireDistinctBuffers("craigmr: u/v/d/tmpM/tmpN/x/b must be distinct", ptrs, 7);
            }

            // No warm start: the min-norm characterization x = Aᵀ(AAᵀ)⁻¹b requires x₀ = 0.
            for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
            for (int i = 0; i < d.N; i++) d[i] = (fProxy)0;

            fProxy bnorm = math.sqrt(Blas.dot(b, b));
            if (bnorm == (fProxy)0)
                // b = 0: the min-norm solution is trivially x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (fProxy)0, (fProxy)0, (fProxy)0, in x);

            // beta_1 u_1 = b
            u.CopyFrom(in b);
            u.divInPlace(bnorm);
            fProxy beta = bnorm;

            // alpha_1 v_1 = A^T u_1
            A.ApplyT(in u, ref tmpN);
            v.CopyFrom(in tmpN);
            fProxy alpha = math.sqrt(Blas.dot(v, v));

            if (!(alpha > (fProxy)0)) // NaN-safe: alpha is a norm, nonnegative
                // v collapsed on the first step: b is orthogonal to range(A) -- A is not full row
                // rank (or b lies outside its row space). x = 0 (Breakdown contract: undefined).
                return LstsqInfoTracked(IterativeSolveStatus.Breakdown, 0, beta, (fProxy)0, (fProxy)0, in x);

            v.divInPlace(alpha);

            // Running QR state of the CRAIG lower-bidiagonal system L (diag alpha, subdiag beta):
            // rhobar carries the previous alpha into the next Givens elimination, zetabar carries
            // the residual weight, theta feeds the next d update. Iterate-0 values (x=0): rNorm =
            // ‖b-Ax‖ = beta, ArNorm = ‖Aᵀ(b-Ax)‖ = ‖Aᵀb‖ = alpha*beta.
            fProxy zetabar = beta;
            fProxy rhobar = alpha;
            fProxy theta = (fProxy)0;
            fProxy rNorm = beta;
            fProxy ArNorm = alpha * beta;

            for (int k = 0; k < maxIter; k++)
            {
                // ---- extend the bidiagonalization forward: beta_{k+1} u_{k+1} = A v_k - alpha_k u_k. ----
                fProxy betaNew = GolubKahanUStep(in A, in v, alpha, ref tmpM, ref u);

                // ---- Givens rotation continuing the running QR of L: eliminates betaNew against
                // the current rhobar (rhobar can be negative -- it is a signed R-factor entry, not
                // a magnitude). ----
                fProxy rho = math.sqrt(rhobar * rhobar + betaNew * betaNew);
                if (!(rho > (fProxy)0)) // NaN-safe: both terms are squared, sum is nonnegative
                    // Both the running rotation radius and the new beta collapsed: the QR
                    // continuation can make no further progress. x/rNorm/ArNorm describe the
                    // previous (last successfully updated) iterate.
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, rNorm, ArNorm, (fProxy)0, in x);

                fProxy c = rhobar / rho;
                fProxy s = betaNew / rho;

                fProxy zetaOld = zetabar;
                fProxy zeta = c * zetaOld;
                zetabar = s * zetaOld;
                rNorm = math.abs(zetabar);

                // ---- x/d update: d is a running Aᵀ-image combination of v_1..v_k (row(A)-minimal
                // by construction, CRAIGMR's analog of craig's direct v-weighted sum), using the
                // CURRENT v (= v_k, not yet overwritten -- the backward bidiagonalization step below
                // only reads it). ----
                d.scaleAddInPlace(-theta, v);   // d = -theta*d + v
                d.divInPlace(rho);
                x.addScaledInPlace(zeta, d);

                if (rNorm <= tol * bnorm)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, rNorm, ArNorm, (fProxy)0, in x);

                if (!(betaNew > (fProxy)0)) // NaN-safe: betaNew is a norm, nonnegative
                    // Bidiagonalization exhausted on the forward side without reaching tol.
                    // Unreachable in practice: betaNew == 0 forces rho = |rhobar|, s = 0, zetabar =
                    // 0, rNorm = 0, which converges above for any tol >= 0 -- kept as a defensive
                    // guard on the division below.
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, rNorm, ArNorm, (fProxy)0, in x);

                u.divInPlace(betaNew);

                // ---- extend the bidiagonalization backward: alpha_{k+1} v_{k+1} = A^T u_{k+1} -
                // beta_{k+1} v_k (overwrites v_k -> v_{k+1}). ----
                fProxy alphaNew = GolubKahanVStep(in A, in u, betaNew, ref tmpN, ref v);

                // ‖Aᵀ(b-Ax)‖ for the just-updated x, free from the same recurrence -- no extra matvec.
                ArNorm = alphaNew * betaNew * math.abs(zeta) / rho;

                if (!(alphaNew > (fProxy)0)) // NaN-safe: alphaNew is a norm, nonnegative
                    // Krylov space exhausted on the backward side: A is not full row rank beyond
                    // this point.
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, rNorm, ArNorm, (fProxy)0, in x);

                v.divInPlace(alphaNew);

                // ---- prepare next rotation ----
                theta = s * alphaNew;
                rhobar = -c * alphaNew;
                alpha = alphaNew;
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIter, rNorm, ArNorm, (fProxy)0, in x);
        }

        /// <summary>
        /// CRAIGMR over a dense <see cref="fProxyMxN"/> (Rows &lt;= Cols) -- zero-alloc primitive.
        /// Forwards into <see cref="craigmr{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo craigmr(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN d,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return craigmr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref d, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>CRAIGMR over a dense matrix -- allocates five scratch vectors from Allocator.Temp.</summary>
        public static LstsqInfo craigmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN d    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return craigmr(in A, in b, ref x, ref u, ref v, ref d, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>CRAIGMR over a dense matrix with default maxIter (A.M_Rows -- the
        /// bidiagonalization on a full-row-rank A terminates within m = Rows steps in exact
        /// arithmetic, same bound as craig) and tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo craigmr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return craigmr(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// CRAIGMR over a (possibly rectangular, Rows &lt;= Cols) block-sparse (BSR) matrix --
        /// zero-alloc primitive. Forwards into <see cref="craigmr{TOp}"/> via
        /// <c>fProxyBSROperator</c>: matrix-free minimum-norm solve over a sparse operator, never
        /// forming AAᵀ.
        /// </summary>
        public static LstsqInfo craigmr(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN d,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return craigmr(new fProxyBSROperator(in A), in b, ref x, ref u, ref v, ref d, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// CRAIGMR over a BSR matrix -- zero-alloc primitive variant that takes a CALLER-PROVIDED
        /// precomputed transpose AT (e.g. built once via <c>A.Transpose(allocator)</c>
        /// outside a hot loop) and routes every ApplyT call through the resulting cache-friendly
        /// forward spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSROperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it.
        /// </summary>
        public static LstsqInfo craigmr(in fProxyBSR A, in fProxyBSR AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN d,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return craigmr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref d, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// CRAIGMR over a BSR matrix -- allocates five scratch vectors AND materializes A^T ONCE via
        /// <c>A.Transpose(allocator)</c>, then drives CRAIGMR with the two-arg
        /// <see cref="fProxyBSROperator"/> so every ApplyT call routes through a cache-friendly
        /// forward spMV(A^T, x) instead of scatter-heavy spMVT(A, x) every iteration. For a
        /// build-free zero-alloc path, build A^T yourself once and call the zero-alloc
        /// <see cref="craigmr(in fProxyBSR, in fProxyBSR, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// overload above with your own scratch vectors.
        /// </summary>
        public static LstsqInfo craigmr(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN d    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return craigmr(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref d, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>CRAIGMR over a BSR matrix with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps) -- see the dense default overload for the M_Rows rationale.</summary>
        public static LstsqInfo craigmr(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return craigmr(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
