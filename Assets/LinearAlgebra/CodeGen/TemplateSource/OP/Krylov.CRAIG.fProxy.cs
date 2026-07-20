using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        /// <summary>
        /// Zero-alloc CRAIG (Craig 1955; Paige-Saunders, BIT 1995) solver for UNDERDETERMINED
        /// consistent systems: among all x with A x = b, finds the one of minimum Euclidean norm,
        /// x* = Aᵀ(AAᵀ)⁻¹b, generic over <see cref="IfProxyLinearOperator"/>. Requires A (Rows ≤
        /// Cols) to have full row rank and b ∈ range(A) (a consistent system) -- on a square
        /// nonsingular A it recovers the unique solution. Builds the same Golub-Kahan
        /// bidiagonalization as <see cref="lsqr{TOp}"/>, but instead of folding it through a
        /// Givens-rotated QR (LSQR's least-squares path), solves the resulting lower-bidiagonal
        /// system by forward substitution and accumulates x as a direct sum of bidiagonalization
        /// vectors -- each one an Aᵀ-image, so x stays in row(A) (hence row(A)-minimal) at every
        /// iteration by construction. O(n+m) memory and per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// No warm start (unlike lsqr/lsmr): the min-norm characterization only holds starting from
        /// x₀ = 0, so x is zeroed internally regardless of its incoming contents.
        ///
        /// Converges when ‖b-Ax‖ &lt;= tol*‖b‖. Returns an <see cref="LstsqInfo"/> — see that
        /// struct for the implicit-bool/status/undefined-x contract; rnorm/Arnorm/xnorm are filled
        /// by a certified-exact <see cref="lstsqResidual{TOp}"/> audit (one extra Apply + ApplyT)
        /// rather than a tracked identity, since CRAIG's direct rank-1 x update has no free
        /// per-iteration ‖Aᵀr‖ the way lsqr/lsmr's rotation recurrences do. A claimed Converged is
        /// reconciled against that certified rnorm (no extra matvec beyond the audit already paid
        /// for) before being reported -- falls through to another iteration if the tracked estimate
        /// turned out optimistic. Breakdown when the bidiagonalization collapses before ‖b-Ax‖
        /// reaches tolerance -- A lacks full row rank (or, on the very first step, b lies outside its
        /// row space).
        /// </summary>
        public static LstsqInfo craig<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows > A.Cols) throw new ArgumentException("craig: A must be square or underdetermined (Rows <= Cols)");
            if (b.N != A.Rows) throw new ArgumentException("craig: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("craig: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("craig: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("craig: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("craig: v.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("craig: tmpN.N must equal A.Cols");

            if (maxIter < 1)
                throw new ArgumentException("craig: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[6];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr;
                ptrs[2] = (long)tmpM.Data.Ptr; ptrs[3] = (long)tmpN.Data.Ptr;
                ptrs[4] = (long)x.Data.Ptr; ptrs[5] = (long)b.Data.Ptr;
                RequireDistinctBuffers("craig: u/v/tmpM/tmpN/x/b must be distinct", ptrs, 6);
            }

            // No warm start: the min-norm characterization x = Aᵀ(AAᵀ)⁻¹b requires x₀ = 0.
            for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;

            fProxy bnorm = math.sqrt(Blas.dot(b, b));
            if (bnorm == (fProxy)0)
                // b = 0: the min-norm solution is trivially x = 0.
                return CraigInfo(IterativeSolveStatus.Converged, 0, in A, in b, ref x, ref tmpM, ref tmpN);

            // beta_1 u_1 = b
            u.CopyFrom(in b);
            u.divInPlace(bnorm);
            fProxy beta = bnorm;

            // v_0 = 0 bootstraps the first bidiagonalization step: alpha_1 v_1 = Aᵀu_1 - beta_1·0.
            for (int j = 0; j < v.N; j++) v[j] = (fProxy)0;
            // Dummy pre-loop value; the z-recurrence below produces z_1 = beta_1/alpha_1 on the
            // first iteration (the sign flip cancels the leading minus).
            fProxy z = (fProxy)(-1);

            for (int k = 0; k < maxIter; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                fProxy alfa = GolubKahanVStep(in A, in u, beta, ref tmpN, ref v);

                if (!(alfa > (fProxy)0)) // NaN-safe: alfa is a norm, nonnegative
                    // v collapsed: the Krylov space on AAᵀ is exhausted before reaching b -- A is
                    // not full row rank (or, on the first step, b ∉ range(A)). x is the previous
                    // iterate (undefined per the Breakdown contract).
                    return CraigInfo(IterativeSolveStatus.Breakdown, k + 1, in A, in b, ref x, ref tmpM, ref tmpN);

                v.divInPlace(alfa);

                // Forward-substitution step of the lower-bidiagonal system L z = beta_1 e_1, folded
                // into a direct rank-1 update of x = Σ z_i v_i -- each v_i is an Aᵀ-image, so x stays
                // in row(A) (hence row(A)-minimal) at every iteration. No Givens rotation / momentum
                // buffer needed here, unlike lsqr/lsmr's w-based update.
                z = -(beta / alfa) * z;
                x.addScaledInPlace(z, v);

                // u = A v - alpha*u ; beta = ||u||, same fusion.
                beta = GolubKahanUStep(in A, in v, alfa, ref tmpM, ref u);

                // ‖b - A x‖ for the just-updated x, free from the same recurrence (Paige-Saunders
                // 1995) -- no extra matvec.
                fProxy rnorm = math.abs(beta * z);

                if (rnorm <= tol * bnorm)
                {
                    // Verify-at-exit: the tracked bidiagonalization estimate can drift from the true
                    // residual on an ill-conditioned A (CRAIG is CG-on-AAᵀ in disguise, kappa²
                    // sensitivity). CraigInfo already pays for a certified-exact ‖b-Ax‖ audit
                    // (lstsqResidual: one Apply + one ApplyT) for every exit regardless of status --
                    // reuse THAT instead of a redundant matvec: only commit to Converged if the
                    // certified residual also clears the threshold. tmpM/tmpN are safe to reuse
                    // either way (GolubKahanUStep/VStep fully overwrite them before their next read,
                    // so a fall-through loses nothing).
                    var info = CraigInfo(IterativeSolveStatus.Converged, k + 1, in A, in b, ref x, ref tmpM, ref tmpN);
                    if (info.rnorm <= tol * bnorm)
                        return info;
                }

                if (!(beta > (fProxy)0)) // NaN-safe: beta is a norm, nonnegative
                    // u collapsed without reaching tolerance. Only reachable with a degenerate tol
                    // (beta == 0 forces rnorm == 0 above, which converges for any tol >= 0) --
                    // guards the division below from ever seeing a zero.
                    return CraigInfo(IterativeSolveStatus.Breakdown, k + 1, in A, in b, ref x, ref tmpM, ref tmpN);

                u.divInPlace(beta);
            }

            return CraigInfo(IterativeSolveStatus.MaxIterations, maxIter, in A, in b, ref x, ref tmpM, ref tmpN);
        }

        /// <summary>Assembles the returned <see cref="LstsqInfo"/> from a certified-exact residual
        /// audit (<see cref="lstsqResidual{TOp}"/>: one Apply + one ApplyT) plus the caller's
        /// iteration count and status. <paramref name="rScratch"/>/<paramref name="sScratch"/> are
        /// the solver's own tmpM/tmpN buffers, free to reuse post-update.</summary>
        static LstsqInfo CraigInfo<TOp>(IterativeSolveStatus status, int iterations, in TOp A, in fProxyN b,
                                         ref fProxyN x, ref fProxyN rScratch, ref fProxyN sScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            var info = lstsqResidual(in A, in b, in x, (fProxy)0, ref rScratch, ref sScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        /// <summary>
        /// CRAIG over a dense <see cref="fProxyMxN"/> (Rows ≤ Cols) -- zero-alloc primitive.
        /// Forwards into <see cref="craig{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo craig(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return craig(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>CRAIG over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static LstsqInfo craig(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            return craig(in A, in b, ref x, ref u, ref v, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>CRAIG over a dense matrix with default maxIter (A.M_Rows -- the bidiagonalization
        /// on a full-row-rank A terminates within m = Rows steps in exact arithmetic, unlike
        /// lsqr/lsmr's N_Cols-bounded search) and tol (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo craig(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return craig(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// CRAIG over a (possibly rectangular, Rows ≤ Cols) block-sparse (BSR) matrix -- zero-alloc
        /// primitive. Forwards into <see cref="craig{TOp}"/> via <c>fProxyBSROperator</c>: matrix-free
        /// minimum-norm solve over a sparse operator, never forming AAᵀ.
        /// </summary>
        public static LstsqInfo craig(in fProxyBSR A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return craig(new fProxyBSROperator(in A), in b, ref x, ref u, ref v, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// CRAIG over a BSR matrix -- zero-alloc primitive variant that takes a CALLER-PROVIDED
        /// precomputed transpose AT (e.g. built once via <c>arena.fProxyBSRTranspose(in A)</c> outside
        /// a hot loop) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSROperator"/>'s two-arg ctor. Caller is responsible for AT actually being
        /// A's transpose; this overload does not verify it.
        /// </summary>
        public static LstsqInfo craig(in fProxyBSR A, in fProxyBSR AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIter, fProxy tol)
        {
            return craig(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>
        /// CRAIG over a BSR matrix -- allocates four scratch vectors AND materializes A^T ONCE via
        /// <c>arena.fProxyBSRTranspose</c>, then drives CRAIG with the two-arg
        /// <see cref="fProxyBSROperator"/> so every ApplyT call routes through a cache-friendly
        /// forward spMV(A^T, x) instead of scatter-heavy spMVT(A, x) every iteration. For a
        /// build-free zero-alloc path, build A^T yourself once and call the zero-alloc
        /// <see cref="craig(in fProxyBSR, in fProxyBSR, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>
        /// overload above with your own scratch vectors.
        /// </summary>
        public static LstsqInfo craig(in fProxyBSR A, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            fProxyN u    = b.fProxyTempVec(A.M_Rows);
            fProxyN v    = b.fProxyTempVec(A.N_Cols);
            fProxyN tmpM = b.fProxyTempVec(A.M_Rows);
            fProxyN tmpN = b.fProxyTempVec(A.N_Cols);
            fProxyBSR AT = b.fProxyBSRTranspose(in A);
            return craig(new fProxyBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref tmpM, ref tmpN, maxIter, tol);
        }

        /// <summary>CRAIG over a BSR matrix with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps) -- see the dense default overload for the M_Rows rationale.</summary>
        public static LstsqInfo craig(in fProxyBSR A, in fProxyN b, ref fProxyN x)
        {
            return craig(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
