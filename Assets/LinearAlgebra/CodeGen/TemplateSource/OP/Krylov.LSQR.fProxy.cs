using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

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
            u.CopyFrom(in b);
            u.addScaledInPlace((fProxy)(-1), tmpM);

            fProxy beta = math.sqrt(Blas.dot(u, u));

            if (beta == (fProxy)0)
                // x already exact (r = 0): rnorm = 0, Aᵀr = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (fProxy)0, (fProxy)0, (fProxy)0, in x);

            u.divInPlace(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.CopyFrom(in tmpN);

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

            w.CopyFrom(in v);

            for (int k = 0; k < maxIter; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                beta = GolubKahanUStep(in A, in v, alpha, ref tmpM, ref u);
                if (beta > (fProxy)0) u.divInPlace(beta);

                alpha = GolubKahanVStep(in A, in u, beta, ref tmpN, ref v);
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

                arnorm = math.abs(phibar) * alpha * math.abs(c);  // ‖Aᵀr‖ for the just-updated x (free) --
                // phibar is a Givens-rotated RHS carry, not a norm, and damping's own rotation
                // (rhobar1, above) can flip its sign; alpha is a bidiagonalization norm (>= 0).

                if (arnorm * arnorm <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                if (!(beta > (fProxy)0) || !(alpha > (fProxy)0)) // NaN-safe: both are norms, nonnegative
                    // bidiagonalization breakdown: Krylov space exhausted, no further progress
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);
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
    }
}
