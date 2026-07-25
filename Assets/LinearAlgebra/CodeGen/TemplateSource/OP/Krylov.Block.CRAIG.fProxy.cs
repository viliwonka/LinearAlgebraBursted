using System;
using Unity.Collections;
using Unity.Mathematics;
using BULA.Sparse;

namespace BULA
{
    public static partial class Krylov {

        // ---- bcraig core ----------------------------------------------------------------------------

        /// <summary>
        /// Block CRAIG (Craig 1955; Paige-Saunders BIT 1995, block-generalized) for a WIDE/
        /// underdetermined operator A (A.Rows &lt;= A.Cols, full row rank) and s simultaneous
        /// CONSISTENT right-hand sides: among all X with A X_j = B_j (per row j), finds the
        /// minimum-Euclidean-norm one, generic over <see cref="IfProxyLinearOperator"/>. Block
        /// generalization of <see cref="craig{TOp}"/>: the same block Golub-Kahan bidiagonalization
        /// blsmr uses (s x s blocks factored via a thin <see cref="LQ.decomp"/> each round) feeds a
        /// block-LOWER-TRIANGULAR forward substitution (one <see cref="QRCP"/> solve per round against
        /// the exactly-triangular LQ factor, not a Givens/QR continuation) that accumulates X as a
        /// direct sum of bidiagonalization blocks -- each one row(A)-spanned, so X stays row(A)-minimal
        /// at every round by construction, mirroring the scalar recurrence's own row(A)-minimality
        /// argument.
        ///
        /// B and X are s ROWS x (A.Rows / A.Cols) COLS respectively (row j = the j-th RHS/solution
        /// vector). X is NOT warm-startable (the min-norm characterization requires X0 = 0, matching
        /// <see cref="craig{TOp}"/>); this overload always overwrites X from zero. Owns its whole
        /// workspace via Allocator.Temp (no external scratch params, mirroring <see cref="blsmr{TOp}"/>).
        ///
        /// Breakdown (<see cref="IterativeSolveStatus.Breakdown"/>) on ANY block-bidiagonalization LQ
        /// factor going numerically singular (A lacks full row rank, or the block RHS B lacks full row
        /// rank) or on the per-round triangular solve losing full rank. Never NaN, never a false
        /// Converged. Convergence is checked against a fresh block residual ‖B - A X‖_F each round (no
        /// free per-round identity is used here, unlike scalar craig's ‖beta*z‖ shortcut -- the block
        /// forward-substitution has no published closed-form residual recurrence; see the folder
        /// DEVLOG). <see cref="BlockSolveInfo.minActive"/> is always s (no column deflation; the
        /// stopping test is a single joint Frobenius-norm criterion, matching blsmr).
        /// </summary>
        public static BlockSolveInfo bcraig<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows > A.Cols) throw new ArgumentException("bcraig: A must be square or underdetermined (A.Rows <= A.Cols)");
            int m = A.Rows, n = A.Cols;
            int s = B.M_Rows;
            if (B.N_Cols != m) throw new ArgumentException("bcraig: B.N_Cols must equal A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("bcraig: X must be s x A.Cols");
            if (s < 1 || s > m) throw new ArgumentException("bcraig: B.M_Rows (s) must be in [1, A.Rows]");
            if (maxIter < 1) throw new ArgumentException("bcraig: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[2];
                ptrs[0] = (long)X.Data.Ptr; ptrs[1] = (long)B.Data.Ptr;
                RequireDistinctBuffers("bcraig: X/B must be distinct", ptrs, 2);
            }

            ZeroPrefix(ref X, s, n);

            fProxyN rowN = new fProxyN(n), rowM = new fProxyN(m);

            // ---- persistent state: carried across rounds, updated by explicit copy at the end of
            // each round (never pointer-swapped -- s is small, copy cost is trivial). ----
            var LA    = new fProxyMxN(s, s, Allocator.Temp, true);   // current L_j (block analog of alpha_j)
            var Yprev = new fProxyMxN(s, s, Allocator.Temp, true);   // current Y_j (CRAIG coefficient block)
            var U     = new fProxyMxN(s, m, Allocator.Temp, true);   // current U_j
            var V     = new fProxyMxN(s, n, Allocator.Temp, true);   // current V_j

            // ---- per-round scratch ----
            var LBnew    = new fProxyMxN(s, s, Allocator.Temp, true);
            var LAnew    = new fProxyMxN(s, s, Allocator.Temp, true);
            var Ynew     = new fProxyMxN(s, s, Allocator.Temp, true);
            var termSS   = new fProxyMxN(s, s, Allocator.Temp, true);
            var coefWork = new fProxyMxN(s, s, Allocator.Temp, true);
            var Rqrcp    = new fProxyMxN(s, s, Allocator.Temp, true);
            var Pqrcp    = new Pivot(s, Allocator.Temp);
            var uQrcp    = new fProxyN(s);

            var Wbar   = new fProxyMxN(s, m, Allocator.Temp, true);
            var termSM = new fProxyMxN(s, m, Allocator.Temp, true);
            var Rfinal = new fProxyMxN(s, m, Allocator.Temp, true);
            var Sbar   = new fProxyMxN(s, n, Allocator.Temp, true);
            var Vnext  = new fProxyMxN(s, n, Allocator.Temp, true);
            var termSN = new fProxyMxN(s, n, Allocator.Temp, true);

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int iters = 0;

            // ---- tolerance scale: ||B||_F^2 (matches scalar craig's ||b||-based stopping test) ----
            fProxy bSqF = BlockFrobDot(in B, in B);
            if (bSqF == (fProxy)0)
            {
                // B = 0: the min-norm solution is trivially X = 0.
                status = IterativeSolveStatus.Converged;
                goto cleanup;
            }
            fProxy threshold = tol * tol * bSqF;

            // ---- round 1 (init): thin LQ of B, then of A^T U1, then the first triangular solve ----
            {
                var LB1 = new fProxyMxN(s, s, Allocator.Temp, true);
                LQ.decomp(in B, ref LB1, ref U);
                if (TriNearSingular(in LB1, s)) { LB1.Dispose(); status = IterativeSolveStatus.Breakdown; goto cleanup; }

                BlockApplyOpT(in A, in U, ref Sbar, s, ref rowN, ref rowM);   // Sbar = U1 @ A
                LQ.decomp(in Sbar, ref LA, ref V);
                if (TriNearSingular(in LA, s)) { LB1.Dispose(); status = IterativeSolveStatus.Breakdown; goto cleanup; }

                // L_1 Y_1 = R_1 = LB1^T (our LQ-row convention transposes the classical upper-tri R_1)
                TransposeSmall(in LB1, ref termSS, s);
                var rankY1 = BlockSolveGeneralWide(in LA, ref termSS, ref Yprev, ref coefWork, ref Rqrcp, ref Pqrcp, ref uQrcp, s);
                LB1.Dispose();
                if (rankY1.status != DirectSolveStatus.Success || rankY1.rank != s)
                { status = IterativeSolveStatus.Breakdown; goto cleanup; }

                BlockCTV(in Yprev, in V, ref termSN);
                BlockAdd(ref X, in termSN, (fProxy)1);
            }

            {
                BlockApplyOp(in A, in X, ref Rfinal, s, ref rowN, ref rowM);
                BlockScaleInPlace(ref Rfinal, (fProxy)(-1));
                BlockAdd(ref Rfinal, in B, (fProxy)1);           // Rfinal = B - A X
                fProxy rSqF = BlockFrobDot(in Rfinal, in Rfinal);
                if (rSqF <= threshold) { status = IterativeSolveStatus.Converged; goto cleanup; }
            }

            for (int k = 0; k < maxIter; k++)
            {
                // Step A (relation *): LBnew, Unew from A V_j - LA^T U_j
                BlockApplyOp(in A, in V, ref Wbar, s, ref rowN, ref rowM);
                BlockCTV(in LA, in U, ref termSM);
                BlockAdd(ref Wbar, in termSM, (fProxy)(-1));

                LQ.decomp(in Wbar, ref LBnew, ref U);
                if (TriNearSingular(in LBnew, s)) { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                // Step B (relation **): LAnew, Vnext from A^T U_{j+1} - LBnew^T V_j
                BlockApplyOpT(in A, in U, ref Sbar, s, ref rowN, ref rowM);
                BlockCTV(in LBnew, in V, ref termSN);
                BlockAdd(ref Sbar, in termSN, (fProxy)(-1));

                LQ.decomp(in Sbar, ref LAnew, ref Vnext);
                if (TriNearSingular(in LAnew, s)) { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                // L_{j+1} Y_{j+1} = -R_{j+1} Y_j = -LBnew^T @ Yprev
                BlockCTV(in LBnew, in Yprev, ref termSS);
                BlockScaleInPlace(ref termSS, (fProxy)(-1));
                var rankY = BlockSolveGeneralWide(in LAnew, ref termSS, ref Ynew, ref coefWork, ref Rqrcp, ref Pqrcp, ref uQrcp, s);
                if (rankY.status != DirectSolveStatus.Success || rankY.rank != s)
                { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                BlockCTV(in Ynew, in Vnext, ref termSN);
                BlockAdd(ref X, in termSN, (fProxy)1);
                iters = k + 1;

                CopyMat(in LAnew, ref LA, s);
                CopyMat(in Ynew, ref Yprev, s);
                CopyBlock(in Vnext, ref V, s, n);
                // U was already overwritten in place by this round's first LQ.decomp call.

                BlockApplyOp(in A, in X, ref Rfinal, s, ref rowN, ref rowM);
                BlockScaleInPlace(ref Rfinal, (fProxy)(-1));
                BlockAdd(ref Rfinal, in B, (fProxy)1);
                fProxy rSqF = BlockFrobDot(in Rfinal, in Rfinal);
                if (rSqF <= threshold) { status = IterativeSolveStatus.Converged; goto cleanup; }
            }

        cleanup:
            {
                BlockLstsqExit(in A, in X, in B, ref Rfinal, status, s, m, ref rowN, ref rowM, out double maxr, out int converged);

                rowN.Dispose(); rowM.Dispose();
                LA.Dispose(); Yprev.Dispose(); U.Dispose(); V.Dispose();
                LBnew.Dispose(); LAnew.Dispose(); Ynew.Dispose(); termSS.Dispose();
                coefWork.Dispose(); Rqrcp.Dispose(); Pqrcp.Dispose(); uQrcp.Dispose();
                Wbar.Dispose(); termSM.Dispose(); Rfinal.Dispose(); Sbar.Dispose(); Vnext.Dispose(); termSN.Dispose();

                return new BlockSolveInfo { rhs = s, converged = converged, iterations = iters, maxRnorm = maxr, minActive = s, status = status };
            }
        }

        // ---- bcraig concrete forwarders ----------------------------------------------------------

        /// <summary>Block CRAIG over a dense <see cref="fProxyMxN"/> A (wide or square, full row
        /// rank). Forwards into <see cref="bcraig{TOp}"/> via <see cref="fProxyDenseOperator"/>.</summary>
        public static BlockSolveInfo bcraig(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            => bcraig(new fProxyDenseOperator(in A), in B, ref X, maxIter, tol);

        /// <summary>Block CRAIG over a dense A with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).</summary>
        public static BlockSolveInfo bcraig(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => bcraig(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);
    }
}
