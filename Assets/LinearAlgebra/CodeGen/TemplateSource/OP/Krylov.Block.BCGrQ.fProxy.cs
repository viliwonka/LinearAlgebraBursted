using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ---- bcgrq core -------------------------------------------------------------------------------

        /// <summary>
        /// Zero-alloc block (multi-RHS) Conjugate Gradient for an SPD A and s simultaneous right-hand
        /// sides, generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>) and the
        /// preconditioner (<see cref="IfProxyPreconditioner"/>). Replaces ridge-regularized
        /// <see cref="bcg{TOp, TPre}(in TOp, in TPre, in fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, int, fProxy)"/>'s
        /// s x s Gram solve with a row-pivoted rank-revealing LQ (<see cref="LQRP"/>) factorization of the
        /// (preconditioned) live residual block every iteration, so near-dependent RHS directions are
        /// DROPPED (deflated) from the search subspace rather than ridge-patched -- every still-live
        /// column keeps receiving an X/R update every iteration regardless of the deflated search width.
        ///
        /// B and X are s ROWS x n COLS (row j = the j-th RHS / solution, length n = A.Rows, requires
        /// s &lt;= n); X is warm-startable and its rows are NEVER reordered. R, P, AP, Pa are s x n block
        /// scratch (Pa receives LQRP's orthonormal-rows output); Z is s x n block scratch, required (and
        /// only touched) when <c>!M.IsIdentity</c> -- pass <c>default</c> otherwise. Convergence is per
        /// original column against tol²·‖B[j]‖². Returns a <see cref="BlockSolveInfo"/> whose
        /// <see cref="BlockSolveInfo.minActive"/> is the smallest numerical rank the live residual block
        /// reached over the whole solve (search-subspace deflation), independent of column convergence.
        /// </summary>
        public static BlockSolveInfo bcgrq<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        ref fProxyMxN R, ref fProxyMxN P, ref fProxyMxN AP, ref fProxyMxN Pa,
                                        ref fProxyMxN Z, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("bcgrq (block): A must be square");
            int n = A.Rows;
            int s = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("bcgrq (block): B must be s x A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("bcgrq (block): X must match B");
            if (R.M_Rows != s || R.N_Cols != n) throw new ArgumentException("bcgrq (block): R must match B");
            if (P.M_Rows != s || P.N_Cols != n) throw new ArgumentException("bcgrq (block): P must match B");
            if (AP.M_Rows != s || AP.N_Cols != n) throw new ArgumentException("bcgrq (block): AP must match B");
            if (Pa.M_Rows != s || Pa.N_Cols != n) throw new ArgumentException("bcgrq (block): Pa must match B");
            if (!M.IsIdentity && (Z.M_Rows != s || Z.N_Cols != n))
                throw new ArgumentException("bcgrq (block): Z must match B");
            if (maxIter < 1) throw new ArgumentException("bcgrq (block): maxIter must be >= 1");
            if (s > n) throw new ArgumentException("bcgrq: B.M_Rows (s) must be <= A.Rows (n)");
            if (!M.IsSpd)
                throw new ArgumentException("Krylov.bcgrq: requires an SPD preconditioner (M.IsSpd == false — e.g. ILU0/SPAI/restricted-Schwarz). Use a non-symmetric solver (gmres/biCGStab) for a general preconditioner.");
            if (!M.IsConstant)
                throw new ArgumentException("Krylov.bcgrq: requires a constant (non-flexible) preconditioner (M.IsConstant == false — e.g. an AMG K-cycle). Use the flexible variant (fcg / fgmres).");

            // s x s coefficient scratch (max size; narrowed per iteration via View/RectView) + per-
            // original-column thresholds + row scratch for the preconditioner + the persistent slot
            // pivot tracking "physical live row -> original RHS index".
            var thr      = new fProxyN(s);
            var Live     = new Pivot(s, Allocator.Temp);
            var alphaBuf = new fProxyMxN(s, s, Allocator.Temp, true);
            var betaBuf  = new fProxyMxN(s, s, Allocator.Temp, true);
            var PQbuf    = new fProxyMxN(s, s, Allocator.Temp, true);
            var Lbuf     = new fProxyMxN(s, s, Allocator.Temp, true);
            var workBuf  = new fProxyMxN(s, s, Allocator.Temp, true);
            var Tbuf     = new fProxyMxN(s, n, Allocator.Temp, true);
            fProxyN rowIn = default, rowOut = default;
            if (!M.IsIdentity) { rowIn = new fProxyN(n); rowOut = new fProxyN(n); }

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int iters = maxIter;
            int converged = 0;
            double maxr = 0;
            int sLive = s;
            int minActive = s;
            int saSearch = 0;

            // Per-original-column thresholds tol^2 ||B[j]||^2, computed once before any permutation.
            BuildColumnThresholdsPlain(in B, ref thr, s, n, tol);

            // R = B - A X (AP reused as scratch, mirroring bcg's own reuse of Q).
            BlockResidual(in A, in X, in B, ref AP, ref R, s, n);

            LockConvergedRows(ref R, ref Live, ref sLive, in thr);
            if (sLive == 0) { status = IterativeSolveStatus.Converged; iters = 0; goto cleanup; }

            // ---- setup: establish iteration 0's P_search/AP_search/PQ_search ----
            {
                int sa = FactorLiveResidual(in R, in M, ref Z, sLive, n, ref rowIn, ref rowOut, ref Lbuf, ref Pa);
                minActive = math.min(minActive, sa);
                if (sa == 0) { status = IterativeSolveStatus.Breakdown; iters = 0; goto cleanup; }

                saSearch = sa;
                var Psearch  = RowsView(P, saSearch);
                var PaActive = RowsView(Pa, saSearch);
                CopyBlock(in PaActive, ref Psearch, saSearch, n);

                var APsearch = RowsView(AP, saSearch);
                A.ApplyBlock(in Psearch, ref APsearch, saSearch);

                var PQsetup = View(PQbuf, saSearch);
                BlockGram(in Psearch, in APsearch, ref PQsetup, saSearch);
            }

            for (int k = 0; k < maxIter; k++)
            {
                // Precondition: P, AP, PQ already hold this iteration's P_search/AP_search/PQ_search at
                // width saSearch (from the setup block above, or from the previous iteration's S11-S13).
                var Psearch  = RowsView(P, saSearch);
                var APsearch = RowsView(AP, saSearch);
                var PQ       = View(PQbuf, saSearch);

                // S2/S3: alpha = (P_search^T A P_search)^-1 (P_search R_live^T), saSearch x sLive.
                var Rlive = RowsView(R, sLive);
                var alpha = RectView(alphaBuf, saSearch, sLive);
                Blas.dot(in Psearch, in Rlive, ref alpha, false, true);
                var work = View(workBuf, saSearch);
                if (!BlockSolveSPD(in PQ, ref alpha, ref work, saSearch))
                { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                // S4: X += scatter(alpha^T P_search) into X's ORIGINAL (never-reordered) row order.
                var T = RowsView(Tbuf, sLive);
                BlockCTV(in alpha, in Psearch, ref T);
                BlockScatterAddRows(ref X, in T, in Live, sLive, (fProxy)1);

                // S5: R_live -= alpha^T AP_search (R stays permuted -- plain BlockAdd applies).
                var T2 = RowsView(Tbuf, sLive);
                BlockCTV(in alpha, in APsearch, ref T2);
                var Rlive2 = RowsView(R, sLive);
                BlockAdd(ref Rlive2, in T2, (fProxy)(-1));

                // S6: lock newly converged rows, shrinking sLive.
                LockConvergedRows(ref R, ref Live, ref sLive, in thr);
                if (sLive == 0) { status = IterativeSolveStatus.Converged; iters = k + 1; goto cleanup; }

                // S7/S8: fresh (preconditioned) live residual, rank-revealing factorization.
                int saNew = FactorLiveResidual(in R, in M, ref Z, sLive, n, ref rowIn, ref rowOut, ref Lbuf, ref Pa);
                minActive = math.min(minActive, saNew);
                if (saNew == 0) { status = IterativeSolveStatus.Breakdown; iters = k + 1; goto cleanup; }
                var PaActive = RowsView(Pa, saNew);

                // S9/S10: beta from the A-conjugacy condition P_new _|_A P_search (PQ/APsearch are the
                // SAME ones from S2/S3 -- no extra matvec).
                var beta = RectView(betaBuf, saSearch, saNew);
                Blas.dot(in APsearch, in PaActive, ref beta, false, true);
                var work2 = View(workBuf, saSearch);
                if (!BlockSolveSPD(in PQ, ref beta, ref work2, saSearch))
                { status = IterativeSolveStatus.Breakdown; iters = k + 1; goto cleanup; }

                // S11: P_new = Pa_new - beta^T P_search. Read P_search (Tb) BEFORE overwriting P's
                // storage -- RowsView(P, saSearch) and RowsView(P, saNew) share the same backing buffer.
                var Tb = RowsView(Tbuf, saNew);
                BlockCTV(in beta, in Psearch, ref Tb);
                var Pnew = RowsView(P, saNew);
                CopyBlock(in PaActive, ref Pnew, saNew, n);
                BlockAdd(ref Pnew, in Tb, (fProxy)(-1));

                // S12/S13: A * P_new and its own Gram -- ready for the next iteration's S1-S3/S9-S10.
                var APnew = RowsView(AP, saNew);
                A.ApplyBlock(in Pnew, ref APnew, saNew);
                var PQnew = View(PQbuf, saNew);
                BlockGram(in Pnew, in APnew, ref PQnew, saNew);

                saSearch = saNew;
            }

        cleanup:
            // Recompute the residual fresh from the final X (does not try to unpermute the internal
            // working R) -- doubles as an exit-time sanity check.
            {
                var Rfinal = RowsView(AP, s);
                BlockResidual(in A, in X, in B, ref Rfinal, s, n);
                converged = CountConverged(in Rfinal, in thr, s, n, out maxr);
            }

            thr.Dispose(); Live.Dispose(); alphaBuf.Dispose(); betaBuf.Dispose(); PQbuf.Dispose();
            Lbuf.Dispose(); workBuf.Dispose(); Tbuf.Dispose();
            if (!M.IsIdentity) { rowIn.Dispose(); rowOut.Dispose(); }

            return new BlockSolveInfo { rhs = s, converged = converged, iterations = iters, maxRnorm = maxr, minActive = minActive, status = status };
        }

        // ---- bcgrq unpreconditioned + concrete forwarders ----------------------------------------------

        /// <summary>Unpreconditioned bcgrq -- forwards into the merged block
        /// <see cref="bcgrq{TOp, TPre}(in TOp, in TPre, in fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, ref fProxyMxN, int, fProxy)"/>
        /// with the identity preconditioner (needs no Z block).</summary>
        public static BlockSolveInfo bcgrq<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                        ref fProxyMxN R, ref fProxyMxN P, ref fProxyMxN AP, ref fProxyMxN Pa,
                                        int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyMxN Z = default;
            return bcgrq(in A, default(fProxyIdentityPreconditioner), in B, ref X, ref R, ref P, ref AP, ref Pa, ref Z, maxIter, tol);
        }

        /// <summary>bcgrq over a dense SPD <see cref="fProxyMxN"/> A (n x n) with an s x n block B.
        /// Allocates block scratch from Allocator.Temp.</summary>
        public static BlockSolveInfo bcgrq(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true),
                      AP = B.fProxyTempMat(s, n, true), Pa = B.fProxyTempMat(s, n, true);
            return bcgrq(new fProxyDenseOperator(in A), in B, ref X, ref R, ref P, ref AP, ref Pa, maxIter, tol);
        }

        /// <summary>bcgrq over a dense SPD A with default maxIter (A.M_Rows) and tol (sqrtEps).</summary>
        public static BlockSolveInfo bcgrq(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => bcgrq(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Preconditioned bcgrq over a dense SPD A. Allocates block scratch (incl. Z).</summary>
        public static BlockSolveInfo bcgrq<TPre>(in fProxyMxN A, in TPre M, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true),
                      AP = B.fProxyTempMat(s, n, true), Pa = B.fProxyTempMat(s, n, true), Z = B.fProxyTempMat(s, n, true);
            return bcgrq(new fProxyDenseOperator(in A), in M, in B, ref X, ref R, ref P, ref AP, ref Pa, ref Z, maxIter, tol);
        }

        /// <summary>bcgrq over a block-sparse (BSR) SPD A with an s x n block B. Allocates block
        /// scratch from Allocator.Temp.</summary>
        public static BlockSolveInfo bcgrq(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true),
                      AP = B.fProxyTempMat(s, n, true), Pa = B.fProxyTempMat(s, n, true);
            return bcgrq(new fProxyBSROperator(in A), in B, ref X, ref R, ref P, ref AP, ref Pa, maxIter, tol);
        }

        /// <summary>bcgrq over a BSR SPD A with default maxIter (A.M_Rows) and tol (sqrtEps).</summary>
        public static BlockSolveInfo bcgrq(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
            => bcgrq(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Preconditioned bcgrq over a BSR SPD A. Allocates block scratch (incl. Z).</summary>
        public static BlockSolveInfo bcgrq<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN R = B.fProxyTempMat(s, n, true), P = B.fProxyTempMat(s, n, true),
                      AP = B.fProxyTempMat(s, n, true), Pa = B.fProxyTempMat(s, n, true), Z = B.fProxyTempMat(s, n, true);
            return bcgrq(new fProxyBSROperator(in A), in M, in B, ref X, ref R, ref P, ref AP, ref Pa, ref Z, maxIter, tol);
        }
    }
}
