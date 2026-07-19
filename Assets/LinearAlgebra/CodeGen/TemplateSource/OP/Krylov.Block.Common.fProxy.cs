using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ================= Block (multi-RHS) Krylov solvers =================
        // Block vectors are fProxyMxN with s ROWS x n COLS: row j is RHS/solution vector j (length
        // n = A.Rows). This matches IfProxyLinearOperator.ApplyBlock, which applies A to all s rows in
        // one streaming pass. The s x s block coefficients are formed and solved in the row space.

        // ---- block helpers shared by bcg, bcgrq, and bfbcg (Krylov.Block.CG/BCGrQ/BFBCG.fProxy.cs) ----

        // G = V * W^T  (s x s Gram of the block rows) via the optimized GEMM, then symmetrized
        // exactly (the Grams here -- P^T A P, R^T M^-1 R -- are symmetric by construction; forcing it
        // shields the s x s Cholesky from GEMM round-off asymmetry).
        static void BlockGram(in fProxyMxN V, in fProxyMxN W, ref fProxyMxN G, int s)
        {
            Blas.dot(in V, in W, ref G, false, true);       // G = V W^T
            for (int i = 0; i < s; i++)
                for (int j = i + 1; j < s; j++)
                {
                    fProxy avg = (fProxy)0.5 * (G[i, j] + G[j, i]);
                    G[i, j] = avg;
                    G[j, i] = avg;
                }
        }

        // dst = C^T * V  (s x n block) via the optimized GEMM. dst must be distinct from C and V.
        static void BlockCTV(in fProxyMxN C, in fProxyMxN V, ref fProxyMxN dst)
            => Blas.dot(in C, in V, ref dst, true, false);   // dst = C^T V

        // Y += sign * T over the whole (contiguous) s x n block.
        static unsafe void BlockAdd(ref fProxyMxN Y, in fProxyMxN T, fProxy sign)
        {
            fProxy* yp = Y.Data.Ptr; fProxy* tp = T.Data.Ptr;
            long len = (long)Y.M_Rows * Y.N_Cols;
            for (long i = 0; i < len; i++) yp[i] += sign * tp[i];
        }

        // dst = Z + T over the whole (contiguous) s x n block. dst may be Z? no -- dst distinct.
        static unsafe void BlockZplusT(in fProxyMxN Z, in fProxyMxN T, ref fProxyMxN dst)
        {
            fProxy* zp = Z.Data.Ptr; fProxy* tp = T.Data.Ptr; fProxy* dp = dst.Data.Ptr;
            long len = (long)dst.M_Rows * dst.N_Cols;
            for (long i = 0; i < len; i++) dp[i] = zp[i] + tp[i];
        }

        // Solve the s x s SPD system G * Xsol = RHS_to_X (each column an independent RHS), writing the
        // solution into RHS_to_X. Robust to a rank-deficient G (dependent RHS columns): retries an
        // escalating diagonal ridge scaled to G's own diagonal, mirroring LOBPCG's FactorGram. `work`
        // is s x s scratch (overwritten). Returns false only if even the largest ridge fails.
        static bool BlockSolveSPD(in fProxyMxN G, ref fProxyMxN RHS_to_X, ref fProxyMxN work, int s)
        {
            fProxy diagMax = (fProxy)0;
            for (int i = 0; i < s; i++) { fProxy d = G[i, i]; if (d > diagMax) diagMax = d; }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;

            fProxy ridge = (fProxy)0;
            bool ok = false;
            for (int attempt = 0; attempt < 6; attempt++)
            {
                for (int i = 0; i < s; i++)
                    for (int j = 0; j < s; j++)
                        work[i, j] = G[i, j] + (i == j ? ridge : (fProxy)0);

                var info = CHO.decompInPlace(ref work);   // work -> L
                if (info.status == DirectSolveStatus.Success) { ok = true; break; }
                ridge = ridge == (fProxy)0 ? (fProxy)16 * Consts.fProxyEpsilon * diagMax : ridge * (fProxy)16;
            }
            if (!ok) return false;

            CHO.decompSolve(ref work, ref RHS_to_X);       // in-place forward/back substitution
            return true;
        }

        // Counts columns with ||R[j]||^2 <= thr[j]; also returns the worst ||R[j]||.
        static int CountConverged(in fProxyMxN R, in fProxyN thr, int s, int n, out double maxRnorm)
        {
            int conv = 0; double worst = 0;
            for (int j = 0; j < s; j++)
            {
                fProxy rr = (fProxy)0;
                for (int c = 0; c < n; c++) rr += R[j, c] * R[j, c];
                if (rr <= thr[j]) conv++;
                double rn = math.sqrt((double)rr);
                if (rn > worst) worst = rn;
            }
            maxRnorm = worst;
            return conv;
        }

        static void BlockApplyPre<TPre>(in TPre M, in fProxyMxN R, ref fProxyMxN Z, int s, int n,
                                        ref fProxyN rowIn, ref fProxyN rowOut)
            where TPre : struct, IfProxyPreconditioner
        {
            for (int i = 0; i < s; i++)
            {
                for (int c = 0; c < n; c++) rowIn[c] = R[i, c];
                M.Apply(in rowIn, ref rowOut);
                for (int c = 0; c < n; c++) Z[i, c] = rowOut[c];
            }
        }

        static void CopyBlock(in fProxyMxN src, ref fProxyMxN dst, int s, int n)
        {
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) dst[i, c] = src[i, c];
        }

        static void CopyMat(in fProxyMxN src, ref fProxyMxN dst, int s)
        {
            for (int i = 0; i < s; i++)
                for (int j = 0; j < s; j++) dst[i, j] = src[i, j];
        }

        // Same-buffer, smaller SQUARE logical view (rows == cols == m): a value-copy of the fProxyMxN
        // struct with M_Rows/N_Cols overwritten to the leading m*m elements of buf's storage -- not a
        // new allocation, not a strided sub-block of a larger stride (mirrors LOBPCG.fProxy.cs's View).
        static fProxyMxN View(in fProxyMxN buf, int m)
        {
            var v = buf;
            v.M_Rows = m;
            v.N_Cols = m;
            return v;
        }

        // Same-buffer logical view with `rows` rows and buf's own column count: the leading `rows` rows
        // of a row-major buffer are exactly a standalone rows x N_Cols matrix, so this is a contiguous-
        // prefix reinterpretation, not a new allocation (mirrors LOBPCG.fProxy.cs's RowsView).
        static fProxyMxN RowsView(in fProxyMxN buf, int rows)
        {
            var v = buf;
            v.M_Rows = rows;
            return v;
        }

        // Same-buffer logical view with an independent row/col count (the rectangular generalization of
        // View/RowsView), valid whenever rows*cols does not exceed buf's element capacity.
        static fProxyMxN RectView(in fProxyMxN buf, int rows, int cols)
        {
            var v = buf;
            v.M_Rows = rows;
            v.N_Cols = cols;
            return v;
        }

        // Yfull[Live[i], c] += sign * Tlive[i, c] for i in [0, sLive), all columns of Yfull -- scatters
        // an sLive-wide live update back into Yfull's ORIGINAL (never-reordered) row order.
        static void BlockScatterAddRows(ref fProxyMxN Yfull, in fProxyMxN Tlive, in Pivot Live, int sLive, fProxy sign)
        {
            int n = Yfull.N_Cols;
            for (int i = 0; i < sLive; i++)
            {
                int orig = Live[i];
                for (int c = 0; c < n; c++)
                    Yfull[orig, c] += sign * Tlive[i, c];
            }
        }

        // Backward lock scan (mirrors LOBPCG's numActive lock loop): swaps any row i whose current
        // squared residual is within thr[Live[i]] to the current last live slot and shrinks sLive. A
        // locked row's R value is frozen (correct -- it already satisfied thr) and its original index
        // drops out of every later BlockScatterAddRows (X update) for the rest of the solve.
        static void LockConvergedRows(ref fProxyMxN R, ref Pivot Live, ref int sLive, in fProxyN thr)
        {
            int n = R.N_Cols;
            for (int i = sLive - 1; i >= 0; i--)
            {
                int orig = Live[i];
                fProxy rr = (fProxy)0;
                for (int c = 0; c < n; c++) rr += R[i, c] * R[i, c];
                if (rr <= thr[orig])
                {
                    int last = sLive - 1;
                    if (i != last) { Swap.Rows(ref R, i, last); Live.Swap(i, last); }
                    sLive--;
                }
            }
        }

        // Numerical rank off L's non-increasing |diagonal|, LQRP's own convention: tol = relTol *
        // |L[0,0]|, relTol = max(m, nGlobal) * Consts.fProxyZeroThreshold (matches LQRP.solveInPlace's
        // default relTol / SVD.pinvSolve / Analysis.rank).
        static int LQRPRank(in fProxyMxN L, int m, int nGlobal)
        {
            fProxy relTol = (fProxy)math.max(m, nGlobal) * Consts.fProxyZeroThreshold;
            fProxy tol = relTol * math.abs(L[0, 0]);
            int rank = 0;
            for (int i = 0; i < m; i++)
            {
                if (math.abs(L[i, i]) > tol) rank++;
                else break;
            }
            return rank;
        }

        // Factors the live (preconditioned) residual block via LQRP, writing the fresh orthonormal
        // search basis into Pa's leading sLive rows and returning its numerical rank. Feeds the live
        // RowsView straight into LQRP.decomp -- decomp's own internal scratch copy is now length-checked
        // against its logical M_Rows*N_Cols (fProxyMxN.CopyFrom), so it no longer needs pre-copying into
        // an exactly-sized buffer first (see DEVLOG.md).
        static int FactorLiveResidual<TPre>(in fProxyMxN R, in TPre M, ref fProxyMxN Z, int sLive, int n,
                                            ref fProxyN rowIn, ref fProxyN rowOut, ref fProxyMxN Lbuf, ref fProxyMxN Pa)
            where TPre : struct, IfProxyPreconditioner
        {
            var Ppiv  = new Pivot(sLive, Allocator.Temp);
            var Lv    = View(Lbuf, sLive);
            var Qfull = RowsView(Pa, sLive);

            if (M.IsIdentity)
            {
                var Rlive = RowsView(R, sLive);
                LQRP.decomp(in Rlive, ref Lv, ref Qfull, ref Ppiv);
            }
            else
            {
                var Rlive = RowsView(R, sLive);
                var Zpre = RowsView(Z, sLive);
                BlockApplyPre(in M, in Rlive, ref Zpre, sLive, n, ref rowIn, ref rowOut);
                LQRP.decomp(in Zpre, ref Lv, ref Qfull, ref Ppiv);
            }

            Ppiv.Dispose();

            return LQRPRank(in Lv, sLive, n);
        }

        // Factors the live search block via LQRP, writing the fresh orthonormal search basis into Pa's
        // leading sLive rows and returning its numerical rank. Feeds the live RowsView straight into
        // LQRP.decomp -- see FactorLiveResidual above.
        static int FactorLiveSearch(in fProxyMxN P, int sLive, int n, ref fProxyMxN Lbuf, ref fProxyMxN Pa)
        {
            var Plive = RowsView(P, sLive);

            var Ppiv  = new Pivot(sLive, Allocator.Temp);
            var Lv    = View(Lbuf, sLive);
            var Qfull = RowsView(Pa, sLive);
            LQRP.decomp(in Plive, ref Lv, ref Qfull, ref Ppiv);
            Ppiv.Dispose();

            return LQRPRank(in Lv, sLive, n);
        }

        // Factors the r x r Gram G (SPD by construction: G = Phat^T A Phat with Phat's rows orthonormal)
        // into work via Cholesky ONCE, so the caller can reuse the same factor for both the alpha and
        // beta solves. Retries an escalating diagonal ridge scaled to G's own diagonal as a numerical-
        // noise safety net (mirrors BlockSolveSPD's ladder). Returns false only if even the largest
        // ridge fails.
        static bool FactorGramOnce(in fProxyMxN G, ref fProxyMxN work, int r)
        {
            fProxy diagMax = (fProxy)0;
            for (int i = 0; i < r; i++) { fProxy d = G[i, i]; if (d > diagMax) diagMax = d; }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;

            fProxy ridge = (fProxy)0;
            for (int attempt = 0; attempt < 6; attempt++)
            {
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < r; j++)
                        work[i, j] = G[i, j] + (i == j ? ridge : (fProxy)0);

                var info = CHO.decompInPlace(ref work);
                if (info.status == DirectSolveStatus.Success) return true;
                ridge = ridge == (fProxy)0 ? (fProxy)16 * Consts.fProxyEpsilon * diagMax : ridge * (fProxy)16;
            }
            return false;
        }
    }
}
