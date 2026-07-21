using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ---- bgcrodr private helpers -------------------------------------------------------------

        // dot(A[rowA,:], B[rowB,:]) -- two possibly-distinct rows from two possibly-distinct blocks.
        static fProxy RowDotAt(in fProxyMxN A, int rowA, in fProxyMxN B, int rowB, int n)
        {
            fProxy acc = (fProxy)0;
            for (int c = 0; c < n; c++) acc += A[rowA, c] * B[rowB, c];
            return acc;
        }

        // dst[c] += scale * src[row,c].
        static void AddScaledRowInto(ref fProxyN dst, fProxy scale, in fProxyMxN src, int row, int n)
        {
            for (int c = 0; c < n; c++) dst[c] += scale * src[row, c];
        }

        // dst[dstRow,c] += scale * src[srcRow,c].
        static void AddScaledRowToRow(ref fProxyMxN dst, int dstRow, fProxy scale, in fProxyMxN src, int srcRow, int n)
        {
            for (int c = 0; c < n; c++) dst[dstRow, c] += scale * src[srcRow, c];
        }

        // Resolves combined-space index ai (0 <= ai < kcurAtEntry + off[k]) to its source row: the
        // first kcurAtEntry indices are the OLD recycled columns (Ublk/AUblk row ai); the rest are
        // this-cycle block-Arnoldi columns, located among the k steps via off[0..k] (bgmres's own
        // prefix-summed active-width array) into V[j]/Zv[j] (the basis actually combined into x) and
        // AVlist[j] (the raw pre-recycled-projection A-apply stored there).
        static void ResolveCombinedCol(int ai, int kcurAtEntry, bool flexible,
            in fProxyMxN Ublk, in fProxyMxN AUblk,
            in UnsafeList<fProxyMxN> Vlist, in UnsafeList<fProxyMxN> Zlist, in UnsafeList<fProxyMxN> AVlist,
            in Indices off, int k,
            out fProxyMxN Pmat, out int Prow, out fProxyMxN APmat, out int AProw)
        {
            if (ai < kcurAtEntry)
            {
                Pmat = Ublk; Prow = ai; APmat = AUblk; AProw = ai;
                return;
            }
            int gi = ai - kcurAtEntry;
            int j = 0;
            while (j < k - 1 && gi >= off[j + 1]) j++;
            int row = gi - off[j];
            Pmat = flexible ? Zlist[j] : Vlist[j];
            APmat = AVlist[j];
            Prow = row; AProw = row;
        }

        // Solves the active kcur x kcur upper-triangular Ru * Zout = Rhs for s independent RHS
        // columns. A collapsed diagonal (|Ru[i,i]| <= pivotGuard) writes 0 for that row instead of
        // dividing -- the caller treats this as "no recycled correction for this row", never NaN.
        static void BackSubUpperBlock(in fProxyMxN Ru, int kcur, in fProxyMxN Rhs, ref fProxyMxN Zout, int s, fProxy pivotGuard)
        {
            for (int i = kcur - 1; i >= 0; i--)
            {
                fProxy diag = Ru[i, i];
                bool ok = math.abs(diag) > pivotGuard;
                for (int c = 0; c < s; c++)
                {
                    fProxy sum = Rhs[i, c];
                    for (int l = i + 1; l < kcur; l++) sum -= Ru[i, l] * Zout[l, c];
                    Zout[i, c] = ok ? sum / diag : (fProxy)0;
                }
            }
        }

        // Max |diagonal| of the active kcur x kcur upper-triangular Ru -- scales
        // BackSubUpperBlock's singularity guard to Ru's own (||A||-scaled) magnitude rather than an
        // unrelated norm (e.g. ||B||).
        static fProxy MaxAbsRuDiag(in fProxyMxN Ru, int kcur)
        {
            fProxy m = (fProxy)0;
            for (int i = 0; i < kcur; i++) m = math.max(m, math.abs(Ru[i, i]));
            return m;
        }

        // ---- block GCRO-DR core (bgcrodr) ----------------------------------------------------------

        /// <summary>
        /// Block GCRO-DR (Parks-de Sturler-Mackey-Johnson-Maiti 2006, block-generalized): restarted
        /// block GMRES(m) (<see cref="bgmres{TOp,TPre}"/>'s block Arnoldi / deflating-LQ-rank / periodic
        /// dense-QR least-squares machinery) that additionally RECYCLES a k-dimensional approximate
        /// invariant subspace (refined harmonic-Ritz vectors) across restart cycles, for a general
        /// (nonsymmetric) square A and s simultaneous right-hand sides. Generic over both the operator
        /// (<see cref="IfProxyLinearOperator"/>) and the preconditioner (<see cref="IfProxyPreconditioner"/>).
        ///
        /// Each cycle: project the recycled subspace (recycle individual n-vectors U, image C = A·U
        /// orthonormal, A·U = C·Ru) out of the block residual and fold the correction into X (fresh
        /// matvec before and after, never trusting an incremental update through a possibly
        /// near-singular Ru solve); run an m-block-step Arnoldi (projected against C so every this-
        /// cycle basis vector stays recycle-orthogonal) with bgmres's own basis/Hessenberg/least-
        /// squares machinery; commit X; then rebuild the recycled subspace from a small dense
        /// harmonic-Ritz eigenproblem over the combined (old-recycle + this-cycle-Krylov) space. The
        /// harmonic Ritz VALUES come from <see cref="Eigen.valuesQRInPlace"/> (general nonsymmetric
        /// eigenvalues of Gmat⁻¹Fmat, Gmat solved via <see cref="LU.solveInPlace"/>); each selected
        /// value's REFINED vector (minimizing ‖(A-θI)v‖ over the combined subspace) comes from
        /// <see cref="Eigen.symmetricInPlace"/> on a small symmetric matrix -- this library has no
        /// general nonsymmetric eigenVECTOR solver, so the refined-vector route reuses the two
        /// eigensolvers that exist (mirrors scalar <see cref="gcrodr{TOp,TPre}"/> exactly).
        ///
        /// B and X are s ROWS x n COLS. X is warm-startable. Convergence is per column against
        /// tol²·‖B[j]‖², via the least-squares Pythagorean residual identity (mirrors bgmres). recycle
        /// counts INDIVIDUAL recycled vectors and must be in [0, restart·s); recycle = 0 disables
        /// recycling (a plain block-Arnoldi restart engine, not required to bit-match bgmres). maxIter
        /// counts TOTAL block steps across restarts (mirrors bgmres). Owns its whole workspace via
        /// Allocator.Temp. Status: Converged / MaxIterations / Breakdown (R0 != 0 yet the initial
        /// deflating-LQ rank is 0) -- never NaN, never a false Converged. A failed or ill-conditioned
        /// per-cycle deflation update (LU/QR eigensolve failure, rank-deficient AU) degrades gracefully
        /// (keeps the previous cycle's recycled subspace) rather than aborting the solve.
        /// </summary>
        public static BlockSolveInfo bgcrodr<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int recycle, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("bgcrodr: A must be square");
            int n = A.Rows;
            int s = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("bgcrodr: B.N_Cols must equal A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("bgcrodr: X must match B");
            if (restart < 1) throw new ArgumentException("bgcrodr: restart must be >= 1");
            if (recycle < 0) throw new ArgumentException("bgcrodr: recycle must be >= 0");
            if (recycle >= restart * s) throw new ArgumentException("bgcrodr: recycle must be < restart * B.M_Rows");
            if (maxIter < 1) throw new ArgumentException("bgcrodr: maxIter must be >= 1");
            if (s > n) throw new ArgumentException("bgcrodr: B.M_Rows (s) must be <= A.Rows");

            int m = restart;
            bool flexible = !M.IsIdentity;
            bool recycling = recycle > 0;

            var V = new UnsafeList<fProxyMxN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyMxN(s, n, Allocator.Temp, true));

            var Wbuf   = new fProxyMxN(s, n, Allocator.Temp, false);
            var Tbuf   = new fProxyMxN(s, n, Allocator.Temp, true);
            var R0     = new fProxyMxN(s, n, Allocator.Temp, true);
            var Wcombo = new fProxyMxN(s, n, Allocator.Temp, true);
            var CorrBuf = new fProxyMxN(s, n, Allocator.Temp, true);

            UnsafeList<fProxyMxN> Zv = default;
            fProxyN rowIn = default, rowOut = default;
            if (flexible)
            {
                Zv = new UnsafeList<fProxyMxN>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) Zv.Add(new fProxyMxN(s, n, Allocator.Temp, true));
                rowIn  = new fProxyN(n);
                rowOut = new fProxyN(n);
            }

            var Hbuf = new fProxyMxN((m + 1) * s, m * s, Allocator.Temp, true);
            var Gbuf = new fProxyMxN((m + 1) * s, s, Allocator.Temp, true);

            var HQscratch  = new fProxyMxN((m + 1) * s, m * s, Allocator.Temp, false);
            var Rscratch   = new fProxyMxN(m * s, m * s, Allocator.Temp, true);
            var Yscratch   = new fProxyMxN(m * s, s, Allocator.Temp, true);
            var QtGscratch = new fProxyMxN(m * s, s, Allocator.Temp, true);

            var Lbuf   = new fProxyMxN(s, s, Allocator.Temp, true);
            var HijBuf = new fProxyMxN(s, s, Allocator.Temp, true);
            var YiBuf  = new fProxyMxN(s, s, Allocator.Temp, true);

            var thr = new fProxyN(s);
            var w   = new Indices(m + 1, Allocator.Temp);
            var off = new Indices(m + 2, Allocator.Temp);

            // ---- recycled-subspace state, fixed max size `recycle` individual n-vectors, active
            // prefix `kcur` -- U/C/Ru mirror scalar gcrodr's, stored as block matrices (recycle rows)
            // instead of an UnsafeList<fProxyN> so the projection/uncorrection steps below can use the
            // same GEMM-shaped block helpers bgmres's own machinery uses. ----
            UnsafeList<fProxyMxN> AVlist = default;
            fProxyMxN Ublk = default, Cblk = default, Ru = default, Bmat = default;
            fProxyMxN BijBuf = default, CtrBuf = default, ZprojBuf = default;
            int kcur = 0;
            if (recycling)
            {
                AVlist = new UnsafeList<fProxyMxN>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) AVlist.Add(new fProxyMxN(s, n, Allocator.Temp, true));

                Ublk = new fProxyMxN(recycle, n, Allocator.Temp, true);
                Cblk = new fProxyMxN(recycle, n, Allocator.Temp, true);
                Ru   = new fProxyMxN(recycle, recycle, Allocator.Temp, true);
                Bmat = new fProxyMxN(recycle, m * s, Allocator.Temp, true);
                BijBuf   = new fProxyMxN(recycle, s, Allocator.Temp, true);
                CtrBuf   = new fProxyMxN(recycle, s, Allocator.Temp, true);
                ZprojBuf = new fProxyMxN(recycle, s, Allocator.Temp, true);
            }

            // Per-column thresholds tol^2 ||B[j]||^2, floored for zero/tiny-norm columns.
            BuildColumnThresholds(in B, ref thr, s, n, tol);

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int total = 0;
            int converged = 0;
            double maxr = 0;
            int minActive = s;

            while (total < maxIter)
            {
                A.ApplyBlock(in X, ref R0, s);
                for (int i = 0; i < s; i++)
                    for (int c = 0; c < n; c++) R0[i, c] = B[i, c] - R0[i, c];

                if (kcur > 0)
                {
                    var Cactive = RowsView(Cblk, kcur);
                    var CtrView = RowsView(CtrBuf, kcur);
                    BlockCrossGram(in Cactive, in R0, ref CtrView);
                    var ZprojView = RowsView(ZprojBuf, kcur);
                    fProxy ruGuard = Consts.fProxyEpsilon * (fProxy)100 * MaxAbsRuDiag(in Ru, kcur);
                    BackSubUpperBlock(in Ru, kcur, in CtrView, ref ZprojView, s, ruGuard);
                    var Uactive = RowsView(Ublk, kcur);
                    var corr = RowsView(CorrBuf, s);
                    BlockCTV(in ZprojView, in Uactive, ref corr);
                    BlockAdd(ref X, in corr, (fProxy)1);

                    A.ApplyBlock(in X, ref R0, s);
                    for (int i = 0; i < s; i++)
                        for (int c = 0; c < n; c++) R0[i, c] = B[i, c] - R0[i, c];
                }

                converged = CountConverged(in R0, in thr, s, n, out maxr);
                if (converged == s) { status = IterativeSolveStatus.Converged; break; }

                ZeroPrefix(ref Hbuf, (m + 1) * s, m * s);
                ZeroPrefix(ref Gbuf, (m + 1) * s, s);
                if (recycling) ZeroPrefix(ref Bmat, recycle, m * s);

                var Ppiv0 = new Pivot(s, Allocator.Temp);
                var L0 = View(Lbuf, s);
                var Q0 = V[0];
                LQRP.decomp(in R0, ref L0, ref Q0, ref Ppiv0);
                Ppiv0.Dispose();

                w[0] = LQRPRank(in L0, s, n);
                minActive = math.min(minActive, w[0]);
                if (w[0] == 0) { status = IterativeSolveStatus.Breakdown; break; }

                off[0] = 0; off[1] = w[0];
                var V0 = RowsView(Q0, w[0]);
                var G0 = RowsView(Gbuf, w[0]);
                BlockCrossGram(in V0, in R0, ref G0);

                int k = 0;
                bool cycleConverged = false;
                bool lsBreakdown = false;
                for (int j = 0; j < m && total < maxIter; j++)
                {
                    var Vj = RowsView(V[j], w[j]);
                    var Wj = RowsView(Wbuf, w[j]);
                    if (flexible)
                    {
                        var Ztj = RowsView(Zv[j], w[j]);
                        BlockApplyPre(in M, in Vj, ref Ztj, w[j], n, ref rowIn, ref rowOut);
                        A.ApplyBlock(in Ztj, ref Wj, w[j]);
                    }
                    else
                    {
                        A.ApplyBlock(in Vj, ref Wj, w[j]);
                    }

                    if (recycling)
                    {
                        var AVj = RowsView(AVlist[j], w[j]);
                        CopyBlock(in Wj, ref AVj, w[j], n);
                    }

                    if (kcur > 0)
                    {
                        var Cactive = RowsView(Cblk, kcur);
                        var Bij = RectView(BijBuf, kcur, w[j]);
                        BlockCrossGram(in Cactive, in Wj, ref Bij);
                        StoreBlockAt(ref Bmat, 0, off[j], in Bij, kcur, w[j]);
                        var Tij0 = RowsView(Tbuf, w[j]);
                        BlockCTV(in Bij, in Cactive, ref Tij0);
                        BlockAdd(ref Wj, in Tij0, (fProxy)(-1));
                    }

                    // Pre-orthogonalization magnitude of this step, captured before MGS2 below mutates
                    // Wj (after the recycled-subspace projection above, mirrors
                    // Krylov.BlockArnoldiMGS2Step's own scale capture) -- the absolute floor
                    // LQRPRankFloored applies to the post-orthogonalization LQ diagonals.
                    fProxy scale = Norms.L2(in Wj);

                    // Modified block Gram-Schmidt against V[0..j], ONE unconditional reorthogonalization
                    // pass (MGS2, mirrors bgmres exactly).
                    for (int pass = 0; pass < 2; pass++)
                    {
                        for (int i = 0; i <= j; i++)
                        {
                            var Vi  = RowsView(V[i], w[i]);
                            var Hij = RectView(HijBuf, w[i], w[j]);
                            BlockCrossGram(in Vi, in Wj, ref Hij);
                            StoreBlockAt(ref Hbuf, off[i], off[j], in Hij, w[i], w[j]);
                            var Tij = RowsView(Tbuf, w[j]);
                            BlockCTV(in Hij, in Vi, ref Tij);
                            BlockAdd(ref Wj, in Tij, (fProxy)(-1));
                        }
                    }

                    var Ppiv2 = new Pivot(w[j], Allocator.Temp);
                    var Lv = View(Lbuf, w[j]);
                    var Qout = RowsView(V[j + 1], w[j]);
                    LQRP.decomp(in Wj, ref Lv, ref Qout, ref Ppiv2);
                    Ppiv2.Dispose();

                    int wj1 = LQRPRankFloored(in Lv, w[j], n, scale);
                    w[j + 1] = wj1;
                    minActive = math.min(minActive, wj1);
                    off[j + 2] = off[j + 1] + wj1;

                    if (wj1 > 0)
                    {
                        var Vj1  = RowsView(V[j + 1], wj1);
                        var Hj1j = RectView(HijBuf, wj1, w[j]);
                        BlockCrossGram(in Vj1, in Wj, ref Hj1j);
                        StoreBlockAt(ref Hbuf, off[j + 1], off[j], in Hj1j, wj1, w[j]);
                    }

                    total++;
                    k = j + 1;

                    int totalRows = off[k + 1];
                    int totalCols = off[k];

                    var HQ = RectView(HQscratch, totalRows, totalCols);
                    CopyBlock(in Hbuf, ref HQ, totalRows, totalCols);
                    var Rls = RectView(Rscratch, totalCols, totalCols);
                    QR.decompInPlace(ref HQ, ref Rls);
                    var Gactive = RowsView(Gbuf, totalRows);
                    var Yv = RowsView(Yscratch, totalCols);
                    QR.decompSolve(ref HQ, ref Rls, ref Gactive, ref Yv);

                    // A rank-deficient least-squares system (e.g. A maps this whole cycle's basis to
                    // 0, as a singular/zero operator can) can leave Yv non-finite -- an honest
                    // Breakdown, never a NaN committed into X.
                    for (int r = 0; r < totalCols; r++)
                        for (int c = 0; c < s; c++)
                            if (math.isnan(Yv[r, c]) || math.isinf(Yv[r, c])) lsBreakdown = true;
                    if (lsBreakdown) break;

                    var QtG = RowsView(QtGscratch, totalCols);
                    Blas.dot(in HQ, in Gactive, ref QtG, true, false);

                    bool allConverged = true;
                    for (int c = 0; c < s; c++)
                    {
                        fProxy gg = (fProxy)0;
                        for (int r = 0; r < totalRows; r++) gg += Gactive[r, c] * Gactive[r, c];
                        fProxy qq = (fProxy)0;
                        for (int r = 0; r < totalCols; r++) qq += QtG[r, c] * QtG[r, c];
                        fProxy resid2 = math.max((fProxy)0, gg - qq);
                        if (resid2 > thr[c]) allConverged = false;
                    }
                    cycleConverged = allConverged;

                    if (cycleConverged || wj1 == 0) break;
                }

                if (lsBreakdown)
                {
                    // X was never updated this cycle -- the shared post-loop recompute below reports
                    // the TRUE fresh residual at the returned X, never this cycle's poisoned solve.
                    status = IterativeSolveStatus.Breakdown;
                    break;
                }

                int totalColsFinal = off[k];

                // Commit: X += combine(Y, this-cycle basis [identity: V, flexible: the stored
                // M-applied Zv]) -- mirrors scalar gcrodr's per-vector combine (not bgmres's single
                // apply-M-to-the-combination trick), so U/C below can be built directly from the same
                // basis without a second M-apply.
                {
                    var Yfinal = RowsView(Yscratch, totalColsFinal);
                    ZeroPrefix(ref Wcombo, s, n);
                    for (int i = 0; i < k; i++)
                    {
                        var Yi = RectView(YiBuf, w[i], s);
                        ExtractRowsAt(in Yfinal, off[i], w[i], ref Yi);
                        var Bi = flexible ? RowsView(Zv[i], w[i]) : RowsView(V[i], w[i]);
                        var Ti = RowsView(Tbuf, s);
                        BlockCTV(in Yi, in Bi, ref Ti);
                        BlockAdd(ref Wcombo, in Ti, (fProxy)1);
                    }
                    BlockAdd(ref X, in Wcombo, (fProxy)1);

                    if (kcur > 0 && totalColsFinal > 0)
                    {
                        // Bview/Yscratch at their TRUE (unreshaped) widths: Bmat's columns beyond
                        // totalColsFinal are exactly zero (freshly cleared this cycle), so Yscratch's
                        // possibly-stale rows beyond totalColsFinal (leftover from an earlier, wider
                        // cycle) contribute 0*stale = 0 -- avoids narrowing Bmat's column count into a
                        // non-full-width RectView, which would misalign its flat row-major storage.
                        var Bview = RowsView(Bmat, kcur);
                        var CtrView2 = RowsView(CtrBuf, kcur);
                        Blas.dot(in Bview, in Yscratch, ref CtrView2, false, false);
                        var ZprojView2 = RowsView(ZprojBuf, kcur);
                        fProxy ruGuard2 = Consts.fProxyEpsilon * (fProxy)100 * MaxAbsRuDiag(in Ru, kcur);
                        BackSubUpperBlock(in Ru, kcur, in CtrView2, ref ZprojView2, s, ruGuard2);
                        var Uactive2 = RowsView(Ublk, kcur);
                        var corr2 = RowsView(CorrBuf, s);
                        BlockCTV(in ZprojView2, in Uactive2, ref corr2);
                        BlockAdd(ref X, in corr2, (fProxy)(-1));
                    }
                }

                if (cycleConverged) { status = IterativeSolveStatus.Converged; break; }

                // ---- deflation update: rebuild the recycled subspace from the combined
                // (old-recycle + this-cycle block-Krylov) space via a small dense harmonic-Ritz
                // eigenproblem -- block-generalized scalar gcrodr (see that file's own doc for the
                // derivation). Skipped (old U/C/Ru kept as-is) on a degenerate zero-column cycle or
                // whenever a numerical guard below trips -- recycling is an accelerator, not a
                // correctness requirement, so any failure here degrades gracefully.
                if (recycling && totalColsFinal > 0)
                {
                    int d = kcur + totalColsFinal;
                    int kcurAtEntry = kcur;
                    bool hadOldU = kcurAtEntry > 0;

                    fProxyMxN AUblk = default;
                    if (hadOldU)
                    {
                        AUblk = new fProxyMxN(kcurAtEntry, n, Allocator.Temp, false);   // cleared: AddScaledRowToRow ACCUMULATES (+=) into it
                        for (int i = 0; i < kcurAtEntry; i++)
                            for (int l = 0; l < kcurAtEntry; l++)
                                AddScaledRowToRow(ref AUblk, i, Ru[l, i], in Cblk, l, n);
                    }

                    var Fmat = new fProxyMxN(d, d, Allocator.Temp, false);
                    var Gmat = new fProxyMxN(d, d, Allocator.Temp, false);
                    var Pgram = new fProxyMxN(d, d, Allocator.Temp, false);

                    for (int ai = 0; ai < d; ai++)
                    {
                        ResolveCombinedCol(ai, kcurAtEntry, flexible, in Ublk, in AUblk, in V, in Zv, in AVlist, in off, k,
                                            out fProxyMxN Pa, out int Prow, out fProxyMxN APa, out int AProw);

                        for (int bi = ai; bi < d; bi++)
                        {
                            ResolveCombinedCol(bi, kcurAtEntry, flexible, in Ublk, in AUblk, in V, in Zv, in AVlist, in off, k,
                                                out fProxyMxN Pb, out int Pbrow, out fProxyMxN APb, out int APbrow);

                            fProxy fval = RowDotAt(in APa, AProw, in APb, APbrow, n);
                            Fmat[ai, bi] = fval; Fmat[bi, ai] = fval;

                            fProxy pval = RowDotAt(in Pa, Prow, in Pb, Pbrow, n);
                            Pgram[ai, bi] = pval; Pgram[bi, ai] = pval;
                        }

                        for (int bi = 0; bi < d; bi++)
                        {
                            ResolveCombinedCol(bi, kcurAtEntry, flexible, in Ublk, in AUblk, in V, in Zv, in AVlist, in off, k,
                                                out fProxyMxN Pb, out int Pbrow, out fProxyMxN _, out int _unused);
                            Gmat[ai, bi] = RowDotAt(in APa, AProw, in Pb, Pbrow, n);
                        }
                    }

                    var GmatLU = new fProxyMxN(in Gmat, Allocator.Temp);
                    var Xsol = new fProxyMxN(in Fmat, Allocator.Temp);
                    var piv = new Pivot(d, Allocator.Temp);
                    var luInfo = LU.solveInPlace(ref GmatLU, ref piv, ref Xsol);
                    piv.Dispose();
                    GmatLU.Dispose();

                    var evReal = new fProxyN(d);
                    var evImag = new fProxyN(d);
                    bool haveEig = false;
                    if (luInfo.Solved)
                    {
                        var eiInfo = Eigen.valuesQRInPlace(ref Xsol, ref evReal, ref evImag);
                        haveEig = eiInfo.status == IterativeSolveStatus.Converged;
                    }

                    fProxy huge = (fProxy)1e30;
                    var keys = new fProxyN(d);
                    for (int i = 0; i < d; i++)
                    {
                        if (!haveEig) { keys[i] = huge; continue; }
                        fProxy re = evReal[i], im = evImag[i];
                        fProxy imagTol = Consts.fProxyZeroThreshold * (math.abs(re) + (fProxy)1);
                        keys[i] = math.abs(im) <= imagTol ? math.abs(re) : huge;
                    }

                    int target = math.min(recycle, d);
                    var selIdx = new UnsafeList<int>(math.max(target, 1), Allocator.Temp);
                    for (int sIt = 0; sIt < target; sIt++)
                    {
                        int best = -1;
                        fProxy bestKey = huge;
                        for (int i = 0; i < d; i++)
                            if (keys[i] < bestKey) { bestKey = keys[i]; best = i; }
                        if (best < 0 || bestKey >= huge) break;
                        selIdx.Add(best);
                        keys[best] = huge;
                    }
                    int kNew = selIdx.Length;

                    int allocK = math.max(kNew, 1);
                    var Zsel = new fProxyMxN(d, allocK, Allocator.Temp, false);
                    var Ntheta = new fProxyMxN(d, d, Allocator.Temp, false);
                    var evTmp = new fProxyN(d);
                    var Vtmp = new fProxyMxN(d, d, Allocator.Temp, false);

                    bool eigOk = kNew > 0;
                    for (int sIt = 0; sIt < kNew && eigOk; sIt++)
                    {
                        int idx = selIdx[sIt];
                        fProxy th = evReal[idx];
                        for (int ai = 0; ai < d; ai++)
                            for (int bi = 0; bi < d; bi++)
                                Ntheta[ai, bi] = Fmat[ai, bi] - th * (Gmat[ai, bi] + Gmat[bi, ai]) + th * th * Pgram[ai, bi];

                        var symInfo = Eigen.symmetricInPlace(ref Ntheta, ref evTmp, ref Vtmp);
                        if (symInfo.status != IterativeSolveStatus.Converged) { eigOk = false; break; }
                        for (int r = 0; r < d; r++) Zsel[r, sIt] = Vtmp[r, d - 1];
                    }

                    if (eigOk && kNew > 0)
                    {
                        var Unew = new UnsafeList<fProxyN>(kNew, Allocator.Temp);
                        var AUraw = new fProxyMxN(n, kNew, Allocator.Temp, false);

                        for (int sIt = 0; sIt < kNew; sIt++)
                        {
                            var ucol = new fProxyN(n);
                            var aucol = new fProxyN(n);
                            for (int l = 0; l < d; l++)
                            {
                                fProxy zl = Zsel[l, sIt];
                                ResolveCombinedCol(l, kcurAtEntry, flexible, in Ublk, in AUblk, in V, in Zv, in AVlist, in off, k,
                                                    out fProxyMxN Pl, out int Prow, out fProxyMxN APl, out int AProw);
                                AddScaledRowInto(ref ucol, zl, in Pl, Prow, n);
                                AddScaledRowInto(ref aucol, zl, in APl, AProw, n);
                            }
                            Unew.Add(ucol);
                            for (int r = 0; r < n; r++) AUraw[r, sIt] = aucol[r];
                            aucol.Dispose();
                        }

                        var Rnew = new fProxyMxN(kNew, kNew, Allocator.Temp, false);
                        QR.decompInPlace(ref AUraw, ref Rnew);

                        fProxy rGuard = Consts.fProxyEpsilon * (fProxy)100 * (math.abs(Rnew[0, 0]) + (fProxy)1);
                        int kSafe = 0;
                        while (kSafe < kNew && math.abs(Rnew[kSafe, kSafe]) > rGuard) kSafe++;

                        if (kSafe > 0)
                        {
                            for (int i = 0; i < kSafe; i++)
                            {
                                for (int c = 0; c < n; c++) { Ublk[i, c] = Unew[i][c]; Cblk[i, c] = AUraw[c, i]; }
                                for (int l = 0; l < kSafe; l++) Ru[i, l] = i <= l ? Rnew[i, l] : (fProxy)0;
                            }
                            kcur = kSafe;
                        }

                        Rnew.Dispose();
                        AUraw.Dispose();
                        for (int sIt = 0; sIt < kNew; sIt++) Unew[sIt].Dispose();
                        Unew.Dispose();
                    }

                    Vtmp.Dispose(); evTmp.Dispose(); Ntheta.Dispose(); Zsel.Dispose();
                    selIdx.Dispose(); keys.Dispose();
                    evImag.Dispose(); evReal.Dispose();
                    Xsol.Dispose(); Pgram.Dispose(); Gmat.Dispose(); Fmat.Dispose();
                    if (hadOldU) AUblk.Dispose();
                }

                if (total >= maxIter) { status = IterativeSolveStatus.MaxIterations; break; }
                // else: loop back -- fresh restart, R0 recomputed from the just-updated X.
            }

            A.ApplyBlock(in X, ref R0, s);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) R0[i, c] = B[i, c] - R0[i, c];
            converged = CountConverged(in R0, in thr, s, n, out maxr);
            if (status == IterativeSolveStatus.Converged && converged < s)
                status = IterativeSolveStatus.MaxIterations;

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            Wbuf.Dispose(); Tbuf.Dispose(); R0.Dispose(); Wcombo.Dispose(); CorrBuf.Dispose();
            if (flexible)
            {
                for (int i = 0; i < m; i++) Zv[i].Dispose();
                Zv.Dispose();
                rowIn.Dispose(); rowOut.Dispose();
            }
            Hbuf.Dispose(); Gbuf.Dispose();
            HQscratch.Dispose(); Rscratch.Dispose(); Yscratch.Dispose(); QtGscratch.Dispose();
            Lbuf.Dispose(); HijBuf.Dispose(); YiBuf.Dispose();
            thr.Dispose(); w.Dispose(); off.Dispose();
            if (recycling)
            {
                for (int i = 0; i < m; i++) AVlist[i].Dispose();
                AVlist.Dispose();
                Ublk.Dispose(); Cblk.Dispose(); Ru.Dispose(); Bmat.Dispose();
                BijBuf.Dispose(); CtrBuf.Dispose(); ZprojBuf.Dispose();
            }

            return new BlockSolveInfo { rhs = s, converged = converged, iterations = total, maxRnorm = maxr, minActive = minActive, status = status };
        }

        // ---- unpreconditioned + concrete forwarders ------------------------------------------------

        /// <summary>Unpreconditioned block GCRO-DR -- forwards into the merged
        /// <see cref="bgcrodr{TOp, TPre}"/> with the identity preconditioner.</summary>
        public static BlockSolveInfo bgcrodr<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int recycle, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            return bgcrodr(in A, default(fProxyIdentityPreconditioner), in B, ref X, restart, recycle, maxIter, tol);
        }

        static int fProxyBgcrodrDefaultRecycle(int restart, int s) => math.min(10, math.max(0, restart * s - 1));

        /// <summary>Block GCRO-DR over a dense NON-symmetric <see cref="fProxyMxN"/> A, via
        /// <see cref="fProxyDenseOperatorGeneral"/> (bgmres's own general block-apply route).</summary>
        public static BlockSolveInfo bgcrodr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int recycle, int maxIter, fProxy tol)
            => bgcrodr(new fProxyDenseOperatorGeneral(in A), in B, ref X, restart, recycle, maxIter, tol);

        /// <summary>Block GCRO-DR over a dense non-symmetric A with defaults (restart = min(30, A.M_Rows),
        /// recycle = min(10, restart*B.M_Rows-1), maxIter = A.M_Rows, tol = sqrtEps).</summary>
        public static BlockSolveInfo bgcrodr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
        {
            int r = math.min(30, A.M_Rows);
            return bgcrodr(new fProxyDenseOperatorGeneral(in A), in B, ref X, r, fProxyBgcrodrDefaultRecycle(r, B.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>Block GCRO-DR over a block-sparse (BSR) non-symmetric A. Forwards via
        /// fProxyBSROperator.</summary>
        public static BlockSolveInfo bgcrodr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int recycle, int maxIter, fProxy tol)
            => bgcrodr(new fProxyBSROperator(in A), in B, ref X, restart, recycle, maxIter, tol);

        /// <summary>Block GCRO-DR over a BSR non-symmetric A with defaults (restart = min(30, A.M_Rows)).</summary>
        public static BlockSolveInfo bgcrodr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
        {
            int r = math.min(30, A.M_Rows);
            return bgcrodr(new fProxyBSROperator(in A), in B, ref X, r, fProxyBgcrodrDefaultRecycle(r, B.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>Right-preconditioned block GCRO-DR over a BSR matrix with an ILU(0) preconditioner.</summary>
        public static BlockSolveInfo bgcrodr(in fProxyBSR A, in fProxyILU0 M, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int recycle, int maxIter, fProxy tol)
            => bgcrodr(new fProxyBSROperator(in A), in M, in B, ref X, restart, recycle, maxIter, tol);

        /// <summary>ILU(0)-right-preconditioned block GCRO-DR over a BSR matrix with defaults
        /// (restart = min(30, A.M_Rows)).</summary>
        public static BlockSolveInfo bgcrodr(in fProxyBSR A, in fProxyILU0 M, in fProxyMxN B, ref fProxyMxN X)
        {
            int r = math.min(30, A.M_Rows);
            return bgcrodr(new fProxyBSROperator(in A), in M, in B, ref X, r, fProxyBgcrodrDefaultRecycle(r, B.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
