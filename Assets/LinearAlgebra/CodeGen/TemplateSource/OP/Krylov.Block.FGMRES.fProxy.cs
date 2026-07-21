using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ---- block flexible GMRES core (bfgmres) -------------------------------------------------------

        /// <summary>
        /// Restarted block Flexible GMRES(m) for a general (nonsymmetric) square A and s simultaneous
        /// right-hand sides -- <see cref="bgmres{TOp, TPre}"/>'s block Arnoldi/Givens/Hessenberg machinery
        /// combined with <see cref="fgmres{TOp, TPre}"/>'s flexible-basis update: the preconditioned block
        /// basis Z[j] = M⁻¹ V[j] is STORED per Arnoldi step (M may vary every step -- an inner-iterative or
        /// nonlinear preconditioner), and the solution update reads X += Σ Yᵢᵀ Zᵢ, never re-applying M to a
        /// combined vector the way <see cref="bgmres{TOp, TPre}"/> does. Same block Arnoldi (rank-revealing
        /// LQ deflation, active width w[j] &lt;= s, monotonically non-increasing within a restart cycle),
        /// same periodic dense-QR block-Hessenberg least-squares re-solve, same restart/maxIter contract as
        /// bgmres.
        ///
        /// B and X are s ROWS x n COLS (row j = the j-th RHS/solution, length n = A.Rows; requires
        /// s &lt;= n). X is warm-startable. Convergence is per column against tol²·‖B[j]‖², checked via the
        /// least-squares Pythagorean residual identity (no extra matvec). Owns its whole workspace via
        /// Allocator.Temp.
        ///
        /// With <see cref="fProxyIdentityPreconditioner"/> the IsIdentity fold makes Z[j] == V[j]: no Z
        /// workspace is allocated and the solution update reads straight off V, bit-identical to
        /// <see cref="bgmres{TOp, TPre}"/> under identity.
        ///
        /// Returns a <see cref="BlockSolveInfo"/> whose <see cref="BlockSolveInfo.minActive"/> is the
        /// smallest active search width reached over the whole solve (basis-rank deflation).
        /// </summary>
        public static BlockSolveInfo bfgmres<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("bfgmres: A must be square");
            int n = A.Rows;
            int s = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("bfgmres: B.N_Cols must equal A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("bfgmres: X must match B");
            if (restart < 1) throw new ArgumentException("bfgmres: restart must be >= 1");
            if (maxIter < 1) throw new ArgumentException("bfgmres: maxIter must be >= 1");
            if (s > n) throw new ArgumentException("bfgmres: B.M_Rows (s) must be <= A.Rows");

            int m = restart;

            // Basis V[0..m], each s x n (dominant O(m s n) term), exactly as bgmres. Wbuf/Tbuf/R0/Wcombo:
            // s x n Arnoldi and commit scratch. Hbuf/Gbuf/HQscratch/Rscratch/Yscratch/QtGscratch/Lbuf/
            // HijBuf/YiBuf: the same block-Hessenberg / periodic dense-QR least-squares machinery as bgmres,
            // unchanged.
            var V = new UnsafeList<fProxyMxN>(m + 1, Allocator.Temp);
            for (int i = 0; i <= m; i++) V.Add(new fProxyMxN(s, n, Allocator.Temp, true));

            var Wbuf   = new fProxyMxN(s, n, Allocator.Temp, false);
            var Tbuf   = new fProxyMxN(s, n, Allocator.Temp, true);
            var R0     = new fProxyMxN(s, n, Allocator.Temp, true);
            var Wcombo = new fProxyMxN(s, n, Allocator.Temp, true);

            // Z[0..m-1]: the FLEXIBLE preconditioned basis, one s x n block per Arnoldi step. Unlike
            // bgmres's single reusable scratch buffer (M applied once to the combined vector at commit
            // time, valid only for a fixed M), each step's preconditioned image must persist to the
            // cycle's commit -- M may differ step to step, so it cannot be re-applied to a combined
            // vector after the fact. Allocated only for a real M; under identity Z[i] aliases V[i] at
            // the commit step below, so no workspace is needed.
            UnsafeList<fProxyMxN> Z = default;
            fProxyN rowIn = default, rowOut = default;
            if (!M.IsIdentity)
            {
                Z = new UnsafeList<fProxyMxN>(m, Allocator.Temp);
                for (int i = 0; i < m; i++) Z.Add(new fProxyMxN(s, n, Allocator.Temp, true));
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

            // Per-column thresholds tol^2 ||B[j]||^2 (floored for zero/tiny-norm columns), original RHS order.
            BuildColumnThresholds(in B, ref thr, s, n, tol);

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int total = 0;
            int converged = 0;
            double maxr = 0;
            int minActive = s;

            while (total < maxIter)
            {
                // R0 = B - A X.
                A.ApplyBlock(in X, ref R0, s);
                for (int i = 0; i < s; i++)
                    for (int c = 0; c < n; c++) R0[i, c] = B[i, c] - R0[i, c];

                converged = CountConverged(in R0, in thr, s, n, out maxr);
                if (converged == s) { status = IterativeSolveStatus.Converged; break; }

                // Full reset of this cycle's block-Hessenberg / least-squares accumulators -- see
                // Krylov.bgmres's own DEVLOG entry for why the full (m+1)s x ms / (m+1)s x s extent
                // is cleared, not just the [0, s) slice.
                ZeroPrefix(ref Hbuf, (m + 1) * s, m * s);
                ZeroPrefix(ref Gbuf, (m + 1) * s, s);

                var Ppiv0 = new Pivot(s, Allocator.Temp);
                var L0 = View(Lbuf, s);
                var Q0 = V[0];   // LQRP decomposes directly into V[0]'s own s x n buffer
                LQRP.decomp(in R0, ref L0, ref Q0, ref Ppiv0);
                Ppiv0.Dispose();

                w[0] = LQRPRank(in L0, s, n);
                minActive = math.min(minActive, w[0]);
                if (w[0] == 0) { status = IterativeSolveStatus.Breakdown; break; }   // R0 != 0 yet rank 0 -- defensive

                off[0] = 0; off[1] = w[0];
                var V0 = RowsView(Q0, w[0]);
                var G0 = RowsView(Gbuf, w[0]);
                BlockCrossGram(in V0, in R0, ref G0);   // G0 = V0 R0^T -- R0 expressed in V0's basis

                // ---- inner block-Arnoldi loop: builds V[1..k], Z[0..k-1], H's columns 0..k-1 ----
                int k = 0;
                bool cycleConverged = false;
                bool lsBreakdown = false;
                for (int j = 0; j < m && total < maxIter; j++)
                {
                    var Vj = RowsView(V[j], w[j]);
                    var Wj = RowsView(Wbuf, w[j]);
                    if (M.IsIdentity)
                    {
                        A.ApplyBlock(in Vj, ref Wj, w[j]);
                    }
                    else
                    {
                        // z_j = M⁻¹ v_j -- the CURRENT step's preconditioner, stored into Z[j] (not a
                        // scratch buffer) so a later step's differently-preconditioned M cannot corrupt
                        // it. w = A z_j.
                        var Zj = RowsView(Z[j], w[j]);
                        BlockApplyPre(in M, in Vj, ref Zj, w[j], n, ref rowIn, ref rowOut);
                        A.ApplyBlock(in Zj, ref Wj, w[j]);
                    }

                    int wj1 = BlockArnoldiMGS2Step(ref Wj, in V, ref w, ref off, ref Hbuf, ref HijBuf, ref Tbuf,
                                                    ref minActive, j, n);

                    total++;
                    k = j + 1;

                    // Least-squares solve (periodic dense re-QR of the accumulated Hbuf prefix) + a
                    // per-column residual check via the Pythagorean LS-residual identity -- no extra
                    // matvec, mirrors bgmres/scalar gmres's O(1) per-step check.
                    cycleConverged = BlockLSResolveAndCheck(in Hbuf, in off, k, ref HQscratch, ref Rscratch,
                                                             ref Gbuf, ref Yscratch, ref QtGscratch, in thr, s,
                                                             out lsBreakdown);
                    if (lsBreakdown) break;

                    // Happy breakdown: the block Krylov subspace stopped growing -- the just-computed Y
                    // is already exact for the reachable subspace, so no further step can help.
                    if (cycleConverged || wj1 == 0) break;
                }

                if (lsBreakdown)
                {
                    // X was never updated this cycle -- the shared post-loop recompute below reports
                    // the TRUE fresh residual at the returned X, never this cycle's poisoned solve.
                    status = IterativeSolveStatus.Breakdown;
                    break;
                }

                // Commit: X += combine(Y, Z[0..k-1]) -- the FLEXIBLE update, reading the STORED per-step
                // preconditioned basis (valid even when M varied across i = 0..k-1), unlike bgmres's
                // single apply-M-once-to-the-combination. Identity: Z[i] aliases V[i], so this reduces to
                // bgmres's own commit exactly. k >= 1 always reached here (the for-loop body runs at
                // least once, since total < maxIter held at cycle entry).
                {
                    int totalCols = off[k];
                    var Yfinal = RowsView(Yscratch, totalCols);
                    ZeroPrefix(ref Wcombo, s, n);
                    for (int i = 0; i < k; i++)
                    {
                        var Yi = RectView(YiBuf, w[i], s);
                        CopyRowsFrom(in Yfinal, off[i], w[i], ref Yi);
                        var Ti = RowsView(Tbuf, s);
                        if (M.IsIdentity)
                        {
                            var Vi = RowsView(V[i], w[i]);
                            BlockCTV(in Yi, in Vi, ref Ti);
                        }
                        else
                        {
                            var Zi = RowsView(Z[i], w[i]);
                            BlockCTV(in Yi, in Zi, ref Ti);
                        }
                        BlockAdd(ref Wcombo, in Ti, (fProxy)1);
                    }
                    BlockAdd(ref X, in Wcombo, (fProxy)1);
                }

                if (cycleConverged) { status = IterativeSolveStatus.Converged; break; }
                if (total >= maxIter) { status = IterativeSolveStatus.MaxIterations; break; }
                // else: loop back -- fresh restart, R0 recomputed from the just-updated X.
            }

            // Fresh residual from the final X (every BlockSolveInfo field is documented "at the
            // returned X") -- a cycle's top-of-loop CountConverged predates that cycle's own Commit.
            A.ApplyBlock(in X, ref R0, s);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) R0[i, c] = B[i, c] - R0[i, c];
            converged = CountConverged(in R0, in thr, s, n, out maxr);
            if (status == IterativeSolveStatus.Converged && converged < s)
                status = IterativeSolveStatus.MaxIterations;

            for (int i = 0; i <= m; i++) V[i].Dispose();
            V.Dispose();
            Wbuf.Dispose(); Tbuf.Dispose(); R0.Dispose(); Wcombo.Dispose();
            if (!M.IsIdentity)
            {
                for (int i = 0; i < m; i++) Z[i].Dispose();
                Z.Dispose();
                rowIn.Dispose(); rowOut.Dispose();
            }
            Hbuf.Dispose(); Gbuf.Dispose();
            HQscratch.Dispose(); Rscratch.Dispose(); Yscratch.Dispose(); QtGscratch.Dispose();
            Lbuf.Dispose(); HijBuf.Dispose(); YiBuf.Dispose();
            thr.Dispose(); w.Dispose(); off.Dispose();

            return new BlockSolveInfo { rhs = s, converged = converged, iterations = total, maxRnorm = maxr, minActive = minActive, status = status };
        }

        // ---- unpreconditioned + concrete forwarders --------------------------------------------------

        /// <summary>Unpreconditioned restarted block flexible GMRES(m) -- forwards into the merged
        /// <see cref="bfgmres{TOp, TPre}"/> with the identity preconditioner.</summary>
        public static BlockSolveInfo bfgmres<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            return bfgmres(in A, default(fProxyIdentityPreconditioner), in B, ref X, restart, maxIter, tol);
        }

        /// <summary>Block flexible GMRES(m) over a dense NON-symmetric <see cref="fProxyMxN"/> A, via
        /// <see cref="fProxyDenseOperatorGeneral"/> (general block apply -- <see cref="fProxyDenseOperator"/>'s
        /// ApplyBlock is symmetric-only and would silently solve Aᵀx=b here).</summary>
        public static BlockSolveInfo bfgmres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int maxIter, fProxy tol)
            => bfgmres(new fProxyDenseOperatorGeneral(in A), in B, ref X, restart, maxIter, tol);

        /// <summary>Block flexible GMRES over a dense non-symmetric A with defaults (restart = min(30,
        /// A.M_Rows), maxIter = A.M_Rows, tol = sqrtEps).</summary>
        public static BlockSolveInfo bfgmres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => bfgmres(new fProxyDenseOperatorGeneral(in A), in B, ref X, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Block flexible GMRES(m) over a block-sparse (BSR) non-symmetric A. Allocates block
        /// scratch from the arena.</summary>
        public static BlockSolveInfo bfgmres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int maxIter, fProxy tol)
            => bfgmres(new fProxyBSROperator(in A), in B, ref X, restart, maxIter, tol);

        /// <summary>Block flexible GMRES over a BSR non-symmetric A with defaults (restart = min(30,
        /// A.M_Rows)).</summary>
        public static BlockSolveInfo bfgmres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
            => bfgmres(new fProxyBSROperator(in A), in B, ref X, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Right-preconditioned block flexible GMRES(m) over a BSR non-symmetric A with an
        /// ILU(0) preconditioner. A single fixed ILU(0) here exercises the SAME reduction bgmres already
        /// covers; bfgmres's own advantage is a genuinely varying M -- see the bespoke flexible test.</summary>
        public static BlockSolveInfo bfgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyMxN B, ref fProxyMxN X,
                                        int restart, int maxIter, fProxy tol)
            => bfgmres(new fProxyBSROperator(in A), in M, in B, ref X, restart, maxIter, tol);

        /// <summary>ILU(0)-right-preconditioned block flexible GMRES over a BSR non-symmetric A with
        /// defaults (restart = min(30, A.M_Rows)).</summary>
        public static BlockSolveInfo bfgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyMxN B, ref fProxyMxN X)
            => bfgmres(new fProxyBSROperator(in A), in M, in B, ref X, math.min(30, A.M_Rows), A.M_Rows, Consts.fProxySqrtEps);
    }
}
