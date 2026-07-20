using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ---- bminres private helpers ----------------------------------------------------------------

        // dst[rowOffset+r, c] = src[r, c] for r<rows, c<cols. Stride-safe: uses each buffer's own
        // indexer, so src/dst may have different native column counts.
        static void CopyRowsAt(in fProxyMxN src, ref fProxyMxN dst, int rowOffset, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    dst[rowOffset + r, c] = src[r, c];
        }

        // dst[r, colOffset+c] = src[r, c] for r<rows, c<cols.
        static void CopyColsAt(in fProxyMxN src, ref fProxyMxN dst, int colOffset, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    dst[r, colOffset + c] = src[r, c];
        }

        // dst[r, c] = src[srcRowOff+r, srcColOff+c] for r<rows, c<cols.
        static void CopyBlockAt(in fProxyMxN src, int srcRowOff, int srcColOff, ref fProxyMxN dst, int rows, int cols)
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    dst[r, c] = src[srcRowOff + r, srcColOff + c];
        }

        // Un-pivots Beta's ROWS in place: a row-pivoted normalize (LQRP.decomp / CHOP.decomp) returns
        // Beta with (P.W)[j,:] == (Beta.Vout)[j,:] -- Beta's row j is the CANDIDATE that ended up at
        // pivoted position j, i.e. W's original row P[j], not row j of W itself. Every consumer outside
        // this normalize (the block-Lanczos recurrence's Beta^T . Vprev term, and Beta's placement in
        // M2/BuildOmega) needs Beta indexed by W's ORIGINAL row order instead, since that is the order
        // Vprev/Alfa/OmegaOld already carry -- scatter Beta's rows through P to restore it. Beta's
        // COLUMNS pair with Vout's own row order and are untouched (Vout is not re-pivoted). See
        // OP/DEVLOG.md.
        static void UnpivotBetaRows(ref fProxyMxN Beta, in Pivot P, int s)
        {
            var tmp = new fProxyMxN(s, s, Allocator.Temp, true);
            CopyMat(in Beta, ref tmp, s);
            for (int j = 0; j < s; j++)
            {
                int orig = P[j];
                for (int c = 0; c < s; c++)
                    Beta[orig, c] = tmp[j, c];
            }
            tmp.Dispose();
        }

        // Rank-revealing block-Lanczos normalization, unpreconditioned: row-pivoted LQ (LQRP) of W
        // (s x n) directly. Beta receives the s x s L factor, UN-pivoted back to W's own row order (see
        // UnpivotBetaRows); Vout receives the new orthonormal Lanczos vectors in its leading `rank`
        // rows, with rows [rank, s) forced to zero -- a deflated lane stays exactly zero for the rest of
        // the solve (ApplyBlock of a zero row is zero), so every later block computation absorbs it
        // harmlessly with no width bookkeeping. W and Vout may alias (LQRP.decomp copies its input
        // before writing Q).
        static RankInfo BlockNormalizeIdentity(in fProxyMxN W, ref fProxyMxN Beta, ref fProxyMxN Vout,
                                                ref Pivot P, int s, int n)
        {
            LQRP.decomp(in W, ref Beta, ref Vout, ref P);
            int rank = LQRPRank(in Beta, s, n);
            for (int i = rank; i < s; i++)
                for (int c = 0; c < n; c++)
                    Vout[i, c] = (fProxy)0;
            UnpivotBetaRows(ref Beta, in P, s);
            return new RankInfo { status = rank < s ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success, rank = rank };
        }

        // Rank-revealing block-Lanczos normalization, preconditioned: Beta comes from a pivoted
        // Cholesky (CHOP) of the M^-1-weighted cross Gram G = W.Z^T (Z = M^-1.W, computed by the
        // caller via BlockApplyPre) -- matching scalar minres's own beta = sqrt(<r,z>) weighting, NOT
        // ||z|| (which pivoted-LQ directly on Z would give). Vout receives the new orthonormal Lanczos
        // vectors, gathered from Z's pivoted rows and triangular-solved by Beta's revealed rank x rank
        // corner, in its leading `rank` rows; rows [rank, s) are forced to zero (see
        // BlockNormalizeIdentity). Beta is UN-pivoted back to W's own row order before return (see
        // UnpivotBetaRows). Vout must be distinct from Z. Returns Indefinite only if G itself is
        // not PSD (round-off on an SPD M).
        static RankInfo BlockNormalizePrecond(in fProxyMxN W, in fProxyMxN Z, ref fProxyMxN G, ref fProxyMxN Beta,
                                               ref Pivot P, ref fProxyMxN Vout, int s, int n)
        {
            BlockGram(in W, in Z, ref G, s);
            var info = CHOP.decomp(in G, ref Beta, ref P);

            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                    Vout[i, c] = (fProxy)0;

            if (!info.Solved || info.rank == 0)
                return info;

            int rank = info.rank;
            for (int i = 0; i < rank; i++)
            {
                int p = P[i];
                for (int c = 0; c < n; c++)
                    Vout[i, c] = Z[p, c];
            }

            var corner = new fProxyMxN(rank, rank, Allocator.Temp, true);
            CopyBlock(in Beta, ref corner, rank, rank);
            var voutRank = RowsView(Vout, rank);
            Blas.triLower(ref corner, ref voutRank);
            corner.Dispose();

            UnpivotBetaRows(ref Beta, in P, s);

            return info;
        }

        // Builds the 2s x 2s orthogonal completion Omega = [Qy | Qperp] of the thin-QR left factor Qy
        // of the 2s x s stack [Gbar; Beta^T] (Gamma is that QR's s x s R-factor). Qperp completes Qy to a
        // full orthogonal basis via one of two fixed seed subspaces ([0;I] or [I;0]), whichever yields
        // a full-rank residual -- the specific choice among valid completions is gauge freedom (see
        // OP/DEVLOG.md); only genuine orthogonality and internal consistency matter to the caller.
        // Returns false (breakdown) if Gamma itself is near-singular (Y = [Gbar;Beta^T] is rank-
        // deficient -- this search direction carries no new information) or if BOTH Qperp completion
        // seeds are rank-deficient.
        static bool BuildOmega(in fProxyMxN Gbar, in fProxyMxN Beta, ref fProxyMxN Omega, ref fProxyMxN Gamma, int s)
        {
            int s2 = 2 * s;
            var Y = new fProxyMxN(s2, s, Allocator.Temp, true);
            CopyRowsAt(in Gbar, ref Y, 0, s, s);
            for (int r = 0; r < s; r++)
                for (int c = 0; c < s; c++)
                    Y[s + r, c] = Beta[c, r];   // Beta^T

            var Qy = new fProxyMxN(s2, s, Allocator.Temp, true);
            QR.decomp(in Y, ref Qy, ref Gamma);

            bool gammaOk = true;
            for (int i = 0; i < s; i++)
                if (math.abs(Gamma[i, i]) <= Consts.fProxyZeroThreshold) { gammaOk = false; break; }

            var Z0 = new fProxyMxN(s2, s, Allocator.Temp, true);
            var T = new fProxyMxN(s, s, Allocator.Temp, true);
            var QyT = new fProxyMxN(s2, s, Allocator.Temp, true);
            var Z1 = new fProxyMxN(s2, s, Allocator.Temp, true);
            var Qperp = new fProxyMxN(s2, s, Allocator.Temp, true);
            var Rz = new fProxyMxN(s, s, Allocator.Temp, true);

            bool ok = false;
            for (int seed = 0; seed < 2 && !ok && gammaOk; seed++)
            {
                for (int r = 0; r < s2; r++)
                    for (int c = 0; c < s; c++)
                        Z0[r, c] = (fProxy)0;
                if (seed == 0) { for (int i = 0; i < s; i++) Z0[s + i, i] = (fProxy)1; }   // [0; I]
                else           { for (int i = 0; i < s; i++) Z0[i, i] = (fProxy)1; }       // [I; 0]

                Blas.dot(in Qy, in Z0, ref T, true, false);     // T = Qy^T . Z0
                Blas.dot(in Qy, in T, ref QyT, false, false);    // QyT = Qy . T
                for (int r = 0; r < s2; r++)
                    for (int c = 0; c < s; c++)
                        Z1[r, c] = Z0[r, c] - QyT[r, c];

                QR.decomp(in Z1, ref Qperp, ref Rz);

                bool rankOk = true;
                for (int i = 0; i < s; i++)
                    if (math.abs(Rz[i, i]) <= Consts.fProxyZeroThreshold) { rankOk = false; break; }
                ok = rankOk;
            }

            if (ok)
            {
                CopyColsAt(in Qy, ref Omega, 0, s2, s);
                CopyColsAt(in Qperp, ref Omega, s, s2, s);
            }

            Y.Dispose(); Qy.Dispose(); Z0.Dispose(); T.Dispose(); QyT.Dispose(); Z1.Dispose(); Qperp.Dispose(); Rz.Dispose();
            return ok;
        }

        // ---- block MINRES core (bminres) -------------------------------------------------------------

        /// <summary>
        /// Zero-alloc block (multi-RHS) MINRES for a SYMMETRIC, possibly INDEFINITE A and s
        /// simultaneous right-hand sides, generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>)
        /// and the preconditioner (<see cref="IfProxyPreconditioner"/>). A TRUE block method: one shared
        /// block-Lanczos subspace built from all s RHS, s x s block coefficients, one <c>ApplyBlock</c>
        /// per iteration -- not s independent scalar <see cref="minres{TOp, TPre}"/> solves. Unlike
        /// <see cref="bcg{TOp, TPre}"/>, A need NOT be SPD; none of this solver's s x s factorizations
        /// (block-Lanczos normalization, the Gamma search-direction solve) assume symmetric-definite
        /// coefficients. Uses fully-normalized block Lanczos vectors throughout (no separate
        /// unnormalized copy, unlike scalar minres's r1/r2 bookkeeping -- a block-friendlier but
        /// mathematically equivalent formulation of the same recurrence).
        ///
        /// B and X are s ROWS x n COLS (row j = the j-th RHS/solution, length n = A.Rows; requires
        /// s &lt;= n). X is warm-startable. Vprev, Vcur, Wk, W, W1, W2 are s x n block scratch; Z is s x n,
        /// UNUSED under the identity preconditioner (pass <c>default</c>). Convergence is per column
        /// against tol²·‖B[j]‖². A rank-deficient block-Lanczos step (dependent/near-dependent
        /// candidate vector) is handled by a rank-revealing normalization (LQRP unpreconditioned, a
        /// pivoted Cholesky of the M-weighted Gram under preconditioning): the deflated lane is zeroed
        /// and contributes nothing further, without shrinking any buffer or reporting
        /// <see cref="BlockSolveInfo.minActive"/> below the smallest rank actually revealed -- rows of
        /// B that are exactly (or near-exactly) dependent are NOT currently guaranteed to produce
        /// matching rows of X (see OP/DEVLOG.md). A <see cref="IterativeSolveStatus.Converged"/> result
        /// is ALWAYS confirmed by a fresh B - A·X residual check before being reported (never trusted
        /// from the internal recursion alone). Returns a <see cref="BlockSolveInfo"/>.
        /// </summary>
        /// <exception cref="NotSupportedException">M is not the identity preconditioner -- the
        /// preconditioned path is not yet verified correct (see OP/DEVLOG.md).</exception>
        public static BlockSolveInfo bminres<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        ref fProxyMxN Vprev, ref fProxyMxN Vcur, ref fProxyMxN Wk,
                                        ref fProxyMxN W, ref fProxyMxN W1, ref fProxyMxN W2, ref fProxyMxN Z,
                                        int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("bminres (block): A must be square");
            int n = A.Rows;
            int s = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("bminres (block): B must be s x A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("bminres (block): X must match B");
            if (Vprev.M_Rows != s || Vprev.N_Cols != n) throw new ArgumentException("bminres (block): Vprev must match B");
            if (Vcur.M_Rows != s || Vcur.N_Cols != n) throw new ArgumentException("bminres (block): Vcur must match B");
            if (Wk.M_Rows != s || Wk.N_Cols != n) throw new ArgumentException("bminres (block): Wk must match B");
            if (W.M_Rows != s || W.N_Cols != n) throw new ArgumentException("bminres (block): W must match B");
            if (W1.M_Rows != s || W1.N_Cols != n) throw new ArgumentException("bminres (block): W1 must match B");
            if (W2.M_Rows != s || W2.N_Cols != n) throw new ArgumentException("bminres (block): W2 must match B");
            if (!M.IsIdentity && (Z.M_Rows != s || Z.N_Cols != n))
                throw new ArgumentException("bminres (block): Z must match B");
            if (s > n) throw new ArgumentException("bminres (block): B.M_Rows (s) must be <= A.Rows");
            if (maxIter < 1) throw new ArgumentException("bminres (block): maxIter must be >= 1");
            if (!M.IsIdentity)
                throw new NotSupportedException("bminres (block): a non-identity preconditioner is not yet verified correct -- use the unpreconditioned overload, or per-column scalar minres with a preconditioner, instead (see OP/DEVLOG.md).");

            unsafe
            {
                int cnt = M.IsIdentity ? 8 : 9;
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)Vprev.Data.Ptr; ptrs[1] = (long)Vcur.Data.Ptr; ptrs[2] = (long)Wk.Data.Ptr;
                ptrs[3] = (long)W.Data.Ptr;     ptrs[4] = (long)W1.Data.Ptr;   ptrs[5] = (long)W2.Data.Ptr;
                ptrs[6] = (long)X.Data.Ptr;     ptrs[7] = (long)B.Data.Ptr;
                if (!M.IsIdentity) ptrs[8] = (long)Z.Data.Ptr;
                RequireDistinctBuffers("bminres (block): Vprev/Vcur/Wk/W/W1/W2/Z/X/B must be distinct", ptrs, cnt);
            }

            int s2 = 2 * s;
            var Alfa = new fProxyMxN(s, s, Allocator.Temp, true);
            var Beta = new fProxyMxN(s, s, Allocator.Temp, true);
            var Dbar = new fProxyMxN(s, s, Allocator.Temp, true);
            var Epsln = new fProxyMxN(s, s, Allocator.Temp, true);
            var OldEps = new fProxyMxN(s, s, Allocator.Temp, true);
            var Phibar = new fProxyMxN(s, s, Allocator.Temp, true);
            var Phi = new fProxyMxN(s, s, Allocator.Temp, true);
            var Delta = new fProxyMxN(s, s, Allocator.Temp, true);
            var Gbar = new fProxyMxN(s, s, Allocator.Temp, true);
            var Gamma = new fProxyMxN(s, s, Allocator.Temp, true);
            var GammaCopy = new fProxyMxN(s, s, Allocator.Temp, true);
            var OmegaOld = new fProxyMxN(s2, s2, Allocator.Temp, true);
            var OmegaNew = new fProxyMxN(s2, s2, Allocator.Temp, true);
            var M2 = new fProxyMxN(s2, s2, Allocator.Temp, true);
            var Result = new fProxyMxN(s2, s2, Allocator.Temp, true);
            var PhibarStack = new fProxyMxN(s2, s, Allocator.Temp, true);
            var Res2 = new fProxyMxN(s2, s, Allocator.Temp, true);
            var T = new fProxyMxN(s, n, Allocator.Temp, true);
            var thr = new fProxyN(s);
            var Pnorm = new Pivot(s, Allocator.Temp);

            fProxyMxN Gnorm = default;
            fProxyN rowIn = default, rowOut = default;
            if (!M.IsIdentity)
            {
                Gnorm = new fProxyMxN(s, s, Allocator.Temp, true);
                rowIn = new fProxyN(n);
                rowOut = new fProxyN(n);
            }

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int iters = maxIter;
            int converged = 0;
            double maxr = 0;
            int minActive = s;

            for (int j = 0; j < s; j++)
            {
                fProxy bb = (fProxy)0;
                for (int c = 0; c < n; c++) bb += B[j, c] * B[j, c];
                thr[j] = tol * tol * bb;
            }

            bool allZero = true;
            for (int j = 0; j < s; j++) if (thr[j] != (fProxy)0) { allZero = false; break; }
            if (allZero)
            {
                CopyBlock(in B, ref X, s, n);
                status = IterativeSolveStatus.Converged;
                iters = 0;
                goto cleanup;
            }

            // R0 = B - A.X (into Wk).
            A.ApplyBlock(in X, ref Wk, s);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) Wk[i, c] = B[i, c] - Wk[i, c];

            converged = CountConverged(in Wk, in thr, s, n, out maxr);
            if (converged == s) { status = IterativeSolveStatus.Converged; iters = 0; goto cleanup; }

            RankInfo info0;
            if (M.IsIdentity)
            {
                info0 = BlockNormalizeIdentity(in Wk, ref Beta, ref Vcur, ref Pnorm, s, n);
            }
            else
            {
                BlockApplyPre(in M, in Wk, ref Z, s, n, ref rowIn, ref rowOut);
                info0 = BlockNormalizePrecond(in Wk, in Z, ref Gnorm, ref Beta, ref Pnorm, ref Vcur, s, n);
            }
            if (!info0.Solved || info0.rank == 0) { status = IterativeSolveStatus.Breakdown; iters = 0; goto cleanup; }
            minActive = math.min(minActive, info0.rank);

            Blas.trans(in Beta, ref Phibar);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < s; c++) { Dbar[i, c] = (fProxy)0; Epsln[i, c] = (fProxy)0; }
            for (int r = 0; r < s2; r++)
                for (int c = 0; c < s2; c++) OmegaOld[r, c] = (fProxy)0;
            for (int i = 0; i < s; i++) { OmegaOld[i, i] = (fProxy)(-1); OmegaOld[s + i, s + i] = (fProxy)1; }
            // W1/W2 seed the k=0 search-direction recurrence's OldEps/Delta terms, both exactly zero at
            // k=0 -- but W itself must ALSO start zeroed: the roll below folds W into W2 before Delta's
            // (zero) coefficient is applied, and 0 * NaN is NaN, not 0, so an uninitialized W would
            // poison the whole solve if the caller passed unzeroed scratch (see OP/DEVLOG.md).
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) { W1[i, c] = (fProxy)0; W2[i, c] = (fProxy)0; W[i, c] = (fProxy)0; }

            for (int k = 0; k < maxIter; k++)
            {
                // ---- Lanczos step: produces Alfa, the new Beta, and the next Lanczos vector (in Wk) ----
                A.ApplyBlock(in Vcur, ref Wk, s);
                if (k >= 1)
                {
                    BlockCTV(in Beta, in Vprev, ref T);
                    BlockAdd(ref Wk, in T, (fProxy)(-1));
                }
                BlockGram(in Vcur, in Wk, ref Alfa, s);
                BlockCTV(in Alfa, in Vcur, ref T);
                BlockAdd(ref Wk, in T, (fProxy)(-1));

                RankInfo infoK;
                if (M.IsIdentity)
                {
                    infoK = BlockNormalizeIdentity(in Wk, ref Beta, ref Wk, ref Pnorm, s, n);
                }
                else
                {
                    BlockApplyPre(in M, in Wk, ref Z, s, n, ref rowIn, ref rowOut);
                    infoK = BlockNormalizePrecond(in Wk, in Z, ref Gnorm, ref Beta, ref Pnorm, ref Wk, s, n);
                }
                if (!infoK.Solved || infoK.rank == 0) { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }
                minActive = math.min(minActive, infoK.rank);

                // ---- apply the OLD Omega to the stacked block-2x2 (Dbar,0 ; Alfa,Beta) ----
                for (int r = 0; r < s2; r++)
                    for (int c = 0; c < s2; c++) M2[r, c] = (fProxy)0;
                CopyRowsAt(in Dbar, ref M2, 0, s, s);       // top-left
                CopyRowsAt(in Alfa, ref M2, s, s, s);       // bottom-left (cols 0..s of row-block 1)
                for (int r = 0; r < s; r++)
                    for (int c = 0; c < s; c++) M2[s + r, s + c] = Beta[r, c];   // bottom-right

                Blas.dot(in OmegaOld, in M2, ref Result, true, false);

                CopyMat(in Epsln, ref OldEps, s);
                CopyBlockAt(in Result, 0, 0, ref Delta, s, s);
                CopyBlockAt(in Result, 0, s, ref Epsln, s, s);
                CopyBlockAt(in Result, s, 0, ref Gbar, s, s);
                CopyBlockAt(in Result, s, s, ref Dbar, s, s);

                // ---- new Omega from (Gbar, Beta) ----
                if (!BuildOmega(in Gbar, in Beta, ref OmegaNew, ref Gamma, s))
                { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                // ---- RHS update (Phi/Phibar) ----
                for (int r = 0; r < s2; r++)
                    for (int c = 0; c < s; c++) PhibarStack[r, c] = (fProxy)0;
                CopyRowsAt(in Phibar, ref PhibarStack, 0, s, s);
                Blas.dot(in OmegaNew, in PhibarStack, ref Res2, true, false);
                CopyBlockAt(in Res2, 0, 0, ref Phi, s, s);
                CopyBlockAt(in Res2, s, 0, ref Phibar, s, s);

                // ---- search-direction update: Gamma^T.Wnew = Vcur - OldEps^T.W1 - Delta^T.W2 ----
                // (block generalization of scalar's w=(v-oldeps.w1-delta.w2)/gamma: the block search-
                // direction identity M.R=U, transposed to this file's row convention, puts Gamma on
                // the solve's LEFT as its TRANSPOSE, not itself -- see OP/DEVLOG.md.)
                { var tmp = W1; W1 = W2; W2 = W; W = tmp; }
                CopyBlock(in Vcur, ref W, s, n);
                BlockCTV(in OldEps, in W1, ref T); BlockAdd(ref W, in T, (fProxy)(-1));
                BlockCTV(in Delta, in W2, ref T);  BlockAdd(ref W, in T, (fProxy)(-1));

                CopyMat(in Gamma, ref GammaCopy, s);
                var pivS = new Pivot(s, Allocator.Temp);
                var luInfo = LU.solveInPlaceTransA(ref GammaCopy, ref pivS, ref W);
                pivS.Dispose();
                if (luInfo.status != DirectSolveStatus.Success)
                { status = IterativeSolveStatus.Breakdown; iters = k; goto cleanup; }

                // ---- X update ----
                BlockCTV(in Phi, in W, ref T);
                BlockAdd(ref X, in T, (fProxy)1);

                // ---- roll the Lanczos vector history and the old/new Omega ----
                { var tmp = Vprev; Vprev = Vcur; Vcur = Wk; Wk = tmp; }
                { var tmp = OmegaOld; OmegaOld = OmegaNew; OmegaNew = tmp; }

                // ---- cheap per-column probe, then a MANDATORY fresh verify before trusting it ----
                bool probeOk = true;
                for (int j = 0; j < s; j++)
                {
                    fProxy sumsq = (fProxy)0;
                    for (int i = 0; i < s; i++) sumsq += Phibar[i, j] * Phibar[i, j];
                    if (sumsq > thr[j]) { probeOk = false; break; }
                }
                if (probeOk)
                {
                    A.ApplyBlock(in X, ref T, s);
                    for (int i = 0; i < s; i++)
                        for (int c = 0; c < n; c++) T[i, c] = B[i, c] - T[i, c];
                    if (CountConverged(in T, in thr, s, n, out _) == s)
                    { status = IterativeSolveStatus.Converged; iters = k + 1; goto cleanup; }
                }
            }

            status = IterativeSolveStatus.MaxIterations;
            iters = maxIter;

        cleanup:
            A.ApplyBlock(in X, ref T, s);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) T[i, c] = B[i, c] - T[i, c];
            converged = CountConverged(in T, in thr, s, n, out maxr);

            Alfa.Dispose(); Beta.Dispose(); Dbar.Dispose(); Epsln.Dispose(); OldEps.Dispose();
            Phibar.Dispose(); Phi.Dispose(); Delta.Dispose(); Gbar.Dispose(); Gamma.Dispose(); GammaCopy.Dispose();
            OmegaOld.Dispose(); OmegaNew.Dispose(); M2.Dispose(); Result.Dispose();
            PhibarStack.Dispose(); Res2.Dispose(); T.Dispose(); thr.Dispose(); Pnorm.Dispose();
            if (!M.IsIdentity) { Gnorm.Dispose(); rowIn.Dispose(); rowOut.Dispose(); }

            return new BlockSolveInfo { rhs = s, converged = converged, iterations = iters, maxRnorm = maxr, minActive = minActive, status = status };
        }

        // ---- unpreconditioned + concrete forwarders ------------------------------------------------

        /// <summary>Unpreconditioned block-MINRES -- forwards into the merged
        /// <see cref="bminres{TOp, TPre}"/> with the identity preconditioner (needs no Z block).</summary>
        public static BlockSolveInfo bminres<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                        ref fProxyMxN Vprev, ref fProxyMxN Vcur, ref fProxyMxN Wk,
                                        ref fProxyMxN W, ref fProxyMxN W1, ref fProxyMxN W2,
                                        int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyMxN Z = default;
            return bminres(in A, default(fProxyIdentityPreconditioner), in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, ref Z, maxIter, tol);
        }

        /// <summary>Block-MINRES over a dense symmetric (possibly indefinite) <see cref="fProxyMxN"/> A
        /// (n x n) with an s x n block B. Allocates block scratch from the arena.</summary>
        public static BlockSolveInfo bminres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Vprev = B.fProxyTempMat(s, n, true), Vcur = B.fProxyTempMat(s, n, true), Wk = B.fProxyTempMat(s, n, true),
                      W = B.fProxyTempMat(s, n, true), W1 = B.fProxyTempMat(s, n, true), W2 = B.fProxyTempMat(s, n, true);
            return bminres(new fProxyDenseOperator(in A), in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, maxIter, tol);
        }

        /// <summary>Block-MINRES over a dense symmetric A with default maxIter (A.M_Rows) and tol (sqrtEps).</summary>
        public static BlockSolveInfo bminres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => bminres(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Preconditioned block-MINRES over a dense symmetric A. Allocates block scratch (incl. Z).</summary>
        /// <exception cref="NotSupportedException">M is not the identity preconditioner (see
        /// <see cref="bminres{TOp, TPre}"/>).</exception>
        public static BlockSolveInfo bminres<TPre>(in fProxyMxN A, in TPre M, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Vprev = B.fProxyTempMat(s, n, true), Vcur = B.fProxyTempMat(s, n, true), Wk = B.fProxyTempMat(s, n, true),
                      W = B.fProxyTempMat(s, n, true), W1 = B.fProxyTempMat(s, n, true), W2 = B.fProxyTempMat(s, n, true),
                      Z = B.fProxyTempMat(s, n, true);
            return bminres(new fProxyDenseOperator(in A), in M, in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, ref Z, maxIter, tol);
        }

        /// <summary>Block-MINRES over a block-sparse (BSR) symmetric A with an s x n block B. Allocates
        /// block scratch from the arena.</summary>
        public static BlockSolveInfo bminres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Vprev = B.fProxyTempMat(s, n, true), Vcur = B.fProxyTempMat(s, n, true), Wk = B.fProxyTempMat(s, n, true),
                      W = B.fProxyTempMat(s, n, true), W1 = B.fProxyTempMat(s, n, true), W2 = B.fProxyTempMat(s, n, true);
            return bminres(new fProxyBSROperator(in A), in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, maxIter, tol);
        }

        /// <summary>Block-MINRES over a BSR symmetric A with default maxIter (A.M_Rows) and tol (sqrtEps).</summary>
        public static BlockSolveInfo bminres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
            => bminres(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Preconditioned block-MINRES over a BSR symmetric A. Allocates block scratch (incl. Z).</summary>
        /// <exception cref="NotSupportedException">M is not the identity preconditioner (see
        /// <see cref="bminres{TOp, TPre}"/>).</exception>
        public static BlockSolveInfo bminres<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Vprev = B.fProxyTempMat(s, n, true), Vcur = B.fProxyTempMat(s, n, true), Wk = B.fProxyTempMat(s, n, true),
                      W = B.fProxyTempMat(s, n, true), W1 = B.fProxyTempMat(s, n, true), W2 = B.fProxyTempMat(s, n, true),
                      Z = B.fProxyTempMat(s, n, true);
            return bminres(new fProxyBSROperator(in A), in M, in B, ref X, ref Vprev, ref Vcur, ref Wk, ref W, ref W1, ref W2, ref Z, maxIter, tol);
        }
    }
}
