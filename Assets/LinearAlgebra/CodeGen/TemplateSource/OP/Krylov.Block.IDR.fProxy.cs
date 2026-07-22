using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ---- block IDR(s) core (bidr) ---------------------------------------------------------------

        /// <summary>
        /// True block IDR(s) (Du, Sogabe, Yu, Yamamoto, Zhang 2011) for a general (nonsymmetric) square
        /// system A X = B with m simultaneous right-hand sides, generic over BOTH the operator (<see
        /// cref="IfProxyLinearOperator"/>) and the preconditioner (<see cref="IfProxyPreconditioner"/>).
        /// Block-generalizes <see cref="idr{TOp, TPre}(in TOp, in TPre, in fProxyN, ref fProxyN, int,
        /// int, fProxy, uint)"/>'s own recurrence (fixed shadow space P, G/U history buffers starting
        /// at zero, Msys starting at the identity): every scalar coefficient becomes an m x m block
        /// (m = <paramref name="B"/>.M_Rows, the RHS count), every dot product an m x m Gram block
        /// (<see cref="BlockCrossGram"/>), every divide an m x m general solve (<see
        /// cref="BlockSolveGeneral"/>, QRCP rank-revealing -- Msys's blocks are not SPD), and every
        /// scalar-times-vector combination an m x m matrix left-multiplying an m x n block (<see
        /// cref="BlockCTV"/>, C^T @ V). <paramref name="s"/> is the IDR shadow-space DEPTH (the "(s)" in
        /// IDR(s)) -- unrelated to the RHS count, which this codebase's other block solvers call "s";
        /// see OP/DEVLOG.md "Krylov.Block.IDR" for the full derivation and naming rationale. The s x m*n
        /// shadow space P is generated deterministically from <paramref name="seed"/> (same seed =>
        /// bit-identical X), exactly mirroring scalar <c>idr</c>.
        ///
        /// B and X are m ROWS x n COLS (row j = the j-th RHS/solution, length n = A.Rows); X is
        /// warm-startable. Convergence is per column against tol^2*||B[j]||^2 (no column locking/
        /// deflation -- the paper explicitly leaves that to future work, so every live column is
        /// updated every step, like <see cref="bbiCGStab{TOp, TPre}"/>). Owns its whole workspace via
        /// Allocator.Temp (no external scratch params, unlike bcg/bcgrq). Breakdown (a singular m x m
        /// block solve, or a non-positive/NaN end-of-sweep omega) reports <see
        /// cref="IterativeSolveStatus.Breakdown"/> with X holding the last committed iterate -- never
        /// NaN, never throws from the recurrence itself. Returns a <see cref="BlockSolveInfo"/> whose
        /// <see cref="BlockSolveInfo.minActive"/> is always m (no deflation).
        /// </summary>
        public static BlockSolveInfo bidr<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("bidr: A must be square");
            int n = A.Rows;
            int m = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("bidr: B.N_Cols must equal A.Rows");
            if (X.M_Rows != m || X.N_Cols != n) throw new ArgumentException("bidr: X must match B");
            if (s < 1) throw new ArgumentException("bidr: s must be >= 1");
            if (maxIter < 1) throw new ArgumentException("bidr: maxIter must be >= 1");

            fProxy bb = BlockFrobDot(in B, in B);
            if (bb == (fProxy)0)
            {
                CopyBlock(in B, ref X, m, n);
                return new BlockSolveInfo { rhs = m, converged = m, iterations = 0, maxRnorm = 0.0, minActive = m, status = IterativeSolveStatus.Converged };
            }

            // Per-column thresholds tol^2 ||B[j]||^2.
            var thr = new fProxyN(m);
            BuildColumnThresholdsPlain(in B, ref thr, m, n, tol);

            // Shadow space P (s slots, each m x n, deterministic from seed); history G/U (s slots, each
            // m x n, start at zero); Msys (s x s grid of m x m blocks, diagonal blocks start at
            // identity) / f / c (s slots of m x m each) -- the block generalization of scalar idr's
            // P/G/U/Msys/f/c. Only Msys's lower triangle (i >= j) is ever read or written, same as
            // scalar idr's own s x s Msys.
            var P = new UnsafeList<fProxyMxN>(s, Allocator.Temp);
            var G = new UnsafeList<fProxyMxN>(s, Allocator.Temp);
            var U = new UnsafeList<fProxyMxN>(s, Allocator.Temp);
            for (int i = 0; i < s; i++)
            {
                P.Add(new fProxyMxN(m, n, Allocator.Temp, true));
                G.Add(new fProxyMxN(m, n, Allocator.Temp, false));
                U.Add(new fProxyMxN(m, n, Allocator.Temp, false));
            }

            var rng = new Unity.Mathematics.Random(seed == 0 ? 0x9E3779B1u : seed);
            for (int i = 0; i < s; i++)
            {
                var Pi = P[i];
                for (int row = 0; row < m; row++)
                    for (int col = 0; col < n; col++) Pi[row, col] = rng.NextFProxy();
            }

            var Msys = new UnsafeList<fProxyMxN>(s * s, Allocator.Temp);
            for (int i = 0; i < s * s; i++) Msys.Add(new fProxyMxN(m, m, Allocator.Temp, false));
            for (int i = 0; i < s; i++)
            {
                var Mii = Msys[i * s + i];
                for (int a = 0; a < m; a++) Mii[a, a] = (fProxy)1;
            }

            var f = new UnsafeList<fProxyMxN>(s, Allocator.Temp);
            var c = new UnsafeList<fProxyMxN>(s, Allocator.Temp);
            for (int i = 0; i < s; i++) { f.Add(new fProxyMxN(m, m, Allocator.Temp, true)); c.Add(new fProxyMxN(m, m, Allocator.Temp, true)); }

            var R = new fProxyMxN(m, n, Allocator.Temp, true);
            var V = new fProxyMxN(m, n, Allocator.Temp, true);
            var Q = new fProxyMxN(m, n, Allocator.Temp, true);
            var termMN = new fProxyMxN(m, n, Allocator.Temp, true);
            fProxyMxN VHat = default;
            fProxyN rIn = default, rOut = default;
            if (!M.IsIdentity)
            {
                VHat = new fProxyMxN(m, n, Allocator.Temp, true);
                rIn = new fProxyN(n);
                rOut = new fProxyN(n);
            }

            // Small m x m scratch: blkMM (general product accumulator), sumBlk (forward-substitution
            // accumulator), alpha/beta (bi-orthogonalisation / final-coefficient solve outputs), plus
            // BlockSolveGeneral's own QRCP scratch (shared across every m x m solve this call makes --
            // each solve fully overwrites it, never live across two solves at once).
            var blkMM  = new fProxyMxN(m, m, Allocator.Temp, true);
            var sumBlk = new fProxyMxN(m, m, Allocator.Temp, true);
            var alpha  = new fProxyMxN(m, m, Allocator.Temp, true);
            var beta   = new fProxyMxN(m, m, Allocator.Temp, true);

            var coefWork = new fProxyMxN(m, m, Allocator.Temp, true);
            var rhsWork  = new fProxyMxN(m, m, Allocator.Temp, true);
            var Rqrcp    = new fProxyMxN(m, m, Allocator.Temp, true);
            var Pqrcp    = new Pivot(m, Allocator.Temp);
            var uQrcp    = new fProxyN(m);

            // R = B - A X.
            BlockResidual(in A, in X, in B, ref R, m, n);

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int iter = 0;
            double maxr = 0;
            int converged = CountConverged(in R, in thr, m, n, out maxr);
            if (converged == m) { status = IterativeSolveStatus.Converged; goto cleanup; }

            fProxy om = (fProxy)1;

            while (iter < maxIter)
            {
                for (int i = 0; i < s; i++)
                {
                    var fi = f[i];
                    BlockCrossGram(P[i], in R, ref fi);
                }

                for (int k = 0; k < s && iter < maxIter; k++)
                {
                    var Gk = G[k];
                    var Uk = U[k];

                    // Block forward-substitute LowerTriangular(Msys[k..s-1,k..s-1]) c[k..s-1] = f[k..s-1].
                    for (int i = k; i < s; i++)
                    {
                        CopyBlock(f[i], ref sumBlk, m, m);
                        for (int j = k; j < i; j++)
                        {
                            Blas.dot(Msys[i * s + j], c[j], ref blkMM, false, false);
                            BlockAdd(ref sumBlk, in blkMM, (fProxy)(-1));
                        }
                        var ci = c[i];
                        var rankC = BlockSolveGeneral(Msys[i * s + i], in sumBlk, ref ci, ref coefWork, ref rhsWork, ref Rqrcp, ref Pqrcp, ref uQrcp, m);
                        if (rankC.status != DirectSolveStatus.Success)
                        { status = IterativeSolveStatus.Breakdown; goto cleanup; }
                    }

                    // V = sum_{i=k}^{s-1} c[i]^T G[i] ; Q = sum_{i=k}^{s-1} c[i]^T U[i] (BlockCTV: dst = C^T V).
                    BlockCTV(c[k], G[k], ref V);
                    BlockCTV(c[k], U[k], ref Q);
                    for (int i = k + 1; i < s; i++)
                    {
                        BlockCTV(c[i], G[i], ref termMN); BlockAdd(ref V, in termMN, (fProxy)1);
                        BlockCTV(c[i], U[i], ref termMN); BlockAdd(ref Q, in termMN, (fProxy)1);
                    }

                    BlockScaleInPlace(ref V, (fProxy)(-1));
                    BlockAdd(ref V, in R, (fProxy)1);   // V = R - V

                    if (M.IsIdentity)
                    {
                        CopyBlock(in Q, ref Uk, m, n);
                        BlockAdd(ref Uk, in V, om);
                        A.ApplyBlock(in Uk, ref Gk, m);
                    }
                    else
                    {
                        BlockApplyPre(in M, in V, ref VHat, m, n, ref rIn, ref rOut);
                        CopyBlock(in Q, ref Uk, m, n);
                        BlockAdd(ref Uk, in VHat, om);
                        A.ApplyBlock(in Uk, ref Gk, m);
                    }

                    // Bi-orthogonalise Gk/Uk against the columns already refreshed this sweep.
                    for (int i = 0; i < k; i++)
                    {
                        BlockCrossGram(P[i], in Gk, ref blkMM);
                        var rankA = BlockSolveGeneral(Msys[i * s + i], in blkMM, ref alpha, ref coefWork, ref rhsWork, ref Rqrcp, ref Pqrcp, ref uQrcp, m);
                        if (rankA.status != DirectSolveStatus.Success)
                        { status = IterativeSolveStatus.Breakdown; goto cleanup; }
                        BlockCTV(in alpha, G[i], ref termMN); BlockAdd(ref Gk, in termMN, (fProxy)(-1));
                        BlockCTV(in alpha, U[i], ref termMN); BlockAdd(ref Uk, in termMN, (fProxy)(-1));
                    }

                    // New column-block of Msys = P^T G (rows above k stay untouched -- always zero, never read).
                    for (int i = k; i < s; i++)
                    {
                        var mik = Msys[i * s + k];
                        BlockCrossGram(P[i], in Gk, ref mik);
                    }

                    var betaRank = BlockSolveGeneral(Msys[k * s + k], f[k], ref beta, ref coefWork, ref rhsWork, ref Rqrcp, ref Pqrcp, ref uQrcp, m);
                    if (betaRank.status != DirectSolveStatus.Success)
                    { status = IterativeSolveStatus.Breakdown; goto cleanup; }

                    BlockCTV(in beta, in Gk, ref termMN); BlockAdd(ref R, in termMN, (fProxy)(-1));
                    BlockCTV(in beta, in Uk, ref termMN); BlockAdd(ref X, in termMN, (fProxy)1);
                    iter++;

                    converged = CountConverged(in R, in thr, m, n, out maxr);
                    if (converged == m)
                    {
                        // Verify-at-exit: the recurrence R can drift from the true B - A X (IDR is
                        // the family with the worst known tracked-vs-true drift). termMN is idle here
                        // (last read into the BlockAdd above, next write is the next k-step's BlockCTV
                        // or the end-of-sweep block) -- used only to gate the decision; the incremental
                        // f[] history this sweep tracks off R must NOT be perturbed, so a failed check
                        // discards the fresh residual and leaves R untouched.
                        BlockResidual(in A, in X, in B, ref termMN, m, n);
                        int freshConverged = CountConverged(in termMN, in thr, m, n, out double freshMaxr);
                        if (freshConverged == m)
                        {
                            converged = freshConverged; maxr = freshMaxr;
                            status = IterativeSolveStatus.Converged; goto cleanup;
                        }
                    }

                    if (k < s - 1)
                        for (int i = k + 1; i < s; i++)
                        {
                            Blas.dot(Msys[i * s + k], in beta, ref blkMM, false, false);
                            var fi = f[i];
                            BlockAdd(ref fi, in blkMM, (fProxy)(-1));
                        }
                }

                if (iter >= maxIter) break;

                // End-of-sweep step: R is already orthogonal to P, so v = r; refine one level deeper.
                CopyBlock(in R, ref V, m, n);
                if (M.IsIdentity)
                {
                    A.ApplyBlock(in V, ref Q, m);
                }
                else
                {
                    BlockApplyPre(in M, in V, ref VHat, m, n, ref rIn, ref rOut);
                    A.ApplyBlock(in VHat, ref Q, m);
                }

                fProxy nt2 = BlockFrobDot(in Q, in Q);
                if (!(nt2 > (fProxy)0) || math.isnan(nt2))
                { status = IterativeSolveStatus.Breakdown; goto cleanup; }

                fProxy ts = BlockFrobDot(in Q, in R);
                fProxy ns2 = BlockFrobDot(in R, in R);
                fProxy nt = math.sqrt(nt2);
                fProxy ns = math.sqrt(ns2);
                fProxy rho = math.abs(ts / (nt * ns));
                om = ts / nt2;
                if (rho > (fProxy)0 && rho < (fProxy)0.7) om = om * (fProxy)0.7 / rho;

                if (om == (fProxy)0 || math.isnan(om))
                { status = IterativeSolveStatus.Breakdown; goto cleanup; }

                BlockAdd(ref R, in Q, -om);
                if (M.IsIdentity) BlockAdd(ref X, in V, om);
                else              BlockAdd(ref X, in VHat, om);
                iter++;

                converged = CountConverged(in R, in thr, m, n, out maxr);
                if (converged == m)
                {
                    // Verify-at-exit (same rationale as the in-sweep check above). termMN is idle
                    // here (last touched inside the k-loop above, not read again this sweep).
                    BlockResidual(in A, in X, in B, ref termMN, m, n);
                    int freshConverged = CountConverged(in termMN, in thr, m, n, out double freshMaxr);
                    if (freshConverged == m)
                    {
                        converged = freshConverged; maxr = freshMaxr;
                        status = IterativeSolveStatus.Converged; goto cleanup;
                    }
                }
            }

        cleanup:
            for (int i = 0; i < s; i++) { P[i].Dispose(); G[i].Dispose(); U[i].Dispose(); f[i].Dispose(); c[i].Dispose(); }
            P.Dispose(); G.Dispose(); U.Dispose(); f.Dispose(); c.Dispose();
            for (int i = 0; i < s * s; i++) Msys[i].Dispose();
            Msys.Dispose();
            R.Dispose(); V.Dispose(); Q.Dispose(); termMN.Dispose();
            if (!M.IsIdentity) { VHat.Dispose(); rIn.Dispose(); rOut.Dispose(); }
            blkMM.Dispose(); sumBlk.Dispose(); alpha.Dispose(); beta.Dispose();
            coefWork.Dispose(); rhsWork.Dispose(); Rqrcp.Dispose(); Pqrcp.Dispose(); uQrcp.Dispose();
            thr.Dispose();

            return new BlockSolveInfo { rhs = m, converged = converged, iterations = iter, maxRnorm = maxr, minActive = m, status = status };
        }

        // ---- unpreconditioned + concrete forwarders ------------------------------------------------

        /// <summary>Unpreconditioned block IDR(s) -- forwards into the merged
        /// <see cref="bidr{TOp, TPre}"/> with the identity preconditioner.</summary>
        public static BlockSolveInfo bidr<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                        int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            where TOp : struct, IfProxyLinearOperator
        {
            return bidr(in A, default(fProxyIdentityPreconditioner), in B, ref X, s, maxIter, tol, seed);
        }

        /// <summary>Block IDR(s) over a dense NON-symmetric <see cref="fProxyMxN"/> A, via
        /// <see cref="fProxyDenseOperatorGeneral"/> (general block apply -- <see cref="fProxyDenseOperator"/>'s
        /// ApplyBlock is symmetric-only and would silently solve A^Tx=b here).</summary>
        public static BlockSolveInfo bidr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                        int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            => bidr(new fProxyDenseOperatorGeneral(in A), in B, ref X, s, maxIter, tol, seed);

        /// <summary>Block IDR(s) over a dense non-symmetric A with defaults (s = 4, maxIter = A.M_Rows,
        /// tol = Consts.fProxySqrtEps, seed = default).</summary>
        public static BlockSolveInfo bidr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => bidr(new fProxyDenseOperatorGeneral(in A), in B, ref X, 4, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Block IDR(s) over a block-sparse (BSR) non-symmetric A. Allocates its whole
        /// workspace from Allocator.Temp.</summary>
        public static BlockSolveInfo bidr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X,
                                        int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            => bidr(new fProxyBSROperator(in A), in B, ref X, s, maxIter, tol, seed);

        /// <summary>Block IDR(s) over a BSR non-symmetric A with defaults (s = 4, maxIter = A.M_Rows,
        /// tol = Consts.fProxySqrtEps, seed = default).</summary>
        public static BlockSolveInfo bidr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
            => bidr(new fProxyBSROperator(in A), in B, ref X, 4, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Right-preconditioned block IDR(s) over a BSR non-symmetric A with ANY
        /// <see cref="IfProxyPreconditioner"/> (ILU0/block-Jacobi).</summary>
        public static BlockSolveInfo bidr<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                        int s, int maxIter, fProxy tol, uint seed = 0x9E3779B1u)
            where TPre : struct, IfProxyPreconditioner
            => bidr(new fProxyBSROperator(in A), in M, in B, ref X, s, maxIter, tol, seed);

        /// <summary>Right-preconditioned block IDR(s) over BSR with ANY <see cref="IfProxyPreconditioner"/>
        /// (ILU0/block-Jacobi), with defaults (s = 4, maxIter = A.M_Rows, tol = Consts.fProxySqrtEps,
        /// seed = default).</summary>
        public static BlockSolveInfo bidr<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TPre : struct, IfProxyPreconditioner
            => bidr(new fProxyBSROperator(in A), in M, in B, ref X, 4, A.M_Rows, Consts.fProxySqrtEps);
    }
}
