using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// bfbcg: breakdown-free block CG (Ji & Li 2017) -- orthonormalizes the SEARCH block P each iteration
// (rank-revealing LQ), so it never breaks down on rank-deficient / near-parallel RHS blocks. Mirrors
// BlockCGrQTests.fProxy.cs structure exactly: one [BurstCompile] IJob with a TestType switch, every
// scenario built and asserted inside Execute(), so job-safety (the caller sees the final X written
// through ref fProxyMxN) is exercised by construction for every case.
public class fProxyBlockBFBCGTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct BlockBFBCGTestJob : IJob
    {
        public enum TestType
        {
            MatchesScalarCgPerColumn,
            KnownSolutionRecovered,
            BlockAdvantageIterations,
            RankDeficientDeflates,
            PreconditionedMatchesScalar,
            IdentityFoldMatchesUnpreconditioned,
            NeverWorseThanRidge,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MatchesScalarCgPerColumn: MatchesScalarCgPerColumn(); break;
                case TestType.KnownSolutionRecovered: KnownSolutionRecovered(); break;
                case TestType.BlockAdvantageIterations: BlockAdvantageIterations(); break;
                case TestType.RankDeficientDeflates: RankDeficientDeflates(); break;
                case TestType.PreconditionedMatchesScalar: PreconditionedMatchesScalar(); break;
                case TestType.IdentityFoldMatchesUnpreconditioned: IdentityFoldMatchesUnpreconditioned(); break;
                case TestType.NeverWorseThanRidge: NeverWorseThanRidge(); break;
            }
        }

        static fProxy Tol() => /*+choose[2e-2f|1e-5]*/2e-2f/*-choose*/;

        static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
            var A = Blas.dot(M, M, true);                       // M^T M
            for (int d = 0; d < dim; d++) A[d, d] += dim;       // diagonally boost -> SPD, well-conditioned
            return A;
        }

        // A = M^T M with M's columns geometrically scaled across [1, condSpan] -- stretches A's singular
        // spectrum (cond(A) ~ condSpan^2) without a Hilbert matrix's extreme growth.
        static fProxyMxN BuildStretchedSPD(ref Arena arena, int dim, uint seed, fProxy condSpan)
        {
            var M = arena.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
            for (int j = 0; j < dim; j++)
            {
                fProxy t = dim > 1 ? (fProxy)j / (fProxy)(dim - 1) : (fProxy)0;
                fProxy scale = math.pow(condSpan, t);
                for (int i = 0; i < dim; i++) M[i, j] *= scale;
            }
            return Blas.dot(M, M, true);                        // M^T M, SPD
        }

        static fProxyN Row(ref Arena arena, in fProxyMxN B, int j, int n)
        {
            var v = arena.fProxyVec(n);
            for (int c = 0; c < n; c++) v[c] = B[j, c];
            return v;
        }

        // max_j ||X[j,:] - Xk[j,:]||.
        static fProxy MaxForwardError(in fProxyMxN X, in fProxyMxN Xk, int s, int n)
        {
            fProxy worst = (fProxy)0;
            for (int j = 0; j < s; j++)
            {
                fProxy e2 = (fProxy)0;
                for (int c = 0; c < n; c++)
                {
                    fProxy d = X[j, c] - Xk[j, c];
                    e2 += d * d;
                }
                fProxy e = math.sqrt(e2);
                if (e > worst) worst = e;
            }
            return worst;
        }

        // Each column of the bfbcg block solution matches an independent scalar cg solve of that
        // column, and every column reached tolerance.
        void MatchesScalarCgPerColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 20, s = 4;
            var A = BuildDenseSPD(ref arena, n, 88001u);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88002u);

            var X = arena.fProxyMat(s, n);                      // zero initial guess
            var info = Krylov.bfbcg(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.AreEqual(s, info.rhs);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(ref arena, in B, j, n);
                var xj = arena.fProxyVec(n);
                Assert.IsTrue(Krylov.cg(in A, in bj, ref xj, 8 * n, Consts.fProxySqrtEps));

                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }

            arena.Dispose();
        }

        // Independent of the scalar solver: pick a KNOWN block solution Xk, form B = A Xk (via the
        // operator's own ApplyBlock), solve with bfbcg, and recover Xk.
        void KnownSolutionRecovered()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 20, s = 5;
            var A = BuildDenseSPD(ref arena, n, 88011u);
            var Xk = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88012u);   // known solution

            var B = arena.fProxyMat(s, n);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);                 // B[j,:] = A Xk[j,:]

            var X = arena.fProxyMat(s, n);
            var info = Krylov.bfbcg(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));

            arena.Dispose();
        }

        // The block solve converges in <= the worst single-column scalar cg iteration count over the
        // same budget/tol (the block advantage: all RHS share the richer block Krylov subspace).
        void BlockAdvantageIterations()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 24, s = 5;
            var A = BuildDenseSPD(ref arena, n, 88021u);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88022u);
            fProxy tol = Consts.fProxySqrtEps;
            int budget = 8 * n;

            var X = arena.fProxyMat(s, n);
            var blockInfo = Krylov.bfbcg(in A, in B, ref X, budget, tol);
            Assert.IsTrue(blockInfo.Solved);

            int worstScalar = 0;
            for (int j = 0; j < s; j++)
            {
                var bj = Row(ref arena, in B, j, n);
                var xj = arena.fProxyVec(n);
                var si = Krylov.cg(in A, in bj, ref xj, budget, tol);
                Assert.IsTrue(si.Solved);
                if (si.iterations > worstScalar) worstScalar = si.iterations;
            }

            Assert.IsTrue(blockInfo.iterations <= worstScalar);

            arena.Dispose();
        }

        // THE key oracle: a rank-1-in-the-solution RHS block. Perturb the KNOWN Xk (row 2 = 10x row 0)
        // FIRST, then derive B = A Xk -- so B[2,:] = 10 B[0,:] and the residual block is genuinely rank
        // deficient, yet Xk remains the exact ground truth. bfbcg must not NaN, must solve every column,
        // recover Xk (incl. the dependent pair), and REPORT the deflation via minActive < rhs.
        void RankDeficientDeflates()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16, s = 4;
            var A = BuildDenseSPD(ref arena, n, 88031u);
            var Xk = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88032u);
            // Make the KNOWN solution's row 2 a scalar multiple of row 0 -> B[2,:] = 10 B[0,:].
            for (int c = 0; c < n; c++) Xk[2, c] = (fProxy)10 * Xk[0, c];

            var B = arena.fProxyMat(s, n);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);

            var X = arena.fProxyMat(s, n);
            var info = Krylov.bfbcg(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));

            Assert.IsTrue(info.minActive < info.rhs);

            arena.Dispose();
        }

        // Block-Jacobi-preconditioned bfbcg over a BSR SPD system matches per-column scalar pcg.
        void PreconditionedMatchesScalar()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 18, s = 3;
            var Adense = BuildDenseSPD(ref arena, n, 88041u);
            var A = DenseToBSR1x1(ref arena, in Adense, n * n);
            var M = arena.fProxyBlockJacobi(in A);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88042u);

            var X = arena.fProxyMat(s, n);
            var info = Krylov.bfbcg(in A, in M, in B, ref X, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(ref arena, in B, j, n);
                var xj = arena.fProxyVec(n);
                Assert.IsTrue(Krylov.cg(in A, in M, in bj, ref xj, 8 * n, Consts.fProxySqrtEps));
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }

            arena.Dispose();
        }

        // The identity preconditioner fold must be bit-identical to the unpreconditioned rung: same
        // fixed-seed system solved through bfbcg<TOp> and through bfbcg<TOp,TPre> with an EXPLICIT
        // fProxyIdentityPreconditioner (Z = default, never dereferenced) must produce the exact same X,
        // iterations and status -- no tolerance.
        void IdentityFoldMatchesUnpreconditioned()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16, s = 4;
            var A = BuildDenseSPD(ref arena, n, 88051u);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88052u);
            int maxIter = 8 * n;
            fProxy tol = Consts.fProxySqrtEps;
            var op = new fProxyDenseOperator(in A);

            // Unpreconditioned rung (identity folds out at compile time).
            var Xplain = arena.fProxyMat(s, n);
            var Rp = arena.fProxyMat(s, n); var Pp = arena.fProxyMat(s, n);
            var APp = arena.fProxyMat(s, n); var Pap = arena.fProxyMat(s, n);
            var infoPlain = Krylov.bfbcg(in op, in B, ref Xplain, ref Rp, ref Pp, ref APp, ref Pap, maxIter, tol);

            // Explicit identity preconditioner through the merged core; Z = default (unused when identity).
            var Xmerged = arena.fProxyMat(s, n);
            var Rm = arena.fProxyMat(s, n); var Pm = arena.fProxyMat(s, n);
            var APm = arena.fProxyMat(s, n); var Pam = arena.fProxyMat(s, n);
            fProxyMxN Zm = default;
            var id = new fProxyIdentityPreconditioner();
            var infoMerged = Krylov.bfbcg(in op, in id, in B, ref Xmerged, ref Rm, ref Pm, ref APm, ref Pam, ref Zm, maxIter, tol);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.AreEqual((double)Xplain[j, c], (double)Xmerged[j, c]);

            Assert.AreEqual(infoPlain.iterations, infoMerged.iterations);
            Assert.AreEqual((int)infoPlain.status, (int)infoMerged.status);

            arena.Dispose();
        }

        // On an ill-conditioned SPD system (stretched singular spectrum), bfbcg must be NO WORSE than
        // ridge bcg under the SAME budget: worst-column residual, worst-column forward error (vs the
        // known Xk), and iteration count (generous factor -- bfbcg pays extra per-iteration LQ cost).
        // Both solvers stop as soon as they cross the SAME residual threshold, not at identical
        // precision, so their last iterates differ by more than rounding noise -- ResidualSlack allows
        // that stopping-point spread while still catching a genuine (order-of-magnitude) regression.
        static double ResidualSlack() => 3.0;

        void NeverWorseThanRidge()
        {
            var arena = new Arena(Allocator.Persistent);

            // Seeds shared with bcgrq's analogous IllConditionedSPDNeverWorseThanRidge: a proven-benign
            // convergence pattern for this exact BuildStretchedSPD(condSpan=8, n=20, s=4) + row-locking
            // construction, where all columns converge together (no early per-column lock that would let
            // non-locking ridge bcg refine its easy columns far past their threshold and skew maxRnorm).
            int n = 20, s = 4;
            fProxy condSpan = (fProxy)8;
            var A = BuildStretchedSPD(ref arena, n, 86001u, condSpan);
            var Xk = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 86002u);

            var B = arena.fProxyMat(s, n);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);

            int maxIter = 3000;
            fProxy tol = Consts.fProxySqrtEps;

            var Xridge = arena.fProxyMat(s, n);
            var ridgeInfo = Krylov.bcg(in A, in B, ref Xridge, maxIter, tol);
            fProxy ridgeFwdErr = MaxForwardError(in Xridge, in Xk, s, n);

            var Xbf = arena.fProxyMat(s, n);
            var bfInfo = Krylov.bfbcg(in A, in B, ref Xbf, maxIter, tol);
            fProxy bfFwdErr = MaxForwardError(in Xbf, in Xk, s, n);

            Assert.IsTrue(bfInfo.maxRnorm <= ridgeInfo.maxRnorm * ResidualSlack());
            Assert.IsTrue((double)bfFwdErr <= (double)ridgeFwdErr * ResidualSlack());
            Assert.IsTrue(bfInfo.iterations <= ridgeInfo.iterations * 2 + 2);

            arena.Dispose();
        }

        // Dense n x n -> 1x1-block BSR (mirrors the helper used across the sparse solver tests).
        static fProxyBSR DenseToBSR1x1(ref Arena arena, in fProxyMxN A, int nnzHint)
        {
            var builder = arena.fProxyBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(ref arena);
        }
    }

    [Test]
    public void MatchesScalarCgPerColumn()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.MatchesScalarCgPerColumn }.Run();

    [Test]
    public void KnownSolutionRecovered()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.KnownSolutionRecovered }.Run();

    [Test]
    public void BlockAdvantageIterations()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.BlockAdvantageIterations }.Run();

    [Test]
    public void RankDeficientDeflates()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.RankDeficientDeflates }.Run();

    [Test]
    public void PreconditionedMatchesScalar()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.PreconditionedMatchesScalar }.Run();

    [Test]
    public void IdentityFoldMatchesUnpreconditioned()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.IdentityFoldMatchesUnpreconditioned }.Run();

    [Test]
    public void NeverWorseThanRidge()
        => new BlockBFBCGTestJob { Type = BlockBFBCGTestJob.TestType.NeverWorseThanRidge }.Run();
}
