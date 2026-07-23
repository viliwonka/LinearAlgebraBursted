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

        // A = M^T M with M's columns geometrically scaled across [1, condSpan] -- stretches A's singular
        // spectrum (cond(A) ~ condSpan^2) without a Hilbert matrix's extreme growth.
        static fProxyMxN BuildStretchedSPD(int dim, uint seed, fProxy condSpan)
        {
            var M = GenerateOP.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
            for (int j = 0; j < dim; j++)
            {
                fProxy t = dim > 1 ? (fProxy)j / (fProxy)(dim - 1) : (fProxy)0;
                fProxy scale = math.pow(condSpan, t);
                for (int i = 0; i < dim; i++) M[i, j] *= scale;
            }
            var MtM = new fProxyMxN(dim, dim, Allocator.Temp);
            Blas.dot(in M, in M, ref MtM, true);                // M^T M, SPD
            return MtM;
        }

        static fProxyN Row(in fProxyMxN B, int j, int n)
        {
            var v = new fProxyN(n, Allocator.Temp);
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
            int n = 20, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 88001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88002u);

            var X = new fProxyMxN(s, n, Allocator.Temp);        // zero initial guess
            var info = Krylov.bfbcg(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.AreEqual(s, info.rhs);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                Assert.IsTrue(Krylov.cg(in A, in bj, ref xj, 8 * n, Consts.fProxySqrtEps));

                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }
        }

        // Independent of the scalar solver: pick a KNOWN block solution Xk, form B = A Xk (via the
        // operator's own ApplyBlock), solve with bfbcg, and recover Xk.
        void KnownSolutionRecovered()
        {
            int n = 20, s = 5;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 88011u);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88012u);   // known solution

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);                 // B[j,:] = A Xk[j,:]

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bfbcg(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));
        }

        // The block solve converges in <= the worst single-column scalar cg iteration count over the
        // same budget/tol (the block advantage: all RHS share the richer block Krylov subspace).
        void BlockAdvantageIterations()
        {
            int n = 24, s = 5;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 88021u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88022u);
            fProxy tol = Consts.fProxySqrtEps;
            int budget = 8 * n;

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var blockInfo = Krylov.bfbcg(in A, in B, ref X, budget, tol);
            Assert.IsTrue(blockInfo.Solved);

            int worstScalar = 0;
            for (int j = 0; j < s; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                var si = Krylov.cg(in A, in bj, ref xj, budget, tol);
                Assert.IsTrue(si.Solved);
                if (si.iterations > worstScalar) worstScalar = si.iterations;
            }

            Assert.IsTrue(blockInfo.iterations <= worstScalar);
        }

        // THE key oracle: a rank-1-in-the-solution RHS block. Perturb the KNOWN Xk (row 2 = 10x row 0)
        // FIRST, then derive B = A Xk -- so B[2,:] = 10 B[0,:] and the residual block is genuinely rank
        // deficient, yet Xk remains the exact ground truth. bfbcg must not NaN, must solve every column,
        // recover Xk (incl. the dependent pair), and REPORT the deflation via minActive < rhs.
        void RankDeficientDeflates()
        {
            int n = 16, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 88031u);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88032u);
            // Make the KNOWN solution's row 2 a scalar multiple of row 0 -> B[2,:] = 10 B[0,:].
            for (int c = 0; c < n; c++) Xk[2, c] = (fProxy)10 * Xk[0, c];

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bfbcg(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));

            Assert.IsTrue(info.minActive < info.rhs);
        }

        // Block-Jacobi-preconditioned bfbcg over a BSR SPD system matches per-column scalar pcg.
        void PreconditionedMatchesScalar()
        {
            int n = 18, s = 3;
            var Adense = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 88041u);
            var A = fProxyKrylovBatteryOracles.DenseToBSR1x1(in Adense);
            var M = new fProxyBlockJacobi(in A, Allocator.Temp);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88042u);

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bfbcg(in A, in M, in B, ref X, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                Assert.IsTrue(Krylov.cg(in A, in M, in bj, ref xj, 8 * n, Consts.fProxySqrtEps));
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }
        }

        // The identity preconditioner fold must be bit-identical to the unpreconditioned rung: same
        // fixed-seed system solved through bfbcg<TOp> and through bfbcg<TOp,TPre> with an EXPLICIT
        // fProxyIdentityPreconditioner (Z = default, never dereferenced) must produce the exact same X,
        // iterations and status -- no tolerance.
        void IdentityFoldMatchesUnpreconditioned()
        {
            int n = 16, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 88051u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 88052u);
            int maxIter = 8 * n;
            fProxy tol = Consts.fProxySqrtEps;
            var op = new fProxyDenseOperator(in A);

            // Unpreconditioned rung (identity folds out at compile time).
            var Xplain = new fProxyMxN(s, n, Allocator.Temp);
            var Rp = new fProxyMxN(s, n, Allocator.Temp); var Pp = new fProxyMxN(s, n, Allocator.Temp);
            var APp = new fProxyMxN(s, n, Allocator.Temp); var Pap = new fProxyMxN(s, n, Allocator.Temp);
            var infoPlain = Krylov.bfbcg(in op, in B, ref Xplain, ref Rp, ref Pp, ref APp, ref Pap, maxIter, tol);

            // Explicit identity preconditioner through the merged core; Z = default (unused when identity).
            var Xmerged = new fProxyMxN(s, n, Allocator.Temp);
            var Rm = new fProxyMxN(s, n, Allocator.Temp); var Pm = new fProxyMxN(s, n, Allocator.Temp);
            var APm = new fProxyMxN(s, n, Allocator.Temp); var Pam = new fProxyMxN(s, n, Allocator.Temp);
            fProxyMxN Zm = default;
            var id = new fProxyIdentityPreconditioner();
            var infoMerged = Krylov.bfbcg(in op, in id, in B, ref Xmerged, ref Rm, ref Pm, ref APm, ref Pam, ref Zm, maxIter, tol);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.AreEqual((double)Xplain[j, c], (double)Xmerged[j, c]);

            Assert.AreEqual(infoPlain.iterations, infoMerged.iterations);
            Assert.AreEqual((int)infoPlain.status, (int)infoMerged.status);
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
            // Seeds shared with bcgrq's analogous IllConditionedSPDNeverWorseThanRidge: a proven-benign
            // convergence pattern for this exact BuildStretchedSPD(condSpan=8, n=20, s=4) + row-locking
            // construction, where all columns converge together (no early per-column lock that would let
            // non-locking ridge bcg refine its easy columns far past their threshold and skew maxRnorm).
            int n = 20, s = 4;
            fProxy condSpan = (fProxy)8;
            var A = BuildStretchedSPD(n, 86001u, condSpan);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 86002u);

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);

            int maxIter = 3000;
            fProxy tol = Consts.fProxySqrtEps;

            var Xridge = new fProxyMxN(s, n, Allocator.Temp);
            var ridgeInfo = Krylov.bcg(in A, in B, ref Xridge, maxIter, tol);
            fProxy ridgeFwdErr = MaxForwardError(in Xridge, in Xk, s, n);

            var Xbf = new fProxyMxN(s, n, Allocator.Temp);
            var bfInfo = Krylov.bfbcg(in A, in B, ref Xbf, maxIter, tol);
            fProxy bfFwdErr = MaxForwardError(in Xbf, in Xk, s, n);

            Assert.IsTrue(bfInfo.maxRnorm <= ridgeInfo.maxRnorm * ResidualSlack());
            Assert.IsTrue((double)bfFwdErr <= (double)ridgeFwdErr * ResidualSlack());
            Assert.IsTrue(bfInfo.iterations <= ridgeInfo.iterations * 2 + 2);
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
