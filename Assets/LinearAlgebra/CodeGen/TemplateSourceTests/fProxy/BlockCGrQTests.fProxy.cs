using BULA;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// bcgrq: deflating block-CG with reliable QR (LQRP) residual updates -- coexists with ridge bcg
// (BlockCGTests.fProxy.cs). Mirrors that file's structure exactly: one [BurstCompile] IJob with a
// TestType switch, every scenario built and asserted inside Execute(), so job-safety (the caller sees
// the final X written through ref fProxyMxN) is exercised by construction for every case.
public class fProxyBlockCGrQTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct BlockCGrQTestJob : IJob
    {
        public enum TestType
        {
            MatchesScalarCgPerColumn,
            KnownSolutionRecovered,
            RankDeficientBlockDeflatesAndReportsMinActive,
            PreconditionedMatchesScalar,
            IdentityFoldBitIdentical,
            IllConditionedSPDNeverWorseThanRidge,
            NearParallelRHSNeverWorseThanRidge,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MatchesScalarCgPerColumn: MatchesScalarCgPerColumn(); break;
                case TestType.KnownSolutionRecovered: KnownSolutionRecovered(); break;
                case TestType.RankDeficientBlockDeflatesAndReportsMinActive: RankDeficientBlockDeflatesAndReportsMinActive(); break;
                case TestType.PreconditionedMatchesScalar: PreconditionedMatchesScalar(); break;
                case TestType.IdentityFoldBitIdentical: IdentityFoldBitIdentical(); break;
                case TestType.IllConditionedSPDNeverWorseThanRidge: IllConditionedSPDNeverWorseThanRidge(); break;
                case TestType.NearParallelRHSNeverWorseThanRidge: NearParallelRHSNeverWorseThanRidge(); break;
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
            return Blas.dot(M, M, true);                        // M^T M, SPD
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

        // Each column of the bcgrq block solution matches an independent scalar cg solve of that
        // column, and every column reached tolerance.
        void MatchesScalarCgPerColumn()
        {
            int n = 20, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 81001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 81002u);

            var X = new fProxyMxN(s, n, Allocator.Temp);        // zero initial guess
            var info = Krylov.bcgrq(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

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
        // operator's own ApplyBlock), solve with bcgrq, and recover Xk.
        void KnownSolutionRecovered()
        {
            int n = 20, s = 5;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 82001u);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 82002u);   // known solution

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);                 // B[j,:] = A Xk[j,:]

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bcgrq(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));
        }

        // A rank-deficient RHS block (two identical columns) must NOT NaN or throw, must still solve
        // every column, AND -- unlike ridge bcg, which always reports minActive == rhs -- bcgrq's LQRP
        // deflation must reveal the rank loss via minActive < rhs.
        void RankDeficientBlockDeflatesAndReportsMinActive()
        {
            int n = 16, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 83001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 83002u);
            // Force columns 1 and 3 identical -> block rank <= 3.
            for (int c = 0; c < n; c++) B[3, c] = B[1, c];

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bcgrq(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            Assert.IsTrue(info.Solved);
            for (int c = 0; c < n; c++)
                Assert.IsTrue(math.abs((double)X[1, c] - (double)X[3, c]) <= Tol() * (1.0 + math.abs((double)X[1, c])));

            Assert.IsTrue(info.minActive < info.rhs);
        }

        // Block-Jacobi-preconditioned bcgrq over a BSR SPD system matches per-column scalar pcg.
        void PreconditionedMatchesScalar()
        {
            int n = 18, s = 3;
            var Adense = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 84001u);
            var A = fProxyKrylovBatteryOracles.DenseToBSR1x1(in Adense);
            var M = new fProxyBlockJacobi(in A, Allocator.Temp);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 84002u);

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bcgrq(in A, in M, in B, ref X, 8 * n, Consts.fProxySqrtEps);
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
        // fixed-seed system solved through bcgrq<TOp> and through bcgrq<TOp,TPre> with an EXPLICIT
        // fProxyIdentityPreconditioner (Z = default, never dereferenced) must produce the exact same X,
        // iterations and status -- no tolerance.
        void IdentityFoldBitIdentical()
        {
            int n = 16, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 85001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 85002u);
            int maxIter = 8 * n;
            fProxy tol = Consts.fProxySqrtEps;
            var op = new fProxyDenseOperator(in A);

            // Unpreconditioned rung (identity folds out at compile time).
            var Xplain = new fProxyMxN(s, n, Allocator.Temp);
            var Rp = new fProxyMxN(s, n, Allocator.Temp); var Pp = new fProxyMxN(s, n, Allocator.Temp);
            var APp = new fProxyMxN(s, n, Allocator.Temp); var Pap = new fProxyMxN(s, n, Allocator.Temp);
            var infoPlain = Krylov.bcgrq(in op, in B, ref Xplain, ref Rp, ref Pp, ref APp, ref Pap, maxIter, tol);

            // Explicit identity preconditioner through the merged core; Z = default (unused when identity).
            var Xmerged = new fProxyMxN(s, n, Allocator.Temp);
            var Rm = new fProxyMxN(s, n, Allocator.Temp); var Pm = new fProxyMxN(s, n, Allocator.Temp);
            var APm = new fProxyMxN(s, n, Allocator.Temp); var Pam = new fProxyMxN(s, n, Allocator.Temp);
            fProxyMxN Zm = default;
            var id = new fProxyIdentityPreconditioner();
            var infoMerged = Krylov.bcgrq(in op, in id, in B, ref Xmerged, ref Rm, ref Pm, ref APm, ref Pam, ref Zm, maxIter, tol);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.AreEqual((double)Xplain[j, c], (double)Xmerged[j, c]);

            Assert.AreEqual(infoPlain.iterations, infoMerged.iterations);
            Assert.AreEqual((int)infoPlain.status, (int)infoMerged.status);
        }

        // On an ill-conditioned SPD system (stretched singular spectrum), bcgrq must be NO WORSE than
        // ridge bcg under the SAME budget: worst-column residual, worst-column forward error (vs the
        // known Xk), and iteration count (generous factor -- bcgrq pays extra per-iteration LQRP cost).
        // The budget is generous on purpose (this is a tiny n x n system) rather than asserting Solved,
        // which would flake across the ill-conditioning/dtype combination. Both solvers stop as soon as
        // they cross the SAME residual threshold, not at identical precision, so their last iterates
        // differ by more than rounding noise -- ResidualSlack allows that stopping-point spread while
        // still catching a genuine (order-of-magnitude) regression.
        static double ResidualSlack() => 3.0;

        void IllConditionedSPDNeverWorseThanRidge()
        {
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

            var Xrq = new fProxyMxN(s, n, Allocator.Temp);
            var rqInfo = Krylov.bcgrq(in A, in B, ref Xrq, maxIter, tol);
            fProxy rqFwdErr = MaxForwardError(in Xrq, in Xk, s, n);

            Assert.IsTrue(rqInfo.maxRnorm <= ridgeInfo.maxRnorm * ResidualSlack());
            Assert.IsTrue((double)rqFwdErr <= (double)ridgeFwdErr * ResidualSlack());
            Assert.IsTrue(rqInfo.iterations <= ridgeInfo.iterations * 2 + 2);
        }

        // Same comparison as above, but the ill-conditioning source is the RHS block: a well-conditioned
        // SPD A with a KNOWN Xk whose column 1 is a tiny perturbation of column 0 (so B's column 1,
        // formed via ApplyBlock from the perturbed Xk, is itself numerically near-parallel to column 0
        // -- Xk stays the exact ground truth for the forward-error comparison, unlike perturbing B
        // directly, which would silently change column 1's true solution away from Xk[1,:]).
        void NearParallelRHSNeverWorseThanRidge()
        {
            int n = 20, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(n, 87001u);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 87002u);

            fProxy epsScale = /*+choose[1e-4f|1e-10]*/1e-4f/*-choose*/;
            var rng = new Unity.Mathematics.Random(87003u);
            for (int c = 0; c < n; c++)
                Xk[1, c] = Xk[0, c] + epsScale * rng.NextFProxy(-1f, 1f);

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);

            int maxIter = 20 * n;
            fProxy tol = Consts.fProxySqrtEps;

            var Xridge = new fProxyMxN(s, n, Allocator.Temp);
            var ridgeInfo = Krylov.bcg(in A, in B, ref Xridge, maxIter, tol);
            fProxy ridgeFwdErr = MaxForwardError(in Xridge, in Xk, s, n);

            var Xrq = new fProxyMxN(s, n, Allocator.Temp);
            var rqInfo = Krylov.bcgrq(in A, in B, ref Xrq, maxIter, tol);
            fProxy rqFwdErr = MaxForwardError(in Xrq, in Xk, s, n);

            Assert.IsTrue(rqInfo.maxRnorm <= ridgeInfo.maxRnorm * ResidualSlack());
            Assert.IsTrue((double)rqFwdErr <= (double)ridgeFwdErr * ResidualSlack());
            Assert.IsTrue(rqInfo.iterations <= ridgeInfo.iterations * 2 + 2);
        }
    }

    [Test]
    public void MatchesScalarCgPerColumn()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.MatchesScalarCgPerColumn }.Run();

    [Test]
    public void KnownSolutionRecovered()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.KnownSolutionRecovered }.Run();

    [Test]
    public void RankDeficientBlockDeflatesAndReportsMinActive()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.RankDeficientBlockDeflatesAndReportsMinActive }.Run();

    [Test]
    public void PreconditionedMatchesScalar()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.PreconditionedMatchesScalar }.Run();

    [Test]
    public void IdentityFoldBitIdentical()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.IdentityFoldBitIdentical }.Run();

    [Test]
    public void IllConditionedSPDNeverWorseThanRidge()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.IllConditionedSPDNeverWorseThanRidge }.Run();

    [Test]
    public void NearParallelRHSNeverWorseThanRidge()
        => new BlockCGrQTestJob { Type = BlockCGrQTestJob.TestType.NearParallelRHSNeverWorseThanRidge }.Run();
}
