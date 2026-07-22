using LinearAlgebra;
using LinearAlgebra.Sparse;
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

        // Each column of the bcgrq block solution matches an independent scalar cg solve of that
        // column, and every column reached tolerance.
        void MatchesScalarCgPerColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 20, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(ref arena, n, 81001u);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 81002u);

            var X = arena.fProxyMat(s, n);                      // zero initial guess
            var info = Krylov.bcgrq(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

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
        // operator's own ApplyBlock), solve with bcgrq, and recover Xk.
        void KnownSolutionRecovered()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 20, s = 5;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(ref arena, n, 82001u);
            var Xk = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 82002u);   // known solution

            var B = arena.fProxyMat(s, n);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);                 // B[j,:] = A Xk[j,:]

            var X = arena.fProxyMat(s, n);
            var info = Krylov.bcgrq(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));

            arena.Dispose();
        }

        // A rank-deficient RHS block (two identical columns) must NOT NaN or throw, must still solve
        // every column, AND -- unlike ridge bcg, which always reports minActive == rhs -- bcgrq's LQRP
        // deflation must reveal the rank loss via minActive < rhs.
        void RankDeficientBlockDeflatesAndReportsMinActive()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(ref arena, n, 83001u);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 83002u);
            // Force columns 1 and 3 identical -> block rank <= 3.
            for (int c = 0; c < n; c++) B[3, c] = B[1, c];

            var X = arena.fProxyMat(s, n);
            var info = Krylov.bcgrq(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            Assert.IsTrue(info.Solved);
            for (int c = 0; c < n; c++)
                Assert.IsTrue(math.abs((double)X[1, c] - (double)X[3, c]) <= Tol() * (1.0 + math.abs((double)X[1, c])));

            Assert.IsTrue(info.minActive < info.rhs);

            arena.Dispose();
        }

        // Block-Jacobi-preconditioned bcgrq over a BSR SPD system matches per-column scalar pcg.
        void PreconditionedMatchesScalar()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 18, s = 3;
            var Adense = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(ref arena, n, 84001u);
            var A = fProxyKrylovBatteryOracles.DenseToBSR1x1(ref arena, in Adense);
            var M = arena.fProxyBlockJacobi(in A);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 84002u);

            var X = arena.fProxyMat(s, n);
            var info = Krylov.bcgrq(in A, in M, in B, ref X, 8 * n, Consts.fProxySqrtEps);
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
        // fixed-seed system solved through bcgrq<TOp> and through bcgrq<TOp,TPre> with an EXPLICIT
        // fProxyIdentityPreconditioner (Z = default, never dereferenced) must produce the exact same X,
        // iterations and status -- no tolerance.
        void IdentityFoldBitIdentical()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(ref arena, n, 85001u);
            var B = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 85002u);
            int maxIter = 8 * n;
            fProxy tol = Consts.fProxySqrtEps;
            var op = new fProxyDenseOperator(in A);

            // Unpreconditioned rung (identity folds out at compile time).
            var Xplain = arena.fProxyMat(s, n);
            var Rp = arena.fProxyMat(s, n); var Pp = arena.fProxyMat(s, n);
            var APp = arena.fProxyMat(s, n); var Pap = arena.fProxyMat(s, n);
            var infoPlain = Krylov.bcgrq(in op, in B, ref Xplain, ref Rp, ref Pp, ref APp, ref Pap, maxIter, tol);

            // Explicit identity preconditioner through the merged core; Z = default (unused when identity).
            var Xmerged = arena.fProxyMat(s, n);
            var Rm = arena.fProxyMat(s, n); var Pm = arena.fProxyMat(s, n);
            var APm = arena.fProxyMat(s, n); var Pam = arena.fProxyMat(s, n);
            fProxyMxN Zm = default;
            var id = new fProxyIdentityPreconditioner();
            var infoMerged = Krylov.bcgrq(in op, in id, in B, ref Xmerged, ref Rm, ref Pm, ref APm, ref Pam, ref Zm, maxIter, tol);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.AreEqual((double)Xplain[j, c], (double)Xmerged[j, c]);

            Assert.AreEqual(infoPlain.iterations, infoMerged.iterations);
            Assert.AreEqual((int)infoPlain.status, (int)infoMerged.status);

            arena.Dispose();
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
            var arena = new Arena(Allocator.Persistent);

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

            var Xrq = arena.fProxyMat(s, n);
            var rqInfo = Krylov.bcgrq(in A, in B, ref Xrq, maxIter, tol);
            fProxy rqFwdErr = MaxForwardError(in Xrq, in Xk, s, n);

            Assert.IsTrue(rqInfo.maxRnorm <= ridgeInfo.maxRnorm * ResidualSlack());
            Assert.IsTrue((double)rqFwdErr <= (double)ridgeFwdErr * ResidualSlack());
            Assert.IsTrue(rqInfo.iterations <= ridgeInfo.iterations * 2 + 2);

            arena.Dispose();
        }

        // Same comparison as above, but the ill-conditioning source is the RHS block: a well-conditioned
        // SPD A with a KNOWN Xk whose column 1 is a tiny perturbation of column 0 (so B's column 1,
        // formed via ApplyBlock from the perturbed Xk, is itself numerically near-parallel to column 0
        // -- Xk stays the exact ground truth for the forward-error comparison, unlike perturbing B
        // directly, which would silently change column 1's true solution away from Xk[1,:]).
        void NearParallelRHSNeverWorseThanRidge()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 20, s = 4;
            var A = fProxyKrylovBatteryOracles.BuildDenseSpdSystem(ref arena, n, 87001u);
            var Xk = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 87002u);

            fProxy epsScale = /*+choose[1e-4f|1e-10]*/1e-4f/*-choose*/;
            var rng = new Unity.Mathematics.Random(87003u);
            for (int c = 0; c < n; c++)
                Xk[1, c] = Xk[0, c] + epsScale * rng.NextFProxy(-1f, 1f);

            var B = arena.fProxyMat(s, n);
            new fProxyDenseOperator(in A).ApplyBlock(in Xk, ref B, s);

            int maxIter = 20 * n;
            fProxy tol = Consts.fProxySqrtEps;

            var Xridge = arena.fProxyMat(s, n);
            var ridgeInfo = Krylov.bcg(in A, in B, ref Xridge, maxIter, tol);
            fProxy ridgeFwdErr = MaxForwardError(in Xridge, in Xk, s, n);

            var Xrq = arena.fProxyMat(s, n);
            var rqInfo = Krylov.bcgrq(in A, in B, ref Xrq, maxIter, tol);
            fProxy rqFwdErr = MaxForwardError(in Xrq, in Xk, s, n);

            Assert.IsTrue(rqInfo.maxRnorm <= ridgeInfo.maxRnorm * ResidualSlack());
            Assert.IsTrue((double)rqFwdErr <= (double)ridgeFwdErr * ResidualSlack());
            Assert.IsTrue(rqInfo.iterations <= ridgeInfo.iterations * 2 + 2);

            arena.Dispose();
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
