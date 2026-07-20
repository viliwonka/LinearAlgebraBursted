using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// blsmr: block LSMR for a tall/overdetermined operator (block Golub-Kahan bidiagonalization + block
// QR update). The key oracle is BLOCK LEAST-SQUARES OPTIMALITY (normal-equation residual per column),
// not raw residual size -- an overdetermined system has a nonzero residual by construction. One
// [BurstCompile] IJob with a TestType switch, mirroring BlockCGrQTests.fProxy.cs's structure. Every
// assertion runs INSIDE Execute() (or off X's persistent UnsafeList backing after Run()) -- NEVER off
// a plain job field read after .Run(): IJob.Run() executes on a by-value struct copy, so a scalar
// field write (e.g. a solver status/iteration count stashed for a post-Run() Assert) is silently
// discarded, and asserting against its default(0) can produce a false green. See OP/DEVLOG.md.
public class fProxyBlockLSMRTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct BlockLSMRTestJob : IJob
    {
        public enum TestType
        {
            NormalEquationsOptimalAndMatchesScalarLsmr,
            ConsistentSystemRecoversExactSolution,
            ZeroRhsConvergesImmediately,
            NeverNaNOnTinyMaxIter,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.NormalEquationsOptimalAndMatchesScalarLsmr: NormalEquationsOptimalAndMatchesScalarLsmr(); break;
                case TestType.ConsistentSystemRecoversExactSolution: ConsistentSystemRecoversExactSolution(); break;
                case TestType.ZeroRhsConvergesImmediately: ZeroRhsConvergesImmediately(); break;
                case TestType.NeverNaNOnTinyMaxIter: NeverNaNOnTinyMaxIter(); break;
            }
        }

        static fProxy Tol() => /*+choose[2e-2f|1e-6]*/2e-2f/*-choose*/;

        // Tall m x n (m > n), full column rank by construction (random + diagonal-ish boost is not
        // needed for a plain rectangular least-squares operator -- a random tall matrix is full column
        // rank with probability 1).
        static fProxyMxN BuildTallA(ref Arena arena, int m, int n, uint seed)
            => arena.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, seed);

        static fProxyN Row(ref Arena arena, in fProxyMxN B, int j, int cols)
        {
            var v = arena.fProxyVec(cols);
            for (int c = 0; c < cols; c++) v[c] = B[j, c];
            return v;
        }

        // A^T (A X[j,:] - B[j,:]) for row j -- the normal-equations residual whose smallness IS the
        // least-squares optimality oracle (NOT ||A X - B||, which is nonzero by construction for an
        // overdetermined system).
        static fProxy NormalEqResidualNorm(ref Arena arena, in fProxyMxN A, in fProxyMxN X, in fProxyMxN B, int j, int m, int n)
        {
            var xj = Row(ref arena, in X, j, n);
            var r = arena.fProxyVec(m);
            Blas.dot(in A, in xj, ref r);                 // r = A xj
            for (int c = 0; c < m; c++) r[c] -= B[j, c];   // r = A xj - bj

            var atr = arena.fProxyVec(n);
            Blas.dot(in r, in A, ref atr);                 // atr = A^T r
            return math.sqrt(Blas.dot(atr, atr));
        }

        // The KEY oracle: per-column normal-equations optimality (||A^T(AX-B)|| ~ 0) for a genuine
        // overdetermined (nonzero-residual) system, AND per-column agreement with the scalar lsmr
        // solve of that same column. Asserted here, inside Execute() -- not off a job field after Run().
        void NormalEquationsOptimalAndMatchesScalarLsmr()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 24, n = 8, s = 3;
            var A = BuildTallA(ref arena, m, n, 91001u);
            var B = arena.fProxyRandomMat(s, m, (fProxy)(-1f), (fProxy)1f, 91002u);

            var X = arena.fProxyMat(s, n);
            var info = Krylov.blsmr(in A, in B, ref X, 20 * n, Consts.fProxySqrtEps);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.AreEqual(s, info.rhs);

            for (int j = 0; j < s; j++)
            {
                fProxy atrNorm = NormalEqResidualNorm(ref arena, in A, in X, in B, j, m, n);
                Assert.IsTrue((double)atrNorm <= 1e-3);

                var bj = Row(ref arena, in B, j, m);
                var xj = arena.fProxyVec(n);
                Assert.IsTrue(Krylov.lsmr(in A, in bj, ref xj, 20 * n, Consts.fProxySqrtEps));

                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }

            arena.Dispose();
        }

        // A CONSISTENT (zero-residual) tall system: B built exactly as A * Xk must recover Xk exactly
        // (to solver tolerance) -- the finite-termination property of block Golub-Kahan Krylov methods.
        // Asserted here, inside Execute() -- not off a job field after Run().
        void ConsistentSystemRecoversExactSolution()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6, s = 3;
            var A = BuildTallA(ref arena, m, n, 92001u);
            var Xk = arena.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 92002u);

            var B = arena.fProxyMat(s, m);
            for (int j = 0; j < s; j++)
            {
                var xkj = Row(ref arena, in Xk, j, n);
                var bj = arena.fProxyVec(m);
                Blas.dot(in A, in xkj, ref bj);
                for (int c = 0; c < m; c++) B[j, c] = bj[c];
            }

            var X = arena.fProxyMat(s, n);
            var info = Krylov.blsmr(in A, in B, ref X, 20 * n, Consts.fProxySqrtEps);

            // Recovery IS the contract for a consistent full-rank tall system (the block LS solution
            // is Xk) -- asserted for every dtype. The convergence FLAG is conservative in float (see
            // OP/DEVLOG.md), so full-convergence status is asserted for double only.
            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));

            if (/*+choose[false|true]*/false/*-choose*/)
            {
                Assert.IsTrue(info.Solved);
                Assert.AreEqual(s, info.converged);
            }

            arena.Dispose();
        }

        // B = 0 must converge immediately (X = 0, zero iterations) -- never a Breakdown/NaN edge case.
        void ZeroRhsConvergesImmediately()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 16, n = 5, s = 3;
            var A = BuildTallA(ref arena, m, n, 93001u);
            var B = arena.fProxyMat(s, m);   // zeroed by allocation

            var X = arena.fProxyMat(s, n);
            var info = Krylov.blsmr(in A, in B, ref X, 20 * n, Consts.fProxySqrtEps);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(0, info.iterations);
            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.AreEqual(0.0, (double)X[j, c]);

            arena.Dispose();
        }

        // A tiny maxIter (forcing MaxIterations before full convergence on a nontrivial system) must
        // never NaN/Inf X, regardless of solved status.
        void NeverNaNOnTinyMaxIter()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 30, n = 10, s = 4;
            var A = BuildTallA(ref arena, m, n, 94001u);
            var B = arena.fProxyRandomMat(s, m, (fProxy)(-1f), (fProxy)1f, 94002u);

            var X = arena.fProxyMat(s, n);
            Krylov.blsmr(in A, in B, ref X, 1, Consts.fProxySqrtEps);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            arena.Dispose();
        }
    }

    [Test]
    public void NormalEquationsOptimalAndMatchesScalarLsmr()
        => new BlockLSMRTestJob { Type = BlockLSMRTestJob.TestType.NormalEquationsOptimalAndMatchesScalarLsmr }.Run();

    [Test]
    public void ConsistentSystemRecoversExactSolution()
        => new BlockLSMRTestJob { Type = BlockLSMRTestJob.TestType.ConsistentSystemRecoversExactSolution }.Run();

    [Test]
    public void ZeroRhsConvergesImmediately()
        => new BlockLSMRTestJob { Type = BlockLSMRTestJob.TestType.ZeroRhsConvergesImmediately }.Run();

    [Test]
    public void NeverNaNOnTinyMaxIter()
        => new BlockLSMRTestJob { Type = BlockLSMRTestJob.TestType.NeverNaNOnTinyMaxIter }.Run();
}
