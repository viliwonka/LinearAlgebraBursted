using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Aliasing guard for bcraig/bcraigmr: both own their entire internal workspace (Allocator.Temp,
// freshly allocated every call), so X and B are the only two caller-supplied buffers -- an X that
// aliases B must be rejected up front (RequireDistinctBuffers), matching every other block Krylov
// solver's own caller-buffer guard (e.g. Krylov.bbiCGStab's R/Rhat0/P/V/T/Phat/Shat/X/B check).
// Without the guard, an aliased X silently destroys B mid-solve and can report a false Converged
// with X left holding garbage instead of a real solution.
//
// The distinct-buffer solves run inside a [BurstCompile] IJob (matches every other Krylov suite);
// the Assert.Throws guard cases are managed [Test]s (a Burst job cannot surface an assertable
// managed exception), mirroring fProxyKrylovPMinresTests.AliasedScratchThrows.
public class fProxyBlockCraigGuardTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            BcraigDistinctXBSolves,
            BcraigmrDistinctXBSolves,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.BcraigDistinctXBSolves:   BcraigDistinctXBSolves();   break;
                case TestType.BcraigmrDistinctXBSolves: BcraigmrDistinctXBSolves(); break;
            }
        }

        static fProxyMxN BuildWideFullRowRank(ref Arena arena, int m, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, seed);
            for (int d = 0; d < m; d++) A[d, d] += (fProxy)10;
            return A;
        }

        // Normal distinct-buffer path is unaffected by the added guard.
        void BcraigDistinctXBSolves()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 8, s = 2;
            var A = BuildWideFullRowRank(ref arena, m, n, 88101u);
            var B = arena.fProxyRandomMat(s, m, (fProxy)(-1f), (fProxy)1f, 88102u);
            var X = arena.fProxyMat(s, n);

            var info = Krylov.bcraig(in A, in B, ref X);
            Assert.IsTrue(info.Solved);

            arena.Dispose();
        }

        void BcraigmrDistinctXBSolves()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 8, s = 2;
            var A = BuildWideFullRowRank(ref arena, m, n, 89101u);
            var B = arena.fProxyRandomMat(s, m, (fProxy)(-1f), (fProxy)1f, 89102u);
            var X = arena.fProxyMat(s, n);

            var info = Krylov.bcraigmr(in A, in B, ref X);
            Assert.IsTrue(info.Solved);

            arena.Dispose();
        }
    }

    [Test]
    public void BcraigDistinctXBSolves()
        => new TestJob { Type = TestJob.TestType.BcraigDistinctXBSolves }.Run();

    [Test]
    public void BcraigmrDistinctXBSolves()
        => new TestJob { Type = TestJob.TestType.BcraigmrDistinctXBSolves }.Run();

    // ==============================================================================
    // Managed [Test]s: aliasing guard throws.
    // ==============================================================================

    static fProxyMxN BuildSquareFullRank(ref Arena arena, int nn, uint seed)
    {
        var A = arena.fProxyRandomMat(nn, nn, (fProxy)(-1f), (fProxy)1f, seed);
        for (int d = 0; d < nn; d++) A[d, d] += (fProxy)10;
        return A;
    }

    // SQUARE A (bcraig/bcraigmr allow A.Rows <= A.Cols, so Rows == Cols qualifies) so B (s x
    // A.Rows) and X (s x A.Cols) share the SAME shape -- only then can X literally alias B (a
    // wide, non-square A gives B/X different N_Cols, so an aliased X would already be rejected by
    // the earlier shape check, never reaching the aliasing guard under test here).
    [Test]
    public void BcraigAliasedXBThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int nn = 8, s = 2;
            var A = BuildSquareFullRank(ref arena, nn, 88001u);
            var B = arena.fProxyRandomMat(s, nn, (fProxy)(-1f), (fProxy)1f, 88002u);
            var X = B;   // ALIASES B -> distinct-buffer guard must fire

            Assert.Throws<System.ArgumentException>(() => Krylov.bcraig(in A, in B, ref X));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void BcraigmrAliasedXBThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int nn = 8, s = 2;
            var A = BuildSquareFullRank(ref arena, nn, 89001u);
            var B = arena.fProxyRandomMat(s, nn, (fProxy)(-1f), (fProxy)1f, 89002u);
            var X = B;   // ALIASES B -> distinct-buffer guard must fire

            Assert.Throws<System.ArgumentException>(() => Krylov.bcraigmr(in A, in B, ref X));
        }
        finally { arena.Dispose(); }
    }
}
