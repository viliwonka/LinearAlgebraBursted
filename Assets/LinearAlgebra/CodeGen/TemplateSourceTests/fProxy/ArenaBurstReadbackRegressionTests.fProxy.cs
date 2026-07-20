using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Regression guard (task #54): Arena-built random data must read back NONZERO inside a
// [BurstCompile] job, and be assertable there. The trap that motivated this: a test that
// reads a COMPUTED RESULT back from a plain job FIELD after .Run() sees the by-value job
// copy's discarded write (always the field's default 0), NOT the data -- which silently
// turns X == 0 into a passing least-squares / normal-equations oracle (a false green).
// The correct pattern -- the one the block solver tests already use -- is to verify INSIDE
// the job (or write outputs through persistent native storage), never off a job field after
// .Run(). This guard asserts the checksum inside Execute(), so any future "Arena data reads
// back as zero inside Burst" fails loudly instead of passing.
public class fProxyArenaBurstReadbackRegressionTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct ChecksumJob : IJob
    {
        public int M;
        public int N;
        public int S;
        public uint SeedA;
        public uint SeedB;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomMat(M, N, (fProxy)(-1f), (fProxy)1f, SeedA);
            var B = arena.fProxyRandomMat(S, M, (fProxy)(-1f), (fProxy)1f, SeedB);

            double sumA = 0;
            for (int i = 0; i < A.Length; i++) sumA += (double)A[i] * (double)A[i];
            double sumB = 0;
            for (int i = 0; i < B.Length; i++) sumB += (double)B[i] * (double)B[i];

            // AᵀB, built via the same Row-extraction + Blas.dot pattern the block tests use,
            // to exercise a derived quantity too (not just the raw random fill).
            var ATB = arena.fProxyMat(S, N);
            for (int j = 0; j < S; j++)
            {
                var bj = arena.fProxyVec(M);
                for (int c = 0; c < M; c++) bj[c] = B[j, c];
                var atbj = arena.fProxyVec(N);
                Blas.dot(in bj, in A, ref atbj);
                for (int c = 0; c < N; c++) ATB[j, c] = atbj[c];
            }
            double sumATB = 0;
            for (int i = 0; i < ATB.Length; i++) sumATB += (double)ATB[i] * (double)ATB[i];

            // Asserted INSIDE the job -- this is the whole point: the data is real here, so a
            // zero checksum means a genuine Arena-in-Burst regression, not lost field writes.
            Assert.IsTrue(sumA > 0.0);
            Assert.IsTrue(sumB > 0.0);
            Assert.IsTrue(sumATB > 0.0);

            arena.Dispose();
        }
    }

    [Test]
    public void ArenaRandomDataIsNonzeroInsideBurstJob()
        => new ChecksumJob { M = 20, N = 6, S = 3, SeedA = 92001u, SeedB = 92002u }.Run();
}
