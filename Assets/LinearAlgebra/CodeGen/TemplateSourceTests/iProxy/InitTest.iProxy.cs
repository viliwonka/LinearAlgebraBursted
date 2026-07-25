using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class iProxyInitTest
{
    [BurstCompile(CompileSynchronously = true)]
    public struct InitVecTestJob : IJob
    {
        public void Execute()
        {
            int vecLen = 7;

            iProxyN vec = new iProxyN(vecLen, Allocator.Temp);

            Assert.AreEqual(vecLen, vec.N);
        }
    }

    [Test]
    public void InitTestVecPass()
    {
        new InitVecTestJob().Run();
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct InitMatrixTestJob : IJob
    {
        public void Execute()
        {
            int lenRows = 7;
            int lenColumns = 7;

            iProxyMxN vec = new iProxyMxN(lenRows, lenColumns, Allocator.Temp);

            Assert.AreEqual(lenRows * lenColumns, vec.Length);
        }
    }

    [Test]
    public void InitMatrixVecPass()
    {
        new InitMatrixTestJob().Run();
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct LinVecExactTestJob : IJob
    {
        public void Execute()
        {
            // Small exact ramp: (end-start) divisible by (N-1), so every element is exact.
            iProxyN v = GenerateOP.iProxyLinVec(5, 0, 8);
            for (int i = 0; i < 5; i++)
                Assert.IsTrue(v[i] == (iProxy)(2 * i));

            // Large-endpoint ramp: interior values need more mantissa bits than float has
            // (regression: interpolating in float corrupted interior values; the long variant
            // was off by up to ~2^38). Endpoint chosen per type so end/4 is exact.
            iProxy bigEnd = /*+choose[(1 << 28) + 4|(short)32764|(1L << 40) + 4]*/(1 << 28) + 4/*-choose*/;
            iProxyN w = GenerateOP.iProxyLinVec(5, 0, bigEnd);
            long step = (long)bigEnd / 4;
            for (int i = 0; i < 5; i++)
                Assert.IsTrue((long)w[i] == i * step);

            // Single-sample convention: returns {start}.
            iProxyN s = GenerateOP.iProxyLinVec(1, 3, 9);
            Assert.IsTrue(s[0] == (iProxy)3);
        }
    }

    [Test]
    public void LinVecExactPass()
    {
        new LinVecExactTestJob().Run();
    }

}
