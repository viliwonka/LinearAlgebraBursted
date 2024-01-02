using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxyInitTest
{
    [BurstCompile]
    public struct InitVecTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            Assert.AreEqual(0, arena.AllocationsCount);

            int vecLen = 7;

            fProxyN vec = arena.fProxyVec(vecLen);

            Assert.AreEqual(vecLen, vec.N);
            Assert.AreEqual(1, arena.AllocationsCount);
            
            arena.Clear();

            Assert.AreEqual(0, arena.AllocationsCount);
        }
    }

    [Test]
    public void InitTestVecPass()
    {
        new InitVecTestJob().Run();
    }

    [BurstCompile]
    public struct InitMatrixTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int lenRows = 7;
            int lenColumns = 7;

            fProxyMxN vec = arena.fProxyMat(lenRows, lenColumns);

            Assert.AreEqual(lenRows * lenColumns, vec.Length);
            Assert.AreEqual(1, arena.AllocationsCount);

            arena.Dispose();

            Assert.AreEqual(0, arena.AllocationsCount);

        }
    }

    [Test]
    public void InitMatrixVecPass()
    {
        new InitMatrixTestJob().Run();
    }
    
}
