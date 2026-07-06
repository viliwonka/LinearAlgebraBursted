using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class floatInitTest
{
    [BurstCompile(CompileSynchronously = true)]
    public struct InitVecTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            Assert.AreEqual(0, arena.AllocationsCount);

            int vecLen = 7;

            floatN vec = arena.floatVec(vecLen);

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

    [BurstCompile(CompileSynchronously = true)]
    public struct InitMatrixTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int lenRows = 7;
            int lenColumns = 7;

            floatMxN vec = arena.floatMat(lenRows, lenColumns);

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
