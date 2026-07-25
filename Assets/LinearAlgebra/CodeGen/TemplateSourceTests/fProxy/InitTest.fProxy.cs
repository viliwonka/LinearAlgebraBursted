using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxyInitTest
{
    [BurstCompile(CompileSynchronously = true)]
    public struct InitVecTestJob : IJob
    {
        public void Execute()
        {
            int vecLen = 7;

            fProxyN vec = new fProxyN(vecLen, Allocator.Temp);

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

            fProxyMxN vec = new fProxyMxN(lenRows, lenColumns, Allocator.Temp);

            Assert.AreEqual(lenRows * lenColumns, vec.Length);
        }
    }

    [Test]
    public void InitMatrixVecPass()
    {
        new InitMatrixTestJob().Run();
    }
    
}
