using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class iProxyTransposeTests
{
    [BurstCompile]
    public struct TransposeTestsJob : IJob
    {
        public enum TestType
        {
            TransSquare,
            TransNonSquare,
        }

        public TestType Type;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.TransSquare:
                    TransSquare();
                    break;

                case TestType.TransNonSquare:
                    TransNonSquare();
                    break;
            }
        }


        public void TransSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            iProxyMxN A = arena.iProxyRandomMatrix(dim, dim);

            iProxyMxN B = iProxyOP.trans(A);

            Assert.AreEqual(B.M_Rows, dim);
            Assert.AreEqual(B.N_Cols, dim);

            for (int r = 0; r < dim; r++)
            for (int c = 0; c < dim; c++)
                Assert.AreEqual(A[r, c], B[c, r]); 

            arena.Dispose();
        }

        public void TransNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8; 
            int cols = 32;

            iProxyMxN A = arena.iProxyRandomMatrix(rows, cols);

            iProxyMxN B = iProxyOP.trans(A);

            Assert.AreEqual(B.M_Rows, cols);
            Assert.AreEqual(B.N_Cols, rows);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.AreEqual(A[r, c], B[c, r]);

            arena.Dispose();
        }
    }

    [Test]
    public void TransposeTest()
    {
        new TransposeTestsJob() { Type = TransposeTestsJob.TestType.TransSquare }.Run();
    }

    [Test]
    public void TransposeNonSquareTest()
    {
        new TransposeTestsJob() { Type = TransposeTestsJob.TestType.TransNonSquare }.Run();
    }
}
