using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class shortDotOperationTests
{
    [BurstCompile]
    public struct DotOperationTestsJob : IJob
    {
        public enum TestType
        {
            VecVec,
            MatVec,
            VecMat,
            MatMat,
            VecMatNonSquare,
            MatVecNonSquare,
            MatMatNonSquare,
            OuterDot,
        }

        public TestType Type;

        public void Execute()
        {

            switch(Type)
            {
                case TestType.VecVec:
                    VecVecDot();
                    break;
                case TestType.MatVec:
                    MatVecDot();
                break;
                case TestType.VecMat:
                    VecMatDot();
                break;
                case TestType.MatMat:
                    MatMatDot();
                break;
                case TestType.VecMatNonSquare:
                    VecMatDotNonSquare();
                break;
                case TestType.MatVecNonSquare:
                    MatVecDotNonSquare();
                break;
                case TestType.MatMatNonSquare:
                    MatMatDotNonSquare();
                break;
                case TestType.OuterDot:
                    OuterDot();
                break;
            }
        }

        public void VecVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 32;

            shortN x = arena.shortVec(vecLen, 1);
            shortN y = arena.shortVec(vecLen, 1);

            short b = shortOP.dot(x, y);

            Assert.AreEqual((short)vecLen, b);
            
            x = arena.shortVec(vecLen);
            y = arena.shortVec(vecLen);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (short) ((i+0) % 2);
                y[i] = (short) ((i+1) % 2);
            }

            b = shortOP.dot(x, y);

            Assert.AreEqual((short)0f, b);

            arena.Dispose();
        }

        public void MatVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 20;
            int outVecLen = 5;

            shortN x = arena.shortVec(inVecLen, 1);
            shortMxN A = arena.shortRandomMatrix(outVecLen, inVecLen, -100, +100);

            shortN b = shortOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 20;

            shortN x = arena.shortIndexOneVector(vecLen);
            shortMxN A = arena.shortIdentityMatrix(vecLen);

            shortN b = shortOP.dot(x, A);

            Assert.AreEqual(vecLen, b.N);
            
            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual(x[i], b[i]);

            x = arena.shortIndexZeroVector(vecLen);

            b = shortOP.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)i, b[i]);

            arena.Dispose();
        }

        public void MatMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int matLen = 16;

            shortMxN A = arena.shortIdentityMatrix(matLen);
            shortMxN B = arena.shortIdentityMatrix(matLen);

            shortMxN C = shortOP.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.AreEqual((short)1f, C[i, j]);
                else
                    Assert.AreEqual((short)0f, C[i, j]);
            }

            shortMxN R = arena.shortRandomMatrix(matLen, matLen);

            C = shortOP.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.AreEqual(R[i, j], C[i, j]);
            }

            C = arena.shortIdentityMatrix(matLen);

            shortMxN D = shortOP.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.AreEqual((short)1f, C[i, j]);
                else
                    Assert.AreEqual((short)0f, C[i, j]);
            }

            arena.Dispose();
        }

        public void MatVecDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            shortN x = arena.shortVec(inVecLen, 1);
            shortMxN A = arena.shortRandomMatrix(outVecLen, inVecLen, -100, +100);

            shortN b = shortOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            shortN x = arena.shortVec(inVecLen, 1);
            shortMxN A = arena.shortRandomMatrix(inVecLen, outVecLen, -100, +100);

            shortN b = shortOP.dot(x, A);
            
            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void MatMatDotNonSquare()
        {

        }

        public void OuterDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecM = 16;
            int vecN = 32;

            shortN x = arena.shortVec(vecM, 1);
            shortN y = arena.shortVec(vecN, 1);

            shortMxN A = shortOP.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            shortMxN B = shortOP.outerDot(y, x);

            for (int i = 0; A.Length < i; i++)
                Assert.AreEqual((short)1, A[i]);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; B.Length < i; i++)
                Assert.AreEqual((short)1, B[i]);

            x = arena.shortLinVector(vecM, 0, 20);
            y = arena.shortLinVector(vecN, 0, 20);

            shortMxN C = shortOP.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.AreEqual((short)x[i] * y[j], (short)C[i, j]);

            arena.Dispose();
        }
    }

    [Test]
    public void VecVecDotDet()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.VecVec }.Run();
    }

    [Test]
    public void MatrixVectorDotTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.MatVec }.Run();
    }

    [Test]
    public void VectorMatrixDotTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.VecMat }.Run();
    }

    [Test]
    public void MatrixMatrixDotTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.MatMat }.Run();
    }

    [Test]
    public void MatrixVectorDotNonSquareTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.MatVecNonSquare }.Run();
    }

    [Test]
    public void VectorMatrixDotNonSquareTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.VecMatNonSquare }.Run();
    }

    [Test]
    public void MatrixMatrixDotNonSquareTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.MatMatNonSquare }.Run();
    }

    [Test]
    public void OuterDotTest()
    {
        new DotOperationTestsJob() { Type = DotOperationTestsJob.TestType.OuterDot }.Run();
    }
}
