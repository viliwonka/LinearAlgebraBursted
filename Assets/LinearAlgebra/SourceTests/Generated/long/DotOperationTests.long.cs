using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class longDotOperationTests
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

            longN x = arena.longVec(vecLen, 1);
            longN y = arena.longVec(vecLen, 1);

            long b = longOP.dot(x, y);

            Assert.AreEqual((long)vecLen, b);
            
            x = arena.longVec(vecLen);
            y = arena.longVec(vecLen);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (long) ((i+0) % 2);
                y[i] = (long) ((i+1) % 2);
            }

            b = longOP.dot(x, y);

            Assert.AreEqual((long)0f, b);

            arena.Dispose();
        }

        public void MatVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 20;
            int outVecLen = 5;

            longN x = arena.longVec(inVecLen, 1);
            longMxN A = arena.longRandomMatrix(outVecLen, inVecLen, -100, +100);

            longN b = longOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 20;

            longN x = arena.longIndexOneVector(vecLen);
            longMxN A = arena.longIdentityMatrix(vecLen);

            longN b = longOP.dot(x, A);

            Assert.AreEqual(vecLen, b.N);
            
            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual(x[i], b[i]);

            x = arena.longIndexZeroVector(vecLen);

            b = longOP.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((long)i, b[i]);

            arena.Dispose();
        }

        public void MatMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int matLen = 16;

            longMxN A = arena.longIdentityMatrix(matLen);
            longMxN B = arena.longIdentityMatrix(matLen);

            longMxN C = longOP.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.AreEqual((long)1f, C[i, j]);
                else
                    Assert.AreEqual((long)0f, C[i, j]);
            }

            longMxN R = arena.longRandomMatrix(matLen, matLen);

            C = longOP.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.AreEqual(R[i, j], C[i, j]);
            }

            C = arena.longIdentityMatrix(matLen);

            longMxN D = longOP.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.AreEqual((long)1f, C[i, j]);
                else
                    Assert.AreEqual((long)0f, C[i, j]);
            }

            arena.Dispose();
        }

        public void MatVecDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            longN x = arena.longVec(inVecLen, 1);
            longMxN A = arena.longRandomMatrix(outVecLen, inVecLen, -100, +100);

            longN b = longOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            longN x = arena.longVec(inVecLen, 1);
            longMxN A = arena.longRandomMatrix(inVecLen, outVecLen, -100, +100);

            longN b = longOP.dot(x, A);
            
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

            longN x = arena.longVec(vecM, 1);
            longN y = arena.longVec(vecN, 1);

            longMxN A = longOP.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            longMxN B = longOP.outerDot(y, x);

            for (int i = 0; A.Length < i; i++)
                Assert.AreEqual((long)1, A[i]);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; B.Length < i; i++)
                Assert.AreEqual((long)1, B[i]);

            x = arena.longLinVector(vecM, 0, 20);
            y = arena.longLinVector(vecN, 0, 20);

            longMxN C = longOP.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.AreEqual((long)x[i] * y[j], (long)C[i, j]);

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
