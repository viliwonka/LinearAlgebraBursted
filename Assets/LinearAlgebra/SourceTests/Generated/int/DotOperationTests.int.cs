using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class intDotOperationTests
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

            intN x = arena.intVec(vecLen, 1);
            intN y = arena.intVec(vecLen, 1);

            int b = intOP.dot(x, y);

            Assert.AreEqual((int)vecLen, b);
            
            x = arena.intVec(vecLen);
            y = arena.intVec(vecLen);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (int) ((i+0) % 2);
                y[i] = (int) ((i+1) % 2);
            }

            b = intOP.dot(x, y);

            Assert.AreEqual((int)0f, b);

            arena.Dispose();
        }

        public void MatVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 20;
            int outVecLen = 5;

            intN x = arena.intVec(inVecLen, 1);
            intMxN A = arena.intRandomMatrix(outVecLen, inVecLen, -100, +100);

            intN b = intOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 20;

            intN x = arena.intIndexOneVector(vecLen);
            intMxN A = arena.intIdentityMatrix(vecLen);

            intN b = intOP.dot(x, A);

            Assert.AreEqual(vecLen, b.N);
            
            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual(x[i], b[i]);

            x = arena.intIndexZeroVector(vecLen);

            b = intOP.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((int)i, b[i]);

            arena.Dispose();
        }

        public void MatMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int matLen = 16;

            intMxN A = arena.intIdentityMatrix(matLen);
            intMxN B = arena.intIdentityMatrix(matLen);

            intMxN C = intOP.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.AreEqual((int)1f, C[i, j]);
                else
                    Assert.AreEqual((int)0f, C[i, j]);
            }

            intMxN R = arena.intRandomMatrix(matLen, matLen);

            C = intOP.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.AreEqual(R[i, j], C[i, j]);
            }

            C = arena.intIdentityMatrix(matLen);

            intMxN D = intOP.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.AreEqual((int)1f, C[i, j]);
                else
                    Assert.AreEqual((int)0f, C[i, j]);
            }

            arena.Dispose();
        }

        public void MatVecDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            intN x = arena.intVec(inVecLen, 1);
            intMxN A = arena.intRandomMatrix(outVecLen, inVecLen, -100, +100);

            intN b = intOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            intN x = arena.intVec(inVecLen, 1);
            intMxN A = arena.intRandomMatrix(inVecLen, outVecLen, -100, +100);

            intN b = intOP.dot(x, A);
            
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

            intN x = arena.intVec(vecM, 1);
            intN y = arena.intVec(vecN, 1);

            intMxN A = intOP.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            intMxN B = intOP.outerDot(y, x);

            for (int i = 0; A.Length < i; i++)
                Assert.AreEqual((int)1, A[i]);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; B.Length < i; i++)
                Assert.AreEqual((int)1, B[i]);

            x = arena.intLinVector(vecM, 0, 20);
            y = arena.intLinVector(vecN, 0, 20);

            intMxN C = intOP.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.AreEqual((int)x[i] * y[j], (int)C[i, j]);

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
