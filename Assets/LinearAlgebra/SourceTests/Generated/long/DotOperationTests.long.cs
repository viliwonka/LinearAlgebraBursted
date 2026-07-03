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

            long b = Blas.dot(x, y);

            Assert.IsTrue(b == (long)vecLen);

            x = arena.longVec(vecLen);
            y = arena.longVec(vecLen);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (long) ((i+0) % 2);
                y[i] = (long) ((i+1) % 2);
            }

            b = Blas.dot(x, y);

            Assert.IsTrue(b == (long)0f);

            arena.Dispose();
        }

        public void MatVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 20;
            int outVecLen = 5;

            longN x = arena.longVec(inVecLen, 1);
            longMxN A = arena.longRandomMat(outVecLen, inVecLen, -100, +100);

            longN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 20;

            longN x = arena.longIndexOneVec(vecLen);
            longMxN A = arena.longIdentityMat(vecLen);

            longN b = Blas.dot(x, A);

            Assert.AreEqual(vecLen, b.N);
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == x[i]);

            x = arena.longIndexZeroVec(vecLen);

            b = Blas.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == (long)i);

            arena.Dispose();
        }

        public void MatMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int matLen = 16;

            longMxN A = arena.longIdentityMat(matLen);
            longMxN B = arena.longIdentityMat(matLen);

            longMxN C = Blas.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (long)1f);
                else
                    Assert.IsTrue(C[i, j] == (long)0f);
            }

            longMxN R = arena.longRandomMat(matLen, matLen);

            C = Blas.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.IsTrue(C[i, j] == R[i, j]);
            }

            C = arena.longIdentityMat(matLen);

            longMxN D = Blas.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (long)1f);
                else
                    Assert.IsTrue(C[i, j] == (long)0f);
            }

            arena.Dispose();
        }

        public void MatVecDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            longN x = arena.longVec(inVecLen, 1);
            longMxN A = arena.longRandomMat(outVecLen, inVecLen, -100, +100);

            longN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            longN x = arena.longVec(inVecLen, 1);
            longMxN A = arena.longRandomMat(inVecLen, outVecLen, -100, +100);

            longN b = Blas.dot(x, A);
            
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

            longMxN A = Blas.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            longMxN B = Blas.outerDot(y, x);

            for (int i = 0; i < A.Length; i++)
                Assert.IsTrue(A[i] == (long)1);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; i < B.Length; i++)
                Assert.IsTrue(B[i] == (long)1);

            x = arena.longLinVec(vecM, 0, 20);
            y = arena.longLinVec(vecN, 0, 20);

            longMxN C = Blas.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.IsTrue((long)C[i, j] == (long)x[i] * y[j]);

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
