using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class fProxyDotOperationTests
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

            fProxyN x = arena.fProxyVec(vecLen, 1f);
            fProxyN y = arena.fProxyVec(vecLen, 1f);

            fProxy b = fProxyOP.dot(x, y);

            Assert.IsTrue(b == (fProxy)vecLen);

            x = arena.fProxyVec(vecLen);
            y = arena.fProxyVec(vecLen);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (i+0f) % 2f;
                y[i] = (i+1f) % 2f;
            }

            b = fProxyOP.dot(x, y);

            Assert.IsTrue(b == (fProxy)0f);

            arena.Dispose();
        }

        public void MatVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 20;
            int outVecLen = 5;

            fProxyN x = arena.fProxyVec(inVecLen, 1f);
            fProxyMxN A = arena.fProxyRandomMatrix(outVecLen, inVecLen, -0.01f, 0.01f);

            fProxyN b = fProxyOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 20;

            fProxyN x = arena.fProxyRandomUnitVector(vecLen);
            fProxyMxN A = arena.fProxyIdentityMatrix(vecLen);

            fProxyN b = fProxyOP.dot(x, A);

            Assert.AreEqual(vecLen, b.N);
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == x[i]);

            x = arena.fProxyIndexZeroVector(vecLen);

            b = fProxyOP.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == (fProxy)i);

            arena.Dispose();
        }

        public void MatMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int matLen = 16;

            fProxyMxN A = arena.fProxyIdentityMatrix(matLen);
            fProxyMxN B = arena.fProxyIdentityMatrix(matLen);

            fProxyMxN C = fProxyOP.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (fProxy)1f);
                else
                    Assert.IsTrue(C[i, j] == (fProxy)0f);
            }

            fProxyMxN R = arena.fProxyRandomMatrix(matLen, matLen);

            C = fProxyOP.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.IsTrue(C[i, j] == R[i, j]);
            }

            C = arena.fProxyIdentityMatrix(matLen);

            fProxyMxN D = fProxyOP.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (fProxy)1f);
                else
                    Assert.IsTrue(C[i, j] == (fProxy)0f);
            }

            arena.Dispose();
        }

        public void MatVecDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            fProxyN x = arena.fProxyVec(inVecLen, 1f);
            fProxyMxN A = arena.fProxyRandomMatrix(outVecLen, inVecLen, -0.01f, 0.01f);

            fProxyN b = fProxyOP.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            fProxyN x = arena.fProxyVec(inVecLen, 1f);
            fProxyMxN A = arena.fProxyRandomMatrix(inVecLen, outVecLen, -0.01f, 0.01f);

            fProxyN b = fProxyOP.dot(x, A);
            
            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void MatMatDotNonSquare()
        {

        }

        public void OuterDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecM = 32;
            int vecN = 64;

            fProxyN x = arena.fProxyVec(vecM, 1f);
            fProxyN y = arena.fProxyVec(vecN, 1f);

            fProxyMxN A = fProxyOP.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            fProxyMxN B = fProxyOP.outerDot(y, x);

            for (int i = 0; i < A.Length; i++)
                Assert.IsTrue(A[i] == (fProxy)1f);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; i < B.Length; i++)
                Assert.IsTrue(B[i] == (fProxy)1f);

            x = arena.fProxyLinVector(vecM, 0f, 2f);
            y = arena.fProxyLinVector(vecN, 0f, 2f);

            fProxyMxN C = fProxyOP.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.IsTrue(C[i, j] == x[i] * y[j]);

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
