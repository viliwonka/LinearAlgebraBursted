using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class iProxyDotOperationTests
{
    [BurstCompile(CompileSynchronously = true)]
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

            iProxyN x = arena.iProxyVec(vecLen, 1);
            iProxyN y = arena.iProxyVec(vecLen, 1);

            iProxy b = Blas.dot(x, y);

            Assert.IsTrue(b == (iProxy)vecLen);

            x = arena.iProxyVec(vecLen);
            y = arena.iProxyVec(vecLen);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (iProxy) ((i+0) % 2);
                y[i] = (iProxy) ((i+1) % 2);
            }

            b = Blas.dot(x, y);

            Assert.IsTrue(b == (iProxy)0f);

            arena.Dispose();
        }

        public void MatVecDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 20;
            int outVecLen = 5;

            iProxyN x = arena.iProxyVec(inVecLen, 1);
            iProxyMxN A = arena.iProxyRandomMat(outVecLen, inVecLen, -100, +100);

            iProxyN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 20;

            iProxyN x = arena.iProxyIndexOneVec(vecLen);
            iProxyMxN A = arena.iProxyIdentityMat(vecLen);

            iProxyN b = Blas.dot(x, A);

            Assert.AreEqual(vecLen, b.N);
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == x[i]);

            x = arena.iProxyIndexZeroVec(vecLen);

            b = Blas.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == (iProxy)i);

            arena.Dispose();
        }

        public void MatMatDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int matLen = 16;

            iProxyMxN A = arena.iProxyIdentityMat(matLen);
            iProxyMxN B = arena.iProxyIdentityMat(matLen);

            iProxyMxN C = Blas.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (iProxy)1f);
                else
                    Assert.IsTrue(C[i, j] == (iProxy)0f);
            }

            iProxyMxN R = arena.iProxyRandomMat(matLen, matLen);

            C = Blas.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.IsTrue(C[i, j] == R[i, j]);
            }

            C = arena.iProxyIdentityMat(matLen);

            iProxyMxN D = Blas.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (iProxy)1f);
                else
                    Assert.IsTrue(C[i, j] == (iProxy)0f);
            }

            arena.Dispose();
        }

        public void MatVecDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            iProxyN x = arena.iProxyVec(inVecLen, 1);
            iProxyMxN A = arena.iProxyRandomMat(outVecLen, inVecLen, -100, +100);

            iProxyN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);

            arena.Dispose();
        }

        public void VecMatDotNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int inVecLen = 64;
            int outVecLen = 16;

            iProxyN x = arena.iProxyVec(inVecLen, 1);
            iProxyMxN A = arena.iProxyRandomMat(inVecLen, outVecLen, -100, +100);

            iProxyN b = Blas.dot(x, A);
            
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

            iProxyN x = arena.iProxyVec(vecM, 1);
            iProxyN y = arena.iProxyVec(vecN, 1);

            iProxyMxN A = Blas.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            iProxyMxN B = Blas.outerDot(y, x);

            for (int i = 0; i < A.Length; i++)
                Assert.IsTrue(A[i] == (iProxy)1);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; i < B.Length; i++)
                Assert.IsTrue(B[i] == (iProxy)1);

            x = arena.iProxyLinVec(vecM, 0, 20);
            y = arena.iProxyLinVec(vecN, 0, 20);

            iProxyMxN C = Blas.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.IsTrue((iProxy)C[i, j] == (iProxy)x[i] * y[j]);

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
