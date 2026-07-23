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
            int vecLen = 32;

            iProxyN x = GenerateOP.iProxyVec(vecLen, 1);
            iProxyN y = GenerateOP.iProxyVec(vecLen, 1);

            iProxy b = Blas.dot(x, y);

            Assert.IsTrue(b == (iProxy)vecLen);

            x = new iProxyN(vecLen, Allocator.Temp);
            y = new iProxyN(vecLen, Allocator.Temp);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (iProxy) ((i+0) % 2);
                y[i] = (iProxy) ((i+1) % 2);
            }

            b = Blas.dot(x, y);

            Assert.IsTrue(b == (iProxy)0f);
        }

        public void MatVecDot()
        {
            int inVecLen = 20;
            int outVecLen = 5;

            // A[i,j] = i+1 against an all-ones x: b[i] must be exactly (i+1) * inVecLen.
            iProxyN x = GenerateOP.iProxyVec(inVecLen, 1);
            iProxyMxN A = new iProxyMxN(outVecLen, inVecLen, Allocator.Temp);
            for (int i = 0; i < outVecLen; i++)
                for (int j = 0; j < inVecLen; j++)
                    A[i, j] = (iProxy)(i + 1);

            iProxyN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);
            for (int i = 0; i < outVecLen; i++)
                Assert.IsTrue(b[i] == (iProxy)((i + 1) * inVecLen));
        }

        public void VecMatDot()
        {
            int vecLen = 20;

            iProxyN x = GenerateOP.iProxyIndexOneVec(vecLen);
            iProxyMxN A = GenerateOP.iProxyIdentityMat(vecLen);

            iProxyN b = Blas.dot(x, A);

            Assert.AreEqual(vecLen, b.N);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == x[i]);

            x = GenerateOP.iProxyIndexZeroVec(vecLen);

            b = Blas.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == (iProxy)i);
        }

        public void MatMatDot()
        {
            int matLen = 16;

            iProxyMxN A = GenerateOP.iProxyIdentityMat(matLen);
            iProxyMxN B = GenerateOP.iProxyIdentityMat(matLen);

            iProxyMxN C = Blas.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (iProxy)1f);
                else
                    Assert.IsTrue(C[i, j] == (iProxy)0f);
            }

            iProxyMxN R = GenerateOP.iProxyRandomMat(matLen, matLen);

            C = Blas.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.IsTrue(C[i, j] == R[i, j]);
            }

            C = GenerateOP.iProxyIdentityMat(matLen);

            iProxyMxN D = Blas.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(D[i, j] == (iProxy)1f);
                else
                    Assert.IsTrue(D[i, j] == (iProxy)0f);
            }
        }

        public void MatVecDotNonSquare()
        {
            int inVecLen = 64;
            int outVecLen = 16;

            // A[i,j] = i+1 against an all-ones x: b[i] must be exactly (i+1) * inVecLen.
            iProxyN x = GenerateOP.iProxyVec(inVecLen, 1);
            iProxyMxN A = new iProxyMxN(outVecLen, inVecLen, Allocator.Temp);
            for (int i = 0; i < outVecLen; i++)
                for (int j = 0; j < inVecLen; j++)
                    A[i, j] = (iProxy)(i + 1);

            iProxyN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);
            for (int i = 0; i < outVecLen; i++)
                Assert.IsTrue(b[i] == (iProxy)((i + 1) * inVecLen));
        }

        public void VecMatDotNonSquare()
        {
            int inVecLen = 64;
            int outVecLen = 16;

            iProxyN x = GenerateOP.iProxyVec(inVecLen, 1);
            iProxyMxN A = GenerateOP.iProxyRandomMat(inVecLen, outVecLen, -100, +100);

            iProxyN b = Blas.dot(x, A);

            Assert.AreEqual(outVecLen, b.N);
        }

        public void MatMatDotNonSquare()
        {
            int M = 8;
            int K = 24;
            int N = 16;

            iProxyMxN Id = GenerateOP.iProxyIdentityMat(K);
            iProxyMxN R = GenerateOP.iProxyRandomMat(K, N, -100, +100);

            iProxyMxN C = Blas.dot(Id, R);

            Assert.AreEqual(K, C.M_Rows);
            Assert.AreEqual(N, C.N_Cols);

            for (int i = 0; i < K; i++)
            for (int j = 0; j < N; j++)
                Assert.IsTrue(C[i, j] == R[i, j]);

            iProxyMxN R2 = GenerateOP.iProxyRandomMat(M, K, -100, +100);

            iProxyMxN D = Blas.dot(R2, Id);

            Assert.AreEqual(M, D.M_Rows);
            Assert.AreEqual(K, D.N_Cols);

            for (int i = 0; i < M; i++)
            for (int j = 0; j < K; j++)
                Assert.IsTrue(D[i, j] == R2[i, j]);
        }

        public void OuterDot()
        {
            int vecM = 16;
            int vecN = 32;

            iProxyN x = GenerateOP.iProxyVec(vecM, 1);
            iProxyN y = GenerateOP.iProxyVec(vecN, 1);

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

            x = GenerateOP.iProxyLinVec(vecM, 0, 20);
            y = GenerateOP.iProxyLinVec(vecN, 0, 20);

            iProxyMxN C = Blas.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.IsTrue((iProxy)C[i, j] == (iProxy)x[i] * y[j]);
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
