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

            fProxyN x = GenerateOP.fProxyVec(vecLen, 1f);
            fProxyN y = GenerateOP.fProxyVec(vecLen, 1f);

            fProxy b = Blas.dot(x, y);

            Assert.IsTrue(b == (fProxy)vecLen);

            x = new fProxyN(vecLen, Allocator.Temp);
            y = new fProxyN(vecLen, Allocator.Temp);

            for(int i = 0; i < vecLen; i++)
            {
                x[i] = (i+0f) % 2f;
                y[i] = (i+1f) % 2f;
            }

            b = Blas.dot(x, y);

            Assert.IsTrue(b == (fProxy)0f);
        }

        public void MatVecDot()
        {
            int inVecLen = 20;
            int outVecLen = 5;

            // A[i,j] = i+1 against an all-ones x: b[i] must be exactly (i+1) * inVecLen.
            fProxyN x = GenerateOP.fProxyVec(inVecLen, 1f);
            fProxyMxN A = new fProxyMxN(outVecLen, inVecLen, Allocator.Temp);
            for (int i = 0; i < outVecLen; i++)
                for (int j = 0; j < inVecLen; j++)
                    A[i, j] = (fProxy)(i + 1);

            fProxyN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);
            for (int i = 0; i < outVecLen; i++)
                Assert.IsTrue(b[i] == (fProxy)((i + 1) * inVecLen));
        }

        public void VecMatDot()
        {
            int vecLen = 20;

            fProxyN x = GenerateOP.fProxyRandomUnitVec(vecLen);
            fProxyMxN A = GenerateOP.fProxyIdentityMat(vecLen);

            fProxyN b = Blas.dot(x, A);

            Assert.AreEqual(vecLen, b.N);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == x[i]);

            x = GenerateOP.fProxyIndexZeroVec(vecLen);

            b = Blas.dot(x, A);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == (fProxy)i);
        }

        public void MatMatDot()
        {
            int matLen = 16;

            fProxyMxN A = GenerateOP.fProxyIdentityMat(matLen);
            fProxyMxN B = GenerateOP.fProxyIdentityMat(matLen);

            fProxyMxN C = Blas.dot(A, B);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(C[i, j] == (fProxy)1f);
                else
                    Assert.IsTrue(C[i, j] == (fProxy)0f);
            }

            fProxyMxN R = GenerateOP.fProxyRandomMat(matLen, matLen);

            C = Blas.dot(A, R);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                Assert.IsTrue(C[i, j] == R[i, j]);
            }

            C = GenerateOP.fProxyIdentityMat(matLen);

            fProxyMxN D = Blas.dot(C, C);

            for (int i = 0; i < matLen; i++)
            for (int j = 0; j < matLen; j++)
            {
                if (i == j)
                    Assert.IsTrue(D[i, j] == (fProxy)1f);
                else
                    Assert.IsTrue(D[i, j] == (fProxy)0f);
            }
        }

        public void MatVecDotNonSquare()
        {
            int inVecLen = 64;
            int outVecLen = 16;

            // A[i,j] = i+1 against an all-ones x: b[i] must be exactly (i+1) * inVecLen
            // (small integers: exact in float, so == comparison is safe).
            fProxyN x = GenerateOP.fProxyVec(inVecLen, 1f);
            fProxyMxN A = new fProxyMxN(outVecLen, inVecLen, Allocator.Temp);
            for (int i = 0; i < outVecLen; i++)
                for (int j = 0; j < inVecLen; j++)
                    A[i, j] = (fProxy)(i + 1);

            fProxyN b = Blas.dot(A, x);

            Assert.AreEqual(outVecLen, b.N);
            for (int i = 0; i < outVecLen; i++)
                Assert.IsTrue(b[i] == (fProxy)((i + 1) * inVecLen));
        }

        public void VecMatDotNonSquare()
        {
            int inVecLen = 64;
            int outVecLen = 16;

            fProxyN x = GenerateOP.fProxyVec(inVecLen, 1f);
            fProxyMxN A = GenerateOP.fProxyRandomMat(inVecLen, outVecLen, -0.01f, 0.01f);

            fProxyN b = Blas.dot(x, A);

            Assert.AreEqual(outVecLen, b.N);
        }

        public void MatMatDotNonSquare()
        {
            int M = 8;
            int K = 24;
            int N = 16;

            fProxyMxN Id = GenerateOP.fProxyIdentityMat(K);
            fProxyMxN R = GenerateOP.fProxyRandomMat(K, N);

            fProxyMxN C = Blas.dot(Id, R);

            Assert.AreEqual(K, C.M_Rows);
            Assert.AreEqual(N, C.N_Cols);

            for (int i = 0; i < K; i++)
            for (int j = 0; j < N; j++)
                Assert.IsTrue(C[i, j] == R[i, j]);

            fProxyMxN R2 = GenerateOP.fProxyRandomMat(M, K);

            fProxyMxN D = Blas.dot(R2, Id);

            Assert.AreEqual(M, D.M_Rows);
            Assert.AreEqual(K, D.N_Cols);

            for (int i = 0; i < M; i++)
            for (int j = 0; j < K; j++)
                Assert.IsTrue(D[i, j] == R2[i, j]);
        }

        public void OuterDot()
        {
            int vecM = 32;
            int vecN = 64;

            fProxyN x = GenerateOP.fProxyVec(vecM, 1f);
            fProxyN y = GenerateOP.fProxyVec(vecN, 1f);

            fProxyMxN A = Blas.outerDot(x, y);

            Assert.AreEqual(vecM, A.M_Rows);
            Assert.AreEqual(vecN, A.N_Cols);

            fProxyMxN B = Blas.outerDot(y, x);

            for (int i = 0; i < A.Length; i++)
                Assert.IsTrue(A[i] == (fProxy)1f);

            Assert.AreEqual(vecM, B.N_Cols);
            Assert.AreEqual(vecN, B.M_Rows);

            for (int i = 0; i < B.Length; i++)
                Assert.IsTrue(B[i] == (fProxy)1f);

            x = GenerateOP.fProxyLinVec(vecM, 0f, 2f);
            y = GenerateOP.fProxyLinVec(vecN, 0f, 2f);

            fProxyMxN C = Blas.outerDot(x, y);

            for (int i = 0; i < vecM; i++)
                for (int j = 0; j < vecN; j++)
                    Assert.IsTrue(C[i, j] == x[i] * y[j]);
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
