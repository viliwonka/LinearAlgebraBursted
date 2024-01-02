using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class doubleOrthoOpTests
{
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            var Q = arena.doubleRandomMatrix(dim*2, dim);
            var R = arena.doubleMat(dim);

            OrthoOP.qrDecomposition(ref Q, ref R);

            arena.Dispose();
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            QRDecompIdentity,
            QRDecompIdentityNonSquare,
            QRDecompRandomDiagonal,
            QRDecompRandom,
            QRDecompRandomLarge,
            QRDecompHilbert,
            QRDecompPermutation,
            QRDecompZero,
        }

        public TestType Type;


        public void Execute()
        {
            switch(Type)
            {
                case TestType.QRDecompIdentity:
                    QRDecompIdentity();
                break;
                case TestType.QRDecompIdentityNonSquare:
                    QRDecompIdentityNonSquare();
                break;
                case TestType.QRDecompRandomDiagonal:
                    QRDecompRandomDiagonal();
                break;
                case TestType.QRDecompRandom:
                    QRDecompRandom();
                break;
                case TestType.QRDecompRandomLarge:
                    QRDecompRandomLarge();
                    break;
                case TestType.QRDecompHilbert:
                    QRDecompHilbert();
                break;
                case TestType.QRDecompPermutation:
                    QRDecompPermutation();
                    break;
                case TestType.QRDecompZero:
                    QRDecompZero();
                    break;
            }
        }

        public void QRDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.doubleIdentityMatrix(dim);
            var R = arena.doubleMat(dim);

            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            //Print.Log(A);
            //Print.Log(Q);
            //Print.Log(R);
            
            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompIdentityNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.doubleMat(dim*2, dim);
            var R = arena.doubleMat(dim);

            for(int i = 0; i < dim; i++)
                Q[i, i] = 1f;
            
            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            
            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompRandomDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.doubleRandomDiagonalMatrix(dim, 1f, 3f);
            var R = arena.doubleMat(dim);

            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            /*Print.Log(A);
            Print.Log(Q);
            Print.Log(R);*/

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompRandom()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var R = arena.doubleMat(dim);
            var Q = arena.doubleRandomMatrix(dim*2, dim, -0.5f, 0.5f, 94221);
            
            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            Print.Log(R);

            AssertQR(in A, in Q, in R, 1E-05f);

            arena.Dispose();
        }

        public void QRDecompRandomLarge()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var R = arena.doubleMat(dim);
            var Q = arena.doubleRandomMatrix(dim * 2, dim, -5f, 5f, 9612221);

            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            AssertQR(in A, in Q, in R, 1E-03f);

            arena.Dispose();
        }

        public void QRDecompHilbert()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 20;

            var Q = arena.doubleHilbertMatrix(dim);
            var R = arena.doubleMat(dim);

            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            //Print.Log(A);
            //Print.Log(Q);
            //Print.Log(R);

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompPermutation() {

            var arena = new Arena(Allocator.Persistent);

            int tests = 32;
            int dim = 16;
            var rand = new Unity.Mathematics.Random(24011);

            for (int i = 0; i < tests; i++) {

                int p0 = rand.NextInt(0, dim);
                int p1 = rand.NextInt(0, dim);

                while(p0 == p1) {
                    p1 = rand.NextInt(0, dim);
                }

                var Q = arena.doublePermutationMatrix(dim, p0, p1);

                p0 = rand.NextInt(0, dim);
                p1 = rand.NextInt(0, dim);

                while (p0 == p1) {
                    p1 = rand.NextInt(0, dim);
                }

                Q = doubleOP.dot(arena.doublePermutationMatrix(dim, p0, p1), Q);

                var R = arena.doubleMat(dim);

                var A = Q.Copy();

                OrthoOP.qrDecomposition(ref Q, ref R);

                //Print.Log(A);
                //Print.Log(Q);
                //Print.Log(R);

                AssertQR(in A, in Q, in R);
            }
            arena.Dispose();
        }

        public void QRDecompZero() {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.doubleMat(dim, dim);
            var R = arena.doubleMat(dim);

            var A = Q.Copy();

            OrthoOP.qrDecomposition(ref Q, ref R);

            //Print.Log(A);
            //Print.Log(Q);
            //Print.Log(R);

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        private void AssertQR(in doubleMxN A, in doubleMxN Q, in doubleMxN R) => AssertQR(in A, in Q, in R, 1E-6f);
        private void AssertQR(in doubleMxN A, in doubleMxN Q, in doubleMxN R, double precision)
        {
            doubleMxN shouldBeZero = A - doubleOP.dot(Q, R);

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            Debug.Log($"Error of max(abs(A - QR)): {zeroError}");

            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.IsUpperTriangular(R, precision));
            Assert.IsTrue(Analysis.IsOrthogonal(Q, precision));
        }
    }

    [BurstCompile]
    public struct PrecisionReconstructTestJob : IJob {

        public enum TestType {
            Random,
            RandomDiagonal
        }

        public TestType Type;

        public void Execute() {

            var arena = new Arena(Allocator.Persistent);

            int tests = 64;
            double errorSum = 0;

            for (uint i = 0; i < tests; i++) {

                int dim = 32;

                doubleMxN A; 
                
                if(Type == TestType.RandomDiagonal)
                    A = arena.doubleRandomDiagonalMatrix(dim, 1f, 3f, 21410 + i*i + i*7);
                else
                    A = arena.doubleRandomMatrix(dim*2, dim, -25f, +25f, 21410 + i*i + i*7);
                
                var Q = A.Copy();
                var R = arena.doubleMat(dim);

                OrthoOP.qrDecomposition(ref Q, ref R);

                //Print.Log(Q);
                //Print.Log(R);

                errorSum += ErrorCheckQR(in A, in Q, in R);

                arena.Clear();
            }

            double avgError = errorSum / tests;

            Debug.Log($"Average error of max(abs(A - QR)): {avgError}");

            arena.Dispose();
        }

        private double ErrorCheckQR(in doubleMxN A, in doubleMxN Q, in doubleMxN R) {
            
            doubleMxN shouldBeZero = A - doubleOP.dot(Q, R);

            if(Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("PrecisionReconstructTestJob: NaN detected");

            //Print.Log(shouldBeZero); 

            double zeroError = Analysis.MaxZeroError(shouldBeZero);

            return zeroError;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Deterministic, FloatPrecision = FloatPrecision.High)]
    public struct SolveSystemTestJob : IJob {

        public enum TestType {
            SquareFullRank,
            OverdeterminedFullRank,

            SquareFullRankDirect,
            OverdeterminedFullRankDirect,
        }

        public TestType Type;

        public void Execute() {

            switch(Type) {
            
                case TestType.SquareFullRank:
                    SquareFullRank();
                break;
                case TestType.OverdeterminedFullRank:
                    OverdeterminedFullRank();
                break;
                case TestType.SquareFullRankDirect:
                    SquareFullRankDirect();
                break;
                case TestType.OverdeterminedFullRankDirect:
                    OverdeterminedFullRankDirect();
                break;
            }
        }

        void SquareFullRank() {

            var arena = new Arena(Allocator.Persistent);

            int systemDim = 128;
            int randomMatTests = 128;
            int randomVecTests = 32;
            double errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                doubleMxN A = arena.doubleRandomMatrix(systemDim, systemDim, -5, +5, 420 + i - i + i * 7);

                for(int d = 0; d < systemDim; d++)
                    A[d, d] += 5.1f + 10f*random.NextDouble();

                var Q = A.Copy();
                var R = arena.doubleMat(systemDim); 

                OrthoOP.qrDecomposition(ref Q, ref R);

                for(uint j = 0; j < randomVecTests; j++) {

                    doubleN xOrig = arena.doubleRandomVector(systemDim, -25, +25, 1337 + i * i + i * 5);
                    doubleN b = doubleOP.dot(A, xOrig);
                    doubleN y = doubleOP.dot(b, Q);

                    Solvers.SolveUpperTriangular(ref R, ref y);

                    y.subInpl(xOrig);
                    double zeroError = Analysis.MaxZeroError(y);

                    if(Analysis.IsAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    errorSum += zeroError;
                }
            }

            double avgError = errorSum / (randomMatTests*randomVecTests);

            Debug.Log($"Average error: {avgError}");

            arena.Dispose();
        }

        void OverdeterminedFullRank() {


            int sysDimM = 128;
            int sysDimN = 64;
            int randomMatTests = 32;
            int randomVecTests = 16;
            double errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                var arena = new Arena(Allocator.Persistent);
                doubleMxN A = arena.doubleRandomMatrix(sysDimM, sysDimN, -5, +5, 420 + i - i + i * 7);

                for (int d = 0; d < sysDimN; d++)
                    A[d, d] += 5.1f + 10f * random.NextDouble();

                var Q = A.Copy();
                var R = arena.doubleMat(sysDimN);

                OrthoOP.qrDecomposition(ref Q, ref R);

                for (uint j = 0; j < randomVecTests; j++) {

                    doubleN xOrig = arena.doubleRandomVector(sysDimN, -25, +25, 1337 + i * i + i * 5);
                    doubleN b = doubleOP.dot(A, xOrig);
                    doubleN y = doubleOP.dot(b, Q);

                    Solvers.SolveUpperTriangular(ref R, ref y);

                    y.subInpl(xOrig);
                    double zeroError = Analysis.MaxZeroError(y);

                    if (Analysis.IsAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    errorSum += zeroError;
                }
                arena.Dispose();
            }

            double avgError = errorSum / (randomMatTests * randomVecTests);

            Debug.Log($"Average error: {avgError}");

        }

        void SquareFullRankDirect() {

            var arena = new Arena(Allocator.Persistent);

            int systemDim = 128;  
            int randomMatTests = 128;
            double errorSum = 0;
             
            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                doubleMxN A = arena.doubleRandomMatrix(systemDim, systemDim, -5, +5, 420 + i - i + i * 7);

                for (int d = 0; d < systemDim; d++)
                    A[d, d] += 5.1f + 10f * random.NextDouble();

                doubleN xOrig = arena.doubleRandomVector(systemDim, -25, +25, 1337 + i * i + i * 5);
                doubleN b = doubleOP.dot(A, xOrig);
                doubleN x = arena.doubleVec(systemDim);

                OrthoOP.qrDirectSolve(ref A, ref b, ref x);

                if (Analysis.IsAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }
                x.subInpl(xOrig);

                
                double zeroError = Analysis.MaxZeroError(x);
                
                errorSum += zeroError;
            }

            double avgError = errorSum / (randomMatTests);

            Debug.Log($"Average error: {avgError}");

            arena.Dispose();
        }

        void OverdeterminedFullRankDirect() {


            int sysDimM = 128;
            int sysDimN = 64;
            int randomMatTests = 512;
            double errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                var arena = new Arena(Allocator.Persistent);
                doubleMxN A = arena.doubleRandomMatrix(sysDimM, sysDimN, -5, +5, 420 + i - i + i * 7);

                for (int d = 0; d < sysDimN; d++)
                    A[d, d] += 5.1f + 10f * random.NextDouble();

                doubleN xOrig = arena.doubleRandomVector(sysDimN, -25, +25, 1337 + i * i + i * 5);
                doubleN b = doubleOP.dot(A, xOrig);
                doubleN x = arena.doubleVec(sysDimN);

                OrthoOP.qrDirectSolve(ref A, ref b, ref x);

                if (Analysis.IsAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }
                
                x.subInpl(xOrig);

                double zeroError = Analysis.MaxZeroError(x);

                errorSum += zeroError;
                arena.Dispose();
            }

            double avgError = errorSum / (randomMatTests);

            Debug.Log($"Average error: {avgError}");

        }
    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void QRDecompTests(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }

    [Test]
    public void QRDecompErrorBenchRandom() {
        new PrecisionReconstructTestJob() { Type = PrecisionReconstructTestJob.TestType.Random }.Run();
    }

    [Test]
    public void QRDecompErrorBenchDiagonal() {
        new PrecisionReconstructTestJob() { Type = PrecisionReconstructTestJob.TestType.RandomDiagonal }.Run();
    }

    [Test]
    public void QRDecompErrorSolveSquareSystem() {
        new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.SquareFullRank }.Run();
    }

    [Test]
    public void QRDecompErrorSolveOverdeterminedSystem() {
        new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.OverdeterminedFullRank }.Run();
    }

    [Test]
    public void QRDecompErrorSolveSquareSystemDirect() {
        new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.SquareFullRankDirect }.Run(); 
    }

    [Test]
    public void QRDecompErrorSolveOverdeterminedSystemDirect() {
        new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.OverdeterminedFullRankDirect }.Run();
    }
}
