using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class doubleLUTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LUDecompIdentity,
            LUDecompRandomDiagonal,
            LUDecompRandom,
            LUDecompRandomLarge,
            LUDecompHilbert,
            LUDecompPermutation,
            LUDecompZero,
        }

        public TestType Type;


        public void Execute()
        {
            switch(Type)
            {
                case TestType.LUDecompIdentity:
                    LUDecompIdentity();
                break;
                case TestType.LUDecompRandomDiagonal:
                    LUDecompRandomDiagonal();
                break;
                case TestType.LUDecompRandom:
                    LUDecompRandom();
                break;
                case TestType.LUDecompRandomLarge:
                    LUDecompRandomLarge();
                    break;
                case TestType.LUDecompHilbert:
                    LUDecompHilbert();
                break;
                case TestType.LUDecompPermutation:
                    LUDecompPermutation();
                    break;
                case TestType.LUDecompZero:
                    LUDecompZero();
                    break;
            }
        }

        private doubleMxN GetRandomMatrix(ref Arena arena, int dim, double min, double max, uint seed) {

            var mat = arena.doubleRandomMatrix(dim, dim, min, max, seed);

            return mat;
        }

        public void LUDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.doubleIdentityMatrix(dim);
            var L = arena.doubleIdentityMatrix(dim);

            var A = U.Copy();

            LU.luDecompositionNoPivot(ref U, ref L);

            
            AssertLU(in A, in L, in U);

            arena.Dispose();
        }


        public void LUDecompRandomDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.doubleRandomDiagonalMatrix(dim, 1f, 3f);
            var L = arena.doubleIdentityMatrix(dim);

            var A = U.Copy();

            LU.luDecompositionNoPivot(ref U, ref L);


            AssertLU(in A, in U, in L);

            arena.Dispose();
        }

        public void LUDecompRandom()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.doubleRandomMatrix(dim, dim, 1f, 10f, 314221);
            var L = arena.doubleIdentityMatrix(dim);
            
            // add to diagonals of U
            for(int d = 0; d < dim; d++)
                U[d, d] += 5f;
            
            var A = U.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            //LU.luDecompositionNoPivot(ref U, ref L);
            LU.luDecomposition(ref U, ref L, ref pivot);

            var testMat = arena.doubleIdentityMatrix(dim);

            Print.Log(testMat);
            pivot.ApplyRow(ref testMat);
            Print.Log(testMat);
            
            pivot.ApplyInverseRow(ref testMat);
            Print.Log(testMat);

            Assert.IsTrue(Analysis.IsIdentity(testMat, 1E-05f));

            pivot.Dispose();
            /*Print.Log(A);
            Print.Log(U);
            Print.Log(L);
            */
            AssertLU(in A, in L, in U, 1E-05f);

            arena.Dispose();
        }

        public void LUDecompRandomLarge()
        {/*
            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var R = arena.doubleMat(dim);
            var U = arena.doubleRandomMatrix(dim * 2, dim, -5f, 5f, 9612221);

            var A = U.Copy();

            OrthoOP.LUDecomposition(ref U, ref R);

            AssertLU(in A, in U, in R, 1E-03f);

            arena.Dispose();*/
        }

        public void LUDecompHilbert()
        {/*
            var arena = new Arena(Allocator.Persistent);

            int dim = 20;

            var U = arena.doubleHilbertMatrix(dim);
            var R = arena.doubleMat(dim);

            var A = U.Copy();

            OrthoOP.LUDecomposition(ref U, ref R);

            //Print.Log(A);
            //Print.Log(U);
            //Print.Log(R);

            AssertLU(in A, in U, in R);

            arena.Dispose();*/
        }

        public void LUDecompPermutation() {
            /*
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

                var U = arena.doublePermutationMatrix(dim, p0, p1);

                p0 = rand.NextInt(0, dim);
                p1 = rand.NextInt(0, dim);

                while (p0 == p1) {
                    p1 = rand.NextInt(0, dim);
                }

                U = doubleOP.dot(arena.doublePermutationMatrix(dim, p0, p1), U);

                var R = arena.doubleMat(dim);

                var A = U.Copy();

                OrthoOP.LUDecomposition(ref U, ref R);

                //Print.Log(A);
                //Print.Log(U);
                //Print.Log(R);

                AssertLU(in A, in U, in R);
            }
            arena.Dispose();*/
        }

        public void LUDecompZero() {

            /*
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.doubleMat(dim, dim);
            var R = arena.doubleMat(dim);

            var A = U.Copy();

            OrthoOP.LUDecomposition(ref U, ref R);

            //Print.Log(A);
            //Print.Log(U);
            //Print.Log(R);

            AssertLU(in A, in U, in R);

            arena.Dispose();*/
        }

        private void AssertLU(in doubleMxN A, in doubleMxN L, in doubleMxN U) => AssertLU(in A, in L, in U, 1E-6f);
        private void AssertLU(in doubleMxN A, in doubleMxN L, in doubleMxN U, double precision)
        {
            doubleMxN shouldBeZero = A - doubleOP.dot(L, U);

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            Debug.Log($"Error of max(abs(A - LU)): {zeroError}");

            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.IsLowerTriangular(L, precision));
            Assert.IsTrue(Analysis.IsUpperTriangular(U, precision));
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

            /*var arena = new Arena(Allocator.Persistent);

            int tests = 64;
            double errorSum = 0;

            for (uint i = 0; i < tests; i++) {

                int dim = 32;

                doubleMxN A; 
                
                if(Type == TestType.RandomDiagonal)
                    A = arena.doubleRandomDiagonalMatrix(dim, 1f, 3f, 21410 + i*i + i*7);
                else
                    A = arena.doubleRandomMatrix(dim*2, dim, -25f, +25f, 21410 + i*i + i*7);
                
                var U = A.Copy();
                var R = arena.doubleMat(dim);

                OrthoOP.LUDecomposition(ref U, ref R);

                //Print.Log(U);
                //Print.Log(R);

                errorSum += ErrorCheckLU(in A, in U, in R);

                arena.Clear();
            }

            double avgError = errorSum / tests;

            Debug.Log($"Average error of max(abs(A - LU)): {avgError}");

            arena.Dispose();*/
        }

        private double ErrorCheckLU(in doubleMxN A, in doubleMxN Q, in doubleMxN R) {
            
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
            SquareFullRankDirect,
        }

        public TestType Type;

        public void Execute() {

            switch(Type) {
            
                case TestType.SquareFullRank:
                    SquareFullRank();
                break;
                case TestType.SquareFullRankDirect:
                    SquareFullRankDirect();
                break;
            }
        }

        void SquareFullRank() {
            /*
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

                var U = A.Copy();
                var R = arena.doubleMat(systemDim); 

                OrthoOP.LUDecomposition(ref U, ref R);

                for(uint j = 0; j < randomVecTests; j++) {

                    doubleN xOrig = arena.doubleRandomVector(systemDim, -25, +25, 1337 + i * i + i * 5);
                    doubleN b = doubleOP.dot(A, xOrig);
                    doubleN y = doubleOP.dot(b, U);

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

            arena.Dispose();*/
        }

        void OverdeterminedFullRank() {

            /*
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

                var U = A.Copy();
                var R = arena.doubleMat(sysDimN);

                OrthoOP.LUDecomposition(ref U, ref R);

                for (uint j = 0; j < randomVecTests; j++) {

                    doubleN xOrig = arena.doubleRandomVector(sysDimN, -25, +25, 1337 + i * i + i * 5);
                    doubleN b = doubleOP.dot(A, xOrig);
                    doubleN y = doubleOP.dot(b, U);

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
            */
        }

        void SquareFullRankDirect() {
            /*
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

                OrthoOP.LUDirectSolve(ref A, ref b, ref x);

                if (Analysis.IsAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }
                x.subInpl(xOrig);

                
                double zeroError = Analysis.MaxZeroError(x);
                
                errorSum += zeroError;
            }

            double avgError = errorSum / (randomMatTests);

            Debug.Log($"Average error: {avgError}");

            arena.Dispose();*/
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void LUDecompTests(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }

    [Test]
    public void LUDecompErrorBenchRandom() {
        new PrecisionReconstructTestJob() { Type = PrecisionReconstructTestJob.TestType.Random }.Run();
    }

    [Test]
    public void LUDecompErrorBenchDiagonal() {
        new PrecisionReconstructTestJob() { Type = PrecisionReconstructTestJob.TestType.RandomDiagonal }.Run();
    }

    [Test]
    public void LUDecompErrorSolveSquareSystem() {
        new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.SquareFullRank }.Run();
    }

    [Test]
    public void LUDecompErrorSolveSquareSystemDirect() {
        new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.SquareFullRankDirect }.Run(); 
    }

}
