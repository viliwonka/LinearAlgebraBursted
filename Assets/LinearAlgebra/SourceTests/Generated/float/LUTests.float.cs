using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class floatLUTests
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

        private floatMxN GetRandomMatrix(ref Arena arena, int dim, float min, float max, uint seed) {

            var mat = arena.floatRandomMatrix(dim, dim, min, max, seed);

            return mat;
        }

        public void LUDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.floatIdentityMatrix(dim);
            var L = arena.floatIdentityMatrix(dim);

            var A = U.Copy();

            LU.luDecompositionNoPivot(ref U, ref L);

            
            AssertLU(in A, in L, in U);

            arena.Dispose();
        }


        public void LUDecompRandomDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.floatRandomDiagonalMatrix(dim, 1f, 3f);
            var L = arena.floatIdentityMatrix(dim);

            var A = U.Copy();

            LU.luDecompositionNoPivot(ref U, ref L);


            AssertLU(in A, in U, in L);

            arena.Dispose();
        }

        public void LUDecompRandom()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.floatRandomMatrix(dim, dim, 1f, 10f, 314221);
            var L = arena.floatIdentityMatrix(dim);
            
            // add to diagonals of U
            for(int d = 0; d < dim; d++)
                U[d, d] += 5f;
            
            var A = U.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            //LU.luDecompositionNoPivot(ref U, ref L);
            LU.luDecomposition(ref U, ref L, ref pivot);

            var testMat = arena.floatIdentityMatrix(dim);

            Print.Log(testMat);
            pivot.ApplyRow<floatMxN, float>(ref testMat);
            Print.Log(testMat);
            
            pivot.ApplyInverseRow<floatMxN, float>(ref testMat);
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

            var R = arena.floatMat(dim);
            var U = arena.floatRandomMatrix(dim * 2, dim, -5f, 5f, 9612221);

            var A = U.Copy();

            OrthoOP.LUDecomposition(ref U, ref R);

            AssertLU(in A, in U, in R, 1E-03f);

            arena.Dispose();*/
        }

        public void LUDecompHilbert()
        {/*
            var arena = new Arena(Allocator.Persistent);

            int dim = 20;

            var U = arena.floatHilbertMatrix(dim);
            var R = arena.floatMat(dim);

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

                var U = arena.floatPermutationMatrix(dim, p0, p1);

                p0 = rand.NextInt(0, dim);
                p1 = rand.NextInt(0, dim);

                while (p0 == p1) {
                    p1 = rand.NextInt(0, dim);
                }

                U = floatOP.dot(arena.floatPermutationMatrix(dim, p0, p1), U);

                var R = arena.floatMat(dim);

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

            var U = arena.floatMat(dim, dim);
            var R = arena.floatMat(dim);

            var A = U.Copy();

            OrthoOP.LUDecomposition(ref U, ref R);

            //Print.Log(A);
            //Print.Log(U);
            //Print.Log(R);

            AssertLU(in A, in U, in R);

            arena.Dispose();*/
        }

        private void AssertLU(in floatMxN A, in floatMxN L, in floatMxN U) => AssertLU(in A, in L, in U, 1E-6f);
        private void AssertLU(in floatMxN A, in floatMxN L, in floatMxN U, float precision)
        {
            floatMxN shouldBeZero = A - floatOP.dot(L, U);

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
            float errorSum = 0;

            for (uint i = 0; i < tests; i++) {

                int dim = 32;

                floatMxN A; 
                
                if(Type == TestType.RandomDiagonal)
                    A = arena.floatRandomDiagonalMatrix(dim, 1f, 3f, 21410 + i*i + i*7);
                else
                    A = arena.floatRandomMatrix(dim*2, dim, -25f, +25f, 21410 + i*i + i*7);
                
                var U = A.Copy();
                var R = arena.floatMat(dim);

                OrthoOP.LUDecomposition(ref U, ref R);

                //Print.Log(U);
                //Print.Log(R);

                errorSum += ErrorCheckLU(in A, in U, in R);

                arena.Clear();
            }

            float avgError = errorSum / tests;

            Debug.Log($"Average error of max(abs(A - LU)): {avgError}");

            arena.Dispose();*/
        }

        private float ErrorCheckLU(in floatMxN A, in floatMxN Q, in floatMxN R) {
            
            floatMxN shouldBeZero = A - floatOP.dot(Q, R);

            if(Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("PrecisionReconstructTestJob: NaN detected");

            //Print.Log(shouldBeZero); 

            float zeroError = Analysis.MaxZeroError(shouldBeZero);

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
            float errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                floatMxN A = arena.floatRandomMatrix(systemDim, systemDim, -5, +5, 420 + i - i + i * 7);

                for(int d = 0; d < systemDim; d++)
                    A[d, d] += 5.1f + 10f*random.NextFloat();

                var U = A.Copy();
                var R = arena.floatMat(systemDim); 

                OrthoOP.LUDecomposition(ref U, ref R);

                for(uint j = 0; j < randomVecTests; j++) {

                    floatN xOrig = arena.floatRandomVector(systemDim, -25, +25, 1337 + i * i + i * 5);
                    floatN b = floatOP.dot(A, xOrig);
                    floatN y = floatOP.dot(b, U);

                    Solvers.SolveUpperTriangular(ref R, ref y);

                    y.subInpl(xOrig);
                    float zeroError = Analysis.MaxZeroError(y);

                    if(Analysis.IsAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    errorSum += zeroError;
                }
            }

            float avgError = errorSum / (randomMatTests*randomVecTests);

            Debug.Log($"Average error: {avgError}");

            arena.Dispose();*/
        }

        void OverdeterminedFullRank() {

            /*
            int sysDimM = 128;
            int sysDimN = 64;
            int randomMatTests = 32;
            int randomVecTests = 16;
            float errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                var arena = new Arena(Allocator.Persistent);
                floatMxN A = arena.floatRandomMatrix(sysDimM, sysDimN, -5, +5, 420 + i - i + i * 7);

                for (int d = 0; d < sysDimN; d++)
                    A[d, d] += 5.1f + 10f * random.NextFloat();

                var U = A.Copy();
                var R = arena.floatMat(sysDimN);

                OrthoOP.LUDecomposition(ref U, ref R);

                for (uint j = 0; j < randomVecTests; j++) {

                    floatN xOrig = arena.floatRandomVector(sysDimN, -25, +25, 1337 + i * i + i * 5);
                    floatN b = floatOP.dot(A, xOrig);
                    floatN y = floatOP.dot(b, U);

                    Solvers.SolveUpperTriangular(ref R, ref y);

                    y.subInpl(xOrig);
                    float zeroError = Analysis.MaxZeroError(y);

                    if (Analysis.IsAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    errorSum += zeroError;
                }
                arena.Dispose();
            }

            float avgError = errorSum / (randomMatTests * randomVecTests);

            Debug.Log($"Average error: {avgError}");
            */
        }

        void SquareFullRankDirect() {
            /*
            var arena = new Arena(Allocator.Persistent);

            int systemDim = 128;  
            int randomMatTests = 128;
            float errorSum = 0;
             
            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                floatMxN A = arena.floatRandomMatrix(systemDim, systemDim, -5, +5, 420 + i - i + i * 7);

                for (int d = 0; d < systemDim; d++)
                    A[d, d] += 5.1f + 10f * random.NextFloat();

                floatN xOrig = arena.floatRandomVector(systemDim, -25, +25, 1337 + i * i + i * 5);
                floatN b = floatOP.dot(A, xOrig);
                floatN x = arena.floatVec(systemDim);

                OrthoOP.LUDirectSolve(ref A, ref b, ref x);

                if (Analysis.IsAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }
                x.subInpl(xOrig);

                
                float zeroError = Analysis.MaxZeroError(x);
                
                errorSum += zeroError;
            }

            float avgError = errorSum / (randomMatTests);

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
