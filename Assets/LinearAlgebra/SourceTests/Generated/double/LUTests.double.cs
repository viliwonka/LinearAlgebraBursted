using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class doubleLUTests
{

    //[BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LUDecompIdentity,
            LUDecompPredefined,
            LUDecompRandomDiagonal,
            LUDecompRandom,
            LUDecompRandomLarge,
            LUDecompHilbert,
            LUDecompPermutation,
            LUDecompZero,
            LUSolveSystem,
        }

        public TestType Type;


        public void Execute()
        {
            switch(Type)
            {
                case TestType.LUDecompIdentity:
                    LUDecompIdentity();
                break;
                case TestType.LUDecompPredefined:
                    LUDecompPredefined();
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
                case TestType.LUSolveSystem:
                    SolveSystem();
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

            AssertLU(in A, in L, in U, false);

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


            AssertLU(in A, in U, in L, false);

            arena.Dispose();
        }

        public void LUDecompPredefined() {

            var arena = new Arena(Allocator.Persistent);

            var dim = 5;

            var U = arena.doubleMat(dim);
            var L = arena.doubleIdentityMatrix(dim);
            
            var pivot = new Pivot(dim, Allocator.Temp);

            U[0] = -2f;
            U[1] = 1f;
            U[2] = -2f;
            U[3] = 3f;
            U[4] = 1f;

            U[5] = 1f;
            U[6] = -2f;
            U[7] = 3f;
            U[8] = -5f;
            U[9] = 4f;

            U[10] = 4f;
            U[11] = 3f;
            U[12] = -1f;
            U[13] = 2f;
            U[14] = -3f;

            U[15] = 1f;
            U[16] = 1f;
            U[17] = -1f;
            U[18] = -11f;
            U[19] = 11f;

            U[20] = -1f;
            U[21] = -9f;
            U[22] = -1f;
            U[23] = 7f;
            U[24] = 1f;



            Print.Log(L);
            Print.Log(U);

            //LU.luDecompositionNoPivot(ref U, ref L);
            LU.luDecomposition(ref U, ref L, ref pivot);

            Print.Log(L);
            Print.Log(U);


            pivot.Dispose();

            arena.Dispose();
        }

        public void LUDecompRandom()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 18;

            var U = arena.doubleRandomMatrix(dim, dim, 1f, 10f, 314221);
            var L = arena.doubleIdentityMatrix(dim);
            
            // add to diagonals of U
            for(int d = 0; d < dim; d++)
                U[d, d] += 5f;
            
            var A = U.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            //LU.luDecompositionNoPivot(ref U, ref L);
            LU.luDecomposition(ref U, ref L, ref pivot);

            pivot.ApplyInverseRow(ref A);

            Print.Log(U);
            Print.Log(L);

            pivot.Dispose();

            AssertLU(in A, in L, in U, true, 1E-05f);

            arena.Dispose();
        }

        public void LUDecompRandomLarge()
        {
            /*
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
        {
            /*
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

        public void SolveSystem() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 18;

            var A = arena.doubleRandomMatrix(dim, dim, 1f, 10f, 314221);
            // add to diagonals of U
            for (int d = 0; d < dim; d++)
                A[d, d] += 5f;


            var x_Known = arena.doubleRandomVector(dim, 1f, 10f, 901);
            
            var b = doubleOP.dot(A, x_Known);

            var U = A.Copy();
            var L = arena.doubleIdentityMatrix(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            // no pivot version works fine. Either pivot struct is broken or the pivot-LU version is broken
            //LU.luDecompositionNoPivot(ref U, ref L);
            LU.luDecomposition(ref U, ref L, ref pivot);

            var x_Solved = b.Copy();

            LU.LUSolve(ref L, ref U, in pivot, ref x_Solved);

            var zeroError = Analysis.MaxZeroError(x_Known - x_Solved);

            Debug.Log($"Error of max(abs(x_Known - x_Solved)): {zeroError}");


            Assert.IsTrue(zeroError < 1E-06f);

            pivot.Dispose();

            arena.Dispose();
        }

        private void AssertLU(in doubleMxN A, in doubleMxN L, in doubleMxN U, bool pivoted) => AssertLU(in A, in L, in U, pivoted, 1E-6f);
        private void AssertLU(in doubleMxN A, in doubleMxN L, in doubleMxN U, bool pivoted, double precision)
        {
            doubleMxN shouldBeZero = A - doubleOP.dot(L, U);

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            Debug.Log($"Error of max(abs(A - LU)): {zeroError}");

            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.IsLowerTriangular(L, precision));
            Assert.IsTrue(Analysis.IsUpperTriangular(U, precision));

            if(pivoted)
            unsafe {
                var maxAbs = LinearAlgebra.UnsafeOP.maxAbs(L.Data.Ptr, L.Length);

                if(maxAbs > 1f)
                    throw new System.Exception("TestJob: L has values greater than 1f");
            }
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

}
