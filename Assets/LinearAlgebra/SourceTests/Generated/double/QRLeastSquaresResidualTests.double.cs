using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Least-squares residual tests for the (un-pivoted) Householder QR solver QR.qrDirectSolve.
// The existing QR solve tests only cover CONSISTENT right-hand sides (b = A*xOrig, zero residual).
// These add genuinely INCONSISTENT overdetermined systems, where min ||Ax - b|| has a non-zero
// residual r = b - Ax, and verify the defining least-squares property: r is orthogonal to the
// column space of A, i.e. the normal equations Aᵀ(Ax - b) = 0 hold.
//
// Test vector: the classic Strang least-squares example (Gilbert Strang, "Introduction to Linear
// Algebra", best-fit-line section): A = [[1,0],[1,1],[1,2]], b = [6,0,0] has best fit x = [5,-3]
// with residual r = [1,-2,1] (||r||^2 = 6) and Aᵀr = 0.
public class doubleQRLeastSquaresResidualTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            StrangBestFitLine,
            RandomOverdeterminedNormalEquations,
            RandomOverdeterminedOptimality,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.StrangBestFitLine:                   StrangBestFitLine();                   break;
                case TestType.RandomOverdeterminedNormalEquations: RandomOverdeterminedNormalEquations(); break;
                case TestType.RandomOverdeterminedOptimality:      RandomOverdeterminedOptimality();      break;
            }
        }

        // Known closed-form answer: x = [5, -3], residual r = [1, -2, 1], Aᵀr = 0.
        void StrangBestFitLine()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 0f;
            A[1, 0] = 1f; A[1, 1] = 1f;
            A[2, 0] = 1f; A[2, 1] = 2f;

            var b = arena.doubleVec(3);
            b[0] = 6f; b[1] = 0f; b[2] = 0f;

            var Awork = A.Copy();   // qrDirectSolve destroys A and b
            var bwork = b.Copy();
            var x = arena.doubleVec(2);

            QR.qrDirectSolve(ref Awork, ref bwork, ref x);

            if (Analysis.isAnyNan(in x))
                throw new System.Exception("TestJob: NaN detected");

            RecordBound(math.abs(x[0] - (double)5f), (double)1E-4f);
            RecordBound(math.abs(x[1] - (double)(-3f)), (double)1E-4f);

            doubleN r = b - Blas.dot(A, x);
            RecordBound(math.abs(r[0] - (double)1f), (double)1E-4f);
            RecordBound(math.abs(r[1] - (double)(-2f)), (double)1E-4f);
            RecordBound(math.abs(r[2] - (double)1f), (double)1E-4f);

            // normal equations: Aᵀr == 0
            doubleN AtR = Blas.dot(r, A);
            RecordBound(Analysis.MaxZeroError(AtR), (double)1E-4f);

            arena.Dispose();
        }

        // For random inconsistent tall systems, the QR least-squares solution must make the residual
        // orthogonal to every column of A (normal equations), to within conditioning.
        void RandomOverdeterminedNormalEquations()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 24; t++)
            {
                int m = 24, n = 6;
                var A = arena.doubleRandomMat(m, n, -2f, 2f, 5500 + t * 17);
                // b is independent random — generically NOT in the column space => non-zero residual.
                var b = arena.doubleRandomVec(m, -2f, 2f, 99000 + t * 23);

                var Awork = A.Copy();
                var bwork = b.Copy();
                var x = arena.doubleVec(n);

                QR.qrDirectSolve(ref Awork, ref bwork, ref x);

                if (Analysis.isAnyNan(in x))
                    throw new System.Exception("TestJob: NaN detected");

                doubleN r = b - Blas.dot(A, x);
                doubleN AtR = Blas.dot(r, A);

                // scale-relative: ||Aᵀr||_inf small vs ||Aᵀb||_inf (the un-projected scale).
                doubleN AtB = Blas.dot(b, A);
                double scale = Analysis.MaxZeroError(AtB) + (double)1f;
                RecordBound(Analysis.MaxZeroError(AtR), (double)1E-3f * scale);

                // sanity: the residual is genuinely non-zero (this is an inconsistent system).
                double rNorm = doubleNorms_OP.L2(in r);
                if (!(rNorm > (double)1E-2f) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = rNorm;
                    Fail[2] = (double)1E-2f;
                    Fail[3] = rNorm;
                }

                arena.Clear();
            }

            arena.Dispose();
        }

        // The LS solution is the minimizer: perturbing x in any coordinate must not reduce the
        // residual sum of squares.
        void RandomOverdeterminedOptimality()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 12; t++)
            {
                int m = 16, n = 4;
                var A = arena.doubleRandomMat(m, n, -2f, 2f, 1200 + t * 11);
                var b = arena.doubleRandomVec(m, -2f, 2f, 64000 + t * 29);

                var Awork = A.Copy();
                var bwork = b.Copy();
                var x = arena.doubleVec(n);

                QR.qrDirectSolve(ref Awork, ref bwork, ref x);

                if (Analysis.isAnyNan(in x))
                    throw new System.Exception("TestJob: NaN detected");

                double r0 = SumSq(b - Blas.dot(A, x));

                double delta = (double)0.05f;
                for (int k = 0; k < n; k++)
                {
                    double saved = x[k];

                    x[k] = saved + delta;
                    double rp = SumSq(b - Blas.dot(A, x));
                    x[k] = saved - delta;
                    double rm = SumSq(b - Blas.dot(A, x));
                    x[k] = saved;

                    // both perturbations must be >= the optimum (minus tiny float slack).
                    double slack = (double)1E-4f * (r0 + (double)1f);
                    if (!(rp + slack >= r0 && rm + slack >= r0) && Fail[0] == (double)0)
                    {
                        Fail[0] = (double)1;
                        Fail[1] = math.min(rp, rm);
                        Fail[2] = r0;
                        Fail[3] = math.min(rp, rm) - r0;
                    }
                }

                arena.Clear();
            }

            arena.Dispose();
        }

        static double SumSq(in doubleN v)
        {
            double s = 0;
            for (int i = 0; i < v.N; i++)
                s += v[i] * v[i];
            return s;
        }

        void RecordBound(double value, double limit)
        {
            if (!(value <= limit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
            Assert.IsTrue(value <= limit);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void LeastSquaresResidualTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }
}
