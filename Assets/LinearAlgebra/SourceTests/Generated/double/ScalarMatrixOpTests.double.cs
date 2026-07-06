using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Regression tests for review-found bugs:
//  - `scalar - matrix` returned `matrix - scalar` (negated) because subtraction was wrongly
//    treated as commutative (operator delegated to `rhs - lhs`).
//  - `normalizeLP` summed pow(x,p) without abs -> NaN for negative entries with non-even p.
public class doubleScalarMatrixOpTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ScalarMinusMatrix,
            ScalarMinusVector,
            NormalizeLPNegatives,
            ZeroDivMatrix,
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ScalarMinusMatrix:    ScalarMinusMatrix(); break;
                case TestType.ScalarMinusVector:    ScalarMinusVector(); break;
                case TestType.NormalizeLPNegatives: NormalizeLPNegatives(); break;
                case TestType.ZeroDivMatrix:        ZeroDivMatrix(); break;
            }
        }

        // 0 / M is a valid operation (= 0 where M != 0); it must NOT throw (review fix D).
        void ZeroDivMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)2;
            A[1, 0] = (double)4; A[1, 1] = (double)5;

            doubleMxN R = (double)0 / A;   // pre-fix this threw DivideByZeroException

            AssertClose(R[0, 0], (double)0, (double)1E-6);
            AssertClose(R[0, 1], (double)0, (double)1E-6);
            AssertClose(R[1, 0], (double)0, (double)1E-6);
            AssertClose(R[1, 1], (double)0, (double)1E-6);

            arena.Dispose();
        }

        // 5 - [[1,2],[3,4]] must be [[4,3],[2,1]] (NOT the negated [[-4,-3],[-2,-1]]).
        void ScalarMinusMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)2;
            A[1, 0] = (double)3; A[1, 1] = (double)4;

            doubleMxN R = (double)5 - A;

            AssertClose(R[0, 0], (double)4, (double)1E-5);
            AssertClose(R[0, 1], (double)3, (double)1E-5);
            AssertClose(R[1, 0], (double)2, (double)1E-5);
            AssertClose(R[1, 1], (double)1, (double)1E-5);

            // A must be unchanged (operator works on a copy)
            AssertClose(A[0, 0], (double)1, (double)1E-6);

            arena.Dispose();
        }

        // 5 - [1,2,3] must be [4,3,2] (the vector form was already correct; guard against regression).
        void ScalarMinusVector()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(3);
            v[0] = (double)1; v[1] = (double)2; v[2] = (double)3;

            doubleN r = (double)5 - v;

            AssertClose(r[0], (double)4, (double)1E-5);
            AssertClose(r[1], (double)3, (double)1E-5);
            AssertClose(r[2], (double)2, (double)1E-5);

            arena.Dispose();
        }

        // L3 norm of [-1, 2, -2] = (|−1|³+|2|³+|−2|³)^(1/3) = 17^(1/3) ≈ 2.5713 — finite, no NaN.
        // (Without abs, (-1)³+2³+(-2)³ = -1 then (-1)^(1/3) = NaN.)
        void NormalizeLPNegatives()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(3);
            v[0] = (double)(-1); v[1] = (double)2; v[2] = (double)(-2);

            double norm = Norms.normalizeLP(in v, (double)3);

            if (math.isnan(norm))
            {
                Fail[0] = (double)1; Fail[1] = norm; Fail[2] = (double)2.5713; Fail[3] = norm;
            }
            AssertClose(norm, (double)math.pow((double)17, (double)1 / (double)3), (double)1E-3);

            arena.Dispose();
        }

        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ScalarMatrixOpTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
