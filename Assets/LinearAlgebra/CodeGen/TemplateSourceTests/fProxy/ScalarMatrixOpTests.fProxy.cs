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
public class fProxyScalarMatrixOpTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<fProxy> Fail;

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

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
            A[1, 0] = (fProxy)4; A[1, 1] = (fProxy)5;

            fProxyMxN R = (fProxy)0 / A;   // pre-fix this threw DivideByZeroException

            AssertClose(R[0, 0], (fProxy)0, (fProxy)1E-6);
            AssertClose(R[0, 1], (fProxy)0, (fProxy)1E-6);
            AssertClose(R[1, 0], (fProxy)0, (fProxy)1E-6);
            AssertClose(R[1, 1], (fProxy)0, (fProxy)1E-6);

            arena.Dispose();
        }

        // 5 - [[1,2],[3,4]] must be [[4,3],[2,1]] (NOT the negated [[-4,-3],[-2,-1]]).
        void ScalarMinusMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
            A[1, 0] = (fProxy)3; A[1, 1] = (fProxy)4;

            fProxyMxN R = (fProxy)5 - A;

            AssertClose(R[0, 0], (fProxy)4, (fProxy)1E-5);
            AssertClose(R[0, 1], (fProxy)3, (fProxy)1E-5);
            AssertClose(R[1, 0], (fProxy)2, (fProxy)1E-5);
            AssertClose(R[1, 1], (fProxy)1, (fProxy)1E-5);

            // A must be unchanged (operator works on a copy)
            AssertClose(A[0, 0], (fProxy)1, (fProxy)1E-6);

            arena.Dispose();
        }

        // 5 - [1,2,3] must be [4,3,2] (the vector form was already correct; guard against regression).
        void ScalarMinusVector()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(3);
            v[0] = (fProxy)1; v[1] = (fProxy)2; v[2] = (fProxy)3;

            fProxyN r = (fProxy)5 - v;

            AssertClose(r[0], (fProxy)4, (fProxy)1E-5);
            AssertClose(r[1], (fProxy)3, (fProxy)1E-5);
            AssertClose(r[2], (fProxy)2, (fProxy)1E-5);

            arena.Dispose();
        }

        // L3 norm of [-1, 2, -2] = (|−1|³+|2|³+|−2|³)^(1/3) = 17^(1/3) ≈ 2.5713 — finite, no NaN.
        // (Without abs, (-1)³+2³+(-2)³ = -1 then (-1)^(1/3) = NaN.)
        void NormalizeLPNegatives()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(3);
            v[0] = (fProxy)(-1); v[1] = (fProxy)2; v[2] = (fProxy)(-2);

            fProxy norm = fProxyNorms_OP.NormalizeLP(in v, (fProxy)3);

            if (math.isnan(norm))
            {
                Fail[0] = (fProxy)1; Fail[1] = norm; Fail[2] = (fProxy)2.5713; Fail[3] = norm;
            }
            AssertClose(norm, (fProxy)math.pow((fProxy)17, (fProxy)1 / (fProxy)3), (fProxy)1E-3);

            arena.Dispose();
        }

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ScalarMatrixOpTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
