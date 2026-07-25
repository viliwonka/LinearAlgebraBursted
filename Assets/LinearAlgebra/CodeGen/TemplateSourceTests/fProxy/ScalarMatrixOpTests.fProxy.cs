using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// scalar - matrix must equal s - A[i,j] elementwise (subtraction is not commutative).
// normalizeLP must sum pow(|x|,p) (abs), not pow(x,p), so negative entries with non-even p
// don't produce NaN.
public class fProxyScalarMatrixOpTests
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

        // 0 / M is a valid operation (= 0 where M != 0); it must NOT throw.
        void ZeroDivMatrix()
        {
            var A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
            A[1, 0] = (fProxy)4; A[1, 1] = (fProxy)5;

            fProxyMxN R = A.Copy();
            fProxyComp.divInPlace((fProxy)0, R);   // must not throw DivideByZeroException

            AssertClose(R[0, 0], (fProxy)0, (fProxy)1E-6);
            AssertClose(R[0, 1], (fProxy)0, (fProxy)1E-6);
            AssertClose(R[1, 0], (fProxy)0, (fProxy)1E-6);
            AssertClose(R[1, 1], (fProxy)0, (fProxy)1E-6);
        }

        // 5 - [[1,2],[3,4]] must be [[4,3],[2,1]] (NOT the negated [[-4,-3],[-2,-1]]).
        void ScalarMinusMatrix()
        {
            var A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
            A[1, 0] = (fProxy)3; A[1, 1] = (fProxy)4;

            fProxyMxN R = A.Copy();
            fProxyComp.subInPlace((fProxy)5, R);

            AssertClose(R[0, 0], (fProxy)4, (fProxy)1E-5);
            AssertClose(R[0, 1], (fProxy)3, (fProxy)1E-5);
            AssertClose(R[1, 0], (fProxy)2, (fProxy)1E-5);
            AssertClose(R[1, 1], (fProxy)1, (fProxy)1E-5);

            // A must be unchanged (operates on a copy)
            AssertClose(A[0, 0], (fProxy)1, (fProxy)1E-6);
        }

        // 5 - [1,2,3] must be [4,3,2] (the vector form was already correct; guard against regression).
        void ScalarMinusVector()
        {
            var v = new fProxyN(3, Allocator.Temp);
            v[0] = (fProxy)1; v[1] = (fProxy)2; v[2] = (fProxy)3;

            fProxyN r = v.Copy();
            fProxyComp.subInPlace((fProxy)5, r);

            AssertClose(r[0], (fProxy)4, (fProxy)1E-5);
            AssertClose(r[1], (fProxy)3, (fProxy)1E-5);
            AssertClose(r[2], (fProxy)2, (fProxy)1E-5);
        }

        // L3 norm of [-1, 2, -2] = (|−1|³+|2|³+|−2|³)^(1/3) = 17^(1/3) ≈ 2.5713 — finite, no NaN.
        // (Without abs, (-1)³+2³+(-2)³ = -1 then (-1)^(1/3) = NaN.)
        void NormalizeLPNegatives()
        {
            var v = new fProxyN(3, Allocator.Temp);
            v[0] = (fProxy)(-1); v[1] = (fProxy)2; v[2] = (fProxy)(-2);

            fProxy norm = Norms.normalizeLP(in v, (fProxy)3);

            if (math.isnan(norm))
            {
                Fail[0] = (fProxy)1; Fail[1] = norm; Fail[2] = (fProxy)2.5713; Fail[3] = norm;
            }
            AssertClose(norm, (fProxy)math.pow((fProxy)17, (fProxy)1 / (fProxy)3), (fProxy)1E-3);
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
