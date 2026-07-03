using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

// Tests for the integer clampInpl<T> surface (int / short / long), vec + matrix.
// Semantics mirror the float clamp: below lo→lo, above hi→hi, in-range untouched.
// Passing lo > hi throws ArgumentException (eager validation, same as Cholesky / FFT guards).
public class iProxyClampTests
{
    [BurstCompile]
    public struct ClampTestJob : IJob
    {
        public enum TestType
        {
            ClampVector,
            ClampMatrix,
            ClampNoOpInRange,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ClampVector:      ClampVector(); break;
                case TestType.ClampMatrix:      ClampMatrix(); break;
                case TestType.ClampNoOpInRange: ClampNoOpInRange(); break;
            }
        }

        void ClampVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.iProxyVec(6, 0);
            v[0] = (iProxy)(-5); v[1] = (iProxy)(-2); v[2] = (iProxy)0;
            v[3] = (iProxy)2;    v[4] = (iProxy)7;    v[5] = (iProxy)2;

            iProxyElem_OP.clampInpl(in v, (iProxy)(-2), (iProxy)5);

            Assert.IsTrue(v[0] == (iProxy)(-2));
            Assert.IsTrue(v[1] == (iProxy)(-2));
            Assert.IsTrue(v[2] == (iProxy)0);
            Assert.IsTrue(v[3] == (iProxy)2);
            Assert.IsTrue(v[4] == (iProxy)5);
            Assert.IsTrue(v[5] == (iProxy)2);
            arena.Dispose();
        }

        void ClampMatrix()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.iProxyMat(2, 2, 0);
            A[0, 0] = (iProxy)(-10); A[0, 1] = (iProxy)3;
            A[1, 0] = (iProxy)5;     A[1, 1] = (iProxy)20;

            iProxyElem_OP.clampInpl(in A, (iProxy)0, (iProxy)10);

            Assert.IsTrue(A[0, 0] == (iProxy)0);
            Assert.IsTrue(A[0, 1] == (iProxy)3);
            Assert.IsTrue(A[1, 0] == (iProxy)5);
            Assert.IsTrue(A[1, 1] == (iProxy)10);
            arena.Dispose();
        }

        void ClampNoOpInRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.iProxyVec(4, 3); // all 3, inside [0,5]
            iProxyElem_OP.clampInpl(in v, (iProxy)0, (iProxy)5);
            for (int i = 0; i < 4; i++)
                Assert.IsTrue(v[i] == (iProxy)3);
            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(ClampTestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ClampCases(ClampTestJob.TestType type)
    {
        new ClampTestJob() { Type = type }.Run();
    }

    // lo > hi must throw ArgumentException — called directly on the test thread, not inside a Burst job.
    [Test]
    public void ClampLoGreaterThanHiThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.iProxyVec(3, 0);
        v[0] = (iProxy)(-4); v[1] = (iProxy)0; v[2] = (iProxy)9;
        Assert.Throws<ArgumentException>(() => iProxyElem_OP.clampInpl(in v, (iProxy)6, (iProxy)(-1)));
        arena.Dispose();
    }
}
