using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

// Tests for the integer clampInpl<T> surface (int / short / long), vec + matrix.
// Semantics mirror the float clamp: below lo→lo, above hi→hi, in-range untouched.
// Passing lo > hi throws ArgumentException (eager validation, same as Cholesky / FFT guards).
public class longClampTests
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

        // below lo→lo, at/in-range untouched, above hi→hi.
        void ClampVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.longVec(6, 0);
            v[0] = (long)(-5); v[1] = (long)(-2); v[2] = (long)0;
            v[3] = (long)2;    v[4] = (long)7;    v[5] = (long)2;

            longElem_OP.clampInpl(in v, (long)(-2), (long)5);

            Assert.IsTrue(v[0] == (long)(-2)); // below lo → lo
            Assert.IsTrue(v[1] == (long)(-2)); // at lo
            Assert.IsTrue(v[2] == (long)0);    // in range
            Assert.IsTrue(v[3] == (long)2);    // in range
            Assert.IsTrue(v[4] == (long)5);    // above hi → hi
            Assert.IsTrue(v[5] == (long)2);    // in range untouched
            arena.Dispose();
        }

        void ClampMatrix()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.longMat(2, 2, 0);
            A[0, 0] = (long)(-10); A[0, 1] = (long)3;
            A[1, 0] = (long)5;     A[1, 1] = (long)20;

            longElem_OP.clampInpl(in A, (long)0, (long)10);

            Assert.IsTrue(A[0, 0] == (long)0);  // below lo
            Assert.IsTrue(A[0, 1] == (long)3);  // in range
            Assert.IsTrue(A[1, 0] == (long)5);  // in range
            Assert.IsTrue(A[1, 1] == (long)10); // above hi
            arena.Dispose();
        }

        // Already in range → fully untouched.
        void ClampNoOpInRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.longVec(4, 3); // all 3, inside [0,5]
            longElem_OP.clampInpl(in v, (long)0, (long)5);
            for (int i = 0; i < 4; i++)
                Assert.IsTrue(v[i] == (long)3);
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
        var v = arena.longVec(3, 0);
        v[0] = (long)(-4); v[1] = (long)0; v[2] = (long)9;
        Assert.Throws<ArgumentException>(() => longElem_OP.clampInpl(in v, (long)6, (long)(-1)));
        arena.Dispose();
    }
}
