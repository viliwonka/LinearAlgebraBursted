using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

// Tests for the integer clampInPlace<T> surface (int / short / long), vec + matrix.
// Semantics mirror the float clamp: below lo→lo, above hi→hi, in-range untouched.
// Passing lo > hi throws ArgumentException (eager validation, same as Cholesky / FFT guards).
public class intClampTests
{
    [BurstCompile(CompileSynchronously = true)]
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
            var v = arena.intVec(6, 0);
            v[0] = (int)(-5); v[1] = (int)(-2); v[2] = (int)0;
            v[3] = (int)2;    v[4] = (int)7;    v[5] = (int)2;

            intComp.clampInPlace(v, (int)(-2), (int)5);

            Assert.IsTrue(v[0] == (int)(-2));
            Assert.IsTrue(v[1] == (int)(-2));
            Assert.IsTrue(v[2] == (int)0);
            Assert.IsTrue(v[3] == (int)2);
            Assert.IsTrue(v[4] == (int)5);
            Assert.IsTrue(v[5] == (int)2);
            arena.Dispose();
        }

        void ClampMatrix()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.intMat(2, 2, 0);
            A[0, 0] = (int)(-10); A[0, 1] = (int)3;
            A[1, 0] = (int)5;     A[1, 1] = (int)20;

            intComp.clampInPlace(A, (int)0, (int)10);

            Assert.IsTrue(A[0, 0] == (int)0);
            Assert.IsTrue(A[0, 1] == (int)3);
            Assert.IsTrue(A[1, 0] == (int)5);
            Assert.IsTrue(A[1, 1] == (int)10);
            arena.Dispose();
        }

        void ClampNoOpInRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.intVec(4, 3); // all 3, inside [0,5]
            intComp.clampInPlace(v, (int)0, (int)5);
            for (int i = 0; i < 4; i++)
                Assert.IsTrue(v[i] == (int)3);
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
        var v = arena.intVec(3, 0);
        v[0] = (int)(-4); v[1] = (int)0; v[2] = (int)9;
        Assert.Throws<ArgumentException>(() => intComp.clampInPlace(v, (int)6, (int)(-1)));
        arena.Dispose();
    }
}
