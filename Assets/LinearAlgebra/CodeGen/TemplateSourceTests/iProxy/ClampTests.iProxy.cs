using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

// Tests for the brand-new integer clampInpl<T> surface (int / short / long), vec + matrix.
// Semantics mirror the float clamp: below lo→lo, above hi→hi, in-range untouched.
//
// NOTE: the documented degenerate case (lo > hi → every element collapses to lo) currently
// FAILS for the integer kernel — see ClampLoGreaterThanHiCollapsesToLo below. The integer
// kernel (mathUnsafeiProxy.clamp) uses a `x > max ? max : x < min ? min : x` ternary chain
// that does NOT collapse to lo when lo > hi (e.g. value 9, lo 6, hi -1 → -1, not 6), unlike
// the float kernel (math.clamp = max(min, min(max,x))) and unlike the kernel's own docstring.
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

        // below lo→lo, at/in-range untouched, above hi→hi.
        void ClampVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.iProxyVec(6, 0);
            v[0] = (iProxy)(-5); v[1] = (iProxy)(-2); v[2] = (iProxy)0;
            v[3] = (iProxy)2;    v[4] = (iProxy)7;    v[5] = (iProxy)2;

            iProxyOP.clampInpl(in v, (iProxy)(-2), (iProxy)5);

            Assert.IsTrue(v[0] == (iProxy)(-2)); // below lo → lo
            Assert.IsTrue(v[1] == (iProxy)(-2)); // at lo
            Assert.IsTrue(v[2] == (iProxy)0);    // in range
            Assert.IsTrue(v[3] == (iProxy)2);    // in range
            Assert.IsTrue(v[4] == (iProxy)5);    // above hi → hi
            Assert.IsTrue(v[5] == (iProxy)2);    // in range untouched
            arena.Dispose();
        }

        void ClampMatrix()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.iProxyMat(2, 2, 0);
            A[0, 0] = (iProxy)(-10); A[0, 1] = (iProxy)3;
            A[1, 0] = (iProxy)5;     A[1, 1] = (iProxy)20;

            iProxyOP.clampInpl(in A, (iProxy)0, (iProxy)10);

            Assert.IsTrue(A[0, 0] == (iProxy)0);  // below lo
            Assert.IsTrue(A[0, 1] == (iProxy)3);  // in range
            Assert.IsTrue(A[1, 0] == (iProxy)5);  // in range
            Assert.IsTrue(A[1, 1] == (iProxy)10); // above hi
            arena.Dispose();
        }

        // Already in range → fully untouched.
        void ClampNoOpInRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.iProxyVec(4, 3); // all 3, inside [0,5]
            iProxyOP.clampInpl(in v, (iProxy)0, (iProxy)5);
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

    // Documented contract (per the clampInpl docstring AND the float sibling): when lo > hi every
    // element collapses to lo. Run on the main thread (managed) so the bug surfaces as a clean
    // NUnit assertion failure rather than a Burst-aborting exception.
    //
    // EXPECTED TO FAIL against the current integer kernel — this is an intentional bug-exposing
    // regression test (the production fix is to make mathUnsafeiProxy.clamp use
    // max(min, min(max, x)) like the float kernel). Do not weaken to the buggy behavior.
    [Test]
    public void ClampLoGreaterThanHiCollapsesToLo()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.iProxyVec(3, 0);
        v[0] = (iProxy)(-4); v[1] = (iProxy)0; v[2] = (iProxy)9;

        iProxyOP.clampInpl(in v, (iProxy)6, (iProxy)(-1)); // lo > hi
        for (int i = 0; i < 3; i++)
            Assert.AreEqual((iProxy)6, v[i], $"index {i}: lo>hi must collapse to lo (6)");

        arena.Dispose();
    }
}
