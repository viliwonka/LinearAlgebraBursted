using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Tests for the integer scalar-predicate subset of the QueryOP extension (longQuery_OP, Group A
// only). Spec: docs/spec-predicate-queries.md (Section 4b + T1). Groups B/C/D are fProxy-only.
//
// One template expands to int / short / long QueryOP, so every literal must be exact AND safe for
// the tightest type (short): coordinates stay small. Functor struct is NESTED in the outer class
// so the generated int / short / long files do not collide on a namespace-scope type name.
//
// Burst-compatible computational tests live in TestJob; the managed-throw guard is a plain [Test].
public class longQueryPredicateTests
{
    struct GreaterThanInt : IlongPredicate
    {
        public long t;
        public bool Test(long x) => x > t;
    }

    [BurstCompile]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            GroupAScalar,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<long> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.GroupAScalar: GroupAScalar(); break;
            }
        }

        // ---------------------------------------------------------------------
        // GROUP A — FLAT / SCALAR PREDICATE OPS (T1, integer)
        // ---------------------------------------------------------------------

        void GroupAScalar()
        {
            var arena = new Arena(Allocator.Persistent);

            // v = [-2, 0, 3, 1, 4, 2]; threshold 2 -> {3@2, 4@4} pass.
            var v = arena.longVec(6);
            v[0] = (long)(-2); v[1] = (long)0; v[2] = (long)3;
            v[3] = (long)1;    v[4] = (long)4; v[5] = (long)2;

            var pass = new GreaterThanInt { t = (long)2 };
            AssertEqI(longQuery_OP.findFirst(in v, ref pass), 2);
            AssertEqI(longQuery_OP.count(in v, ref pass), 2);
            AssertTrue(longQuery_OP.any(in v, ref pass));
            // not all > 2 (e.g. the -2 fails) -> all == false.
            AssertTrue(!longQuery_OP.all(in v, ref pass));

            var idx = arena.Indices(6);
            int fc = longQuery_OP.findAll(in v, ref pass, ref idx);
            AssertEqI(fc, 2);
            AssertEqI(idx[0], 2); AssertEqI(idx[1], 4);
            // findAll count == count.
            AssertEqI(fc, longQuery_OP.count(in v, ref pass));

            // No element matches -> findFirst -1, count 0, any false, findAll 0.
            var none = new GreaterThanInt { t = (long)100 };
            AssertEqI(longQuery_OP.findFirst(in v, ref none), -1);
            AssertEqI(longQuery_OP.count(in v, ref none), 0);
            AssertTrue(!longQuery_OP.any(in v, ref none));
            AssertEqI(longQuery_OP.findAll(in v, ref none, ref idx), 0);

            // Every element passes -> all true, any true.
            var allPass = new GreaterThanInt { t = (long)(-10) };
            AssertTrue(longQuery_OP.all(in v, ref allPass));
            AssertTrue(longQuery_OP.any(in v, ref allPass));

            // Empty vector: findFirst -1, count 0, any false, all true (vacuous), findAll 0.
            var v0 = arena.longVec(0);
            AssertEqI(longQuery_OP.findFirst(in v0, ref pass), -1);
            AssertEqI(longQuery_OP.count(in v0, ref pass), 0);
            AssertTrue(!longQuery_OP.any(in v0, ref pass));
            AssertTrue(longQuery_OP.all(in v0, ref pass));
            var idx0 = arena.Indices(1);
            AssertEqI(longQuery_OP.findAll(in v0, ref pass, ref idx0), 0);

            // Matrix flat-index variant (generic T over longMxN, row-major flat order).
            // A = [1 5; 2 5] -> flat [1,5,2,5]; threshold 4 -> {5@1, 5@3}.
            var A = arena.longMat(2, 2);
            A[0, 0] = (long)1; A[0, 1] = (long)5;
            A[1, 0] = (long)2; A[1, 1] = (long)5;
            var matPass = new GreaterThanInt { t = (long)4 };
            AssertEqI(longQuery_OP.findFirst(in A, ref matPass), 1);
            AssertEqI(longQuery_OP.count(in A, ref matPass), 2);
            var idxM = arena.Indices(4);
            int mc = longQuery_OP.findAll(in A, ref matPass, ref idxM);
            AssertEqI(mc, 2);
            AssertEqI(idxM[0], 1); AssertEqI(idxM[1], 3);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // helpers (integer ops are exact — no tolerance)
        // ---------------------------------------------------------------------

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
        void AssertEqI(int got, int expected)
        {
            if (got != expected && Fail[0] == (long)0)
            {
                Fail[0] = (long)1;
                Fail[1] = (long)got;
                Fail[2] = (long)expected;
                Fail[3] = (long)(got - expected);
            }
            Assert.AreEqual(expected, got);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (long)0)
            {
                Fail[0] = (long)1;
                Fail[1] = (long)(-1);
                Fail[2] = (long)(-1);
                Fail[3] = (long)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<long>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (long)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (long)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void GroupAScalarTest() => RunJob(TestJob.TestType.GroupAScalar);

    // -------------------------------------------------------------------------
    // Managed-throw guard (main thread): undersized Indices on findAll.
    // -------------------------------------------------------------------------

    [Test]
    public void FindAllUndersizedThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.longVec(5);
        var gt = new GreaterThanInt { t = (long)0 };
        var small = arena.Indices(4);   // < v.Data.Length (5)
        Assert.Throws<ArgumentException>(() => longQuery_OP.findAll(in v, ref gt, ref small));
        arena.Dispose();
    }
}
