using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for the integer uniform-refill core (Rand.nextUniformInPlace), expanded to
// int / short / long. Range is [min, max) per Unity NextInt; min == max is a constant fill
// with NO rng advance; min > max throws.
//
// One template expands to Rand / Rand / Rand, so every literal must
// be exact AND safe for the TIGHTEST type (short, [-32768, 32767]): all bounds and poison
// values are kept small. The long-only out-of-int-range guard cannot be exercised here (the
// proxy and the int/short expansions can never hold a value outside int range); it is covered
// by the concretely-long RandomLongRangeTests in the BurstLinearAlgebra.Tests assembly.
//
// Burst-compatible computational tests live in TestJob (message-free asserts + Fail-buffer
// diagnostics); managed-throw guards are plain [Test] methods on the main thread, mirroring
// the Chunk-1 RandomTests.fProxy.cs convention. FIXED seeds only.
public class shortRandomTests
{
    [BurstCompile]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            RangeVector,
            RangeVariety,
            ConstantFillNoAdvance,
            Determinism,
            StreamAdvance,
            EmptyAndSingle,
            MatrixRange,
            MatrixConstant,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<short> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RangeVector:           RangeVector();           break;
                case TestType.RangeVariety:          RangeVariety();          break;
                case TestType.ConstantFillNoAdvance: ConstantFillNoAdvance(); break;
                case TestType.Determinism:           Determinism();           break;
                case TestType.StreamAdvance:         StreamAdvance();         break;
                case TestType.EmptyAndSingle:        EmptyAndSingle();        break;
                case TestType.MatrixRange:           MatrixRange();           break;
                case TestType.MatrixConstant:        MatrixConstant();        break;
            }
        }

        const int N = 4096;

        // Every element of a large fill lands in [min, max).
        void RangeVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(1234567u);
            short min = (short)(-5), max = (short)10;

            var v = arena.shortVec(N);
            for (int i = 0; i < v.N; i++) v[i] = (short)999;   // poison
            Rand.nextUniformInPlace(ref rng, ref v, min, max);

            for (int i = 0; i < v.N; i++)
                AssertTrue(v[i] >= min && v[i] < max);

            arena.Dispose();
        }

        // A non-degenerate range actually varies: at least two distinct values, and both the
        // minimum value (min) and a value near the top are reachable over a large fill.
        void RangeVariety()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(0xBEEFu);
            short min = (short)0, max = (short)4;   // {0,1,2,3}

            var v = arena.shortVec(N);
            Rand.nextUniformInPlace(ref rng, ref v, min, max);

            bool sawDistinct = false;
            bool sawMin = false, sawTop = false;
            short first = v[0];
            for (int i = 0; i < v.N; i++)
            {
                if (v[i] != first) sawDistinct = true;
                if (v[i] == min) sawMin = true;
                if (v[i] == (short)3) sawTop = true;
            }
            AssertTrue(sawDistinct);
            AssertTrue(sawMin);
            AssertTrue(sawTop);

            arena.Dispose();
        }

        // min == max fills the constant min and does NOT advance the rng stream.
        void ConstantFillNoAdvance()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(424242u);
            short c = (short)7;

            var v = arena.shortVec(64);
            for (int i = 0; i < v.N; i++) v[i] = (short)0;
            uint before = rng.state;
            Rand.nextUniformInPlace(ref rng, ref v, c, c);
            uint after = rng.state;

            for (int i = 0; i < v.N; i++)
                AssertTrue(v[i] == c);
            AssertTrue(before == after);    // no rng advance on the constant-fill path

            arena.Dispose();
        }

        // Same seed + same bounds => identical buffer, element-wise exact.
        void Determinism()
        {
            var arena = new Arena(Allocator.Persistent);
            short min = (short)(-3), max = (short)9;

            var r1 = new Random(55u);
            var v1 = arena.shortVec(256);
            Rand.nextUniformInPlace(ref r1, ref v1, min, max);

            var r2 = new Random(55u);
            var v2 = arena.shortVec(256);
            Rand.nextUniformInPlace(ref r2, ref v2, min, max);

            for (int i = 0; i < v1.N; i++)
                AssertTrue(v1[i] == v2[i]);

            arena.Dispose();
        }

        // Two consecutive fills over the SAME rng advance the stream => buffers differ;
        // re-seeding reproduces the first buffer.
        void StreamAdvance()
        {
            var arena = new Arena(Allocator.Persistent);
            short min = (short)0, max = (short)1000;
            int n = 256;
            var rng = new Random(7777u);

            var v1 = arena.shortVec(n);
            Rand.nextUniformInPlace(ref rng, ref v1, min, max);
            var v2 = arena.shortVec(n);
            Rand.nextUniformInPlace(ref rng, ref v2, min, max);

            bool anyDiff = false;
            for (int i = 0; i < n; i++)
                if (v1[i] != v2[i]) anyDiff = true;
            AssertTrue(anyDiff);

            var rng3 = new Random(7777u);
            var v3 = arena.shortVec(n);
            Rand.nextUniformInPlace(ref rng3, ref v3, min, max);
            for (int i = 0; i < n; i++)
                AssertTrue(v1[i] == v3[i]);

            arena.Dispose();
        }

        // Empty vector fills nothing (no throw); a single-element vector lands in range.
        void EmptyAndSingle()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(909090u);

            var empty = arena.shortVec(0);
            Rand.nextUniformInPlace(ref rng, ref empty, (short)(-2), (short)2);
            AssertTrue(empty.N == 0);

            short min = (short)5, max = (short)6;   // {5}
            var one = arena.shortVec(1);
            Rand.nextUniformInPlace(ref rng, ref one, min, max);
            AssertTrue(one.N == 1);
            AssertTrue(one[0] >= min && one[0] < max);
            AssertTrue(one[0] == (short)5);

            arena.Dispose();
        }

        // Matrix overload fills all M*N flat elements, all in [min, max).
        void MatrixRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(20240626u);
            short min = (short)(-1), max = (short)8;

            var M = arena.shortMat(4, 5);
            for (int i = 0; i < M.Length; i++) M[i] = (short)999;   // poison
            Rand.nextUniformInPlace(ref rng, ref M, min, max);

            AssertTrue(M.Length == 20);
            for (int i = 0; i < M.Length; i++)
                AssertTrue(M[i] >= min && M[i] < max);

            arena.Dispose();
        }

        // Matrix min == max constant-fill.
        void MatrixConstant()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(11u);
            short c = (short)(-4);

            var M = arena.shortMat(3, 3);
            uint before = rng.state;
            Rand.nextUniformInPlace(ref rng, ref M, c, c);
            uint after = rng.state;

            for (int i = 0; i < M.Length; i++)
                AssertTrue(M[i] == c);
            AssertTrue(before == after);

            arena.Dispose();
        }

        // ---------------- helpers ----------------

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (short)0)
            {
                Fail[0] = (short)1;
                Fail[1] = (short)(-1);
                Fail[2] = (short)(-1);
                Fail[3] = (short)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<short>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (short)0)
                Assert.Fail($"got {(int)fail[1]}, expected/limit {(int)fail[2]}, diff/extra {(int)fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (short)0)
                Assert.Fail($"{type}: got {(int)fail[1]}, expected/limit {(int)fail[2]}, diff/extra {(int)fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void RangeVectorTest() => RunJob(TestJob.TestType.RangeVector);
    [Test] public void RangeVarietyTest() => RunJob(TestJob.TestType.RangeVariety);
    [Test] public void ConstantFillNoAdvanceTest() => RunJob(TestJob.TestType.ConstantFillNoAdvance);
    [Test] public void DeterminismTest() => RunJob(TestJob.TestType.Determinism);
    [Test] public void StreamAdvanceTest() => RunJob(TestJob.TestType.StreamAdvance);
    [Test] public void EmptyAndSingleTest() => RunJob(TestJob.TestType.EmptyAndSingle);
    [Test] public void MatrixRangeTest() => RunJob(TestJob.TestType.MatrixRange);
    [Test] public void MatrixConstantTest() => RunJob(TestJob.TestType.MatrixConstant);

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void NextUniformInPlaceMinGreaterMaxThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.shortVec(8);
        Random rng = new Random(1u);
        Assert.Throws<ArgumentException>(() => Rand.nextUniformInPlace(ref rng, ref v, (short)5, (short)1));

        var M = arena.shortMat(3, 3);
        Assert.Throws<ArgumentException>(() => Rand.nextUniformInPlace(ref rng, ref M, (short)5, (short)1));

        arena.Dispose();
    }
}
