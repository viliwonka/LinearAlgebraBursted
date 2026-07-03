using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for the weighted discrete picker (Rand.weightedPick / weightedPickInPlace).
// One template expands to Rand / Rand, so statistics use loose tolerances
// that hold for both precisions; the underlying uniform stream is float-valued for both, so a
// fixed seed makes every count deterministic.
//
//   * weightedPick(in weights, ref rng) -> int : unnormalized PMF, linear cumulative scan.
//   * weightedPickInPlace(in weights, ref dest, ref rng) : k picks with replacement; validates ONCE
//     up front (throws even if dest.N == 0 with bad weights).
//
// Burst-compatible computational tests live in TestJob; managed-throw guards are plain [Test]
// methods on the main thread. FIXED seeds only.
public class floatRandomWeightedTests
{
    [BurstCompile]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            SingleElementAlwaysZero,
            ZeroWeightNeverPicked,
            Proportionality,
            InPlaceFillsInRange,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SingleElementAlwaysZero: SingleElementAlwaysZero(); break;
                case TestType.ZeroWeightNeverPicked:   ZeroWeightNeverPicked();   break;
                case TestType.Proportionality:         Proportionality();         break;
                case TestType.InPlaceFillsInRange:        InPlaceFillsInRange();        break;
            }
        }

        const int Draws = 6000;

        // Single-element weights => always index 0.
        void SingleElementAlwaysZero()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(1234567u);
            var w = arena.floatVec(1);
            w[0] = (float)5;
            for (int i = 0; i < 200; i++)
                AssertTrue(Rand.weightedPick(in w, ref rng) == 0);
            arena.Dispose();
        }

        // weights {1,0,1}: index 1 (zero weight) is NEVER returned; 0 and 2 ~ equally likely.
        void ZeroWeightNeverPicked()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2468013u);
            var w = arena.floatVec(3);
            w[0] = (float)1; w[1] = (float)0; w[2] = (float)1;

            int c0 = 0, c1 = 0, c2 = 0;
            for (int i = 0; i < Draws; i++)
            {
                int idx = Rand.weightedPick(in w, ref rng);
                AssertTrue(idx >= 0 && idx < 3);
                if (idx == 0) c0++; else if (idx == 1) c1++; else c2++;
            }
            AssertTrue(c1 == 0);
            float frac0 = (float)c0 / (float)(c0 + c2);
            AssertClose(frac0, (float)0.5, (float)0.05);        // 0 and 2 ~ equal
            arena.Dispose();
        }

        // weights {1,3}: index 1 chosen ~3x as often as index 0.
        void Proportionality()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(13572468u);
            var w = arena.floatVec(2);
            w[0] = (float)1; w[1] = (float)3;

            int c0 = 0, c1 = 0;
            for (int i = 0; i < Draws; i++)
            {
                int idx = Rand.weightedPick(in w, ref rng);
                if (idx == 0) c0++; else c1++;
            }
            AssertTrue(c0 > 0);
            float ratio = (float)c1 / (float)c0;
            AssertClose(ratio, (float)3, (float)0.35);          // 3:1, loose
            arena.Dispose();
        }

        // weightedPickInPlace fills dest.N picks, all valid indices; zero-weight index excluded.
        void InPlaceFillsInRange()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(97531864u);
            var w = arena.floatVec(4);
            w[0] = (float)2; w[1] = (float)0; w[2] = (float)1; w[3] = (float)1;

            int k = 256;
            var dest = arena.Indices(k);
            Rand.weightedPickInPlace(in w, ref dest, ref rng);
            AssertTrue(dest.N == k);
            for (int i = 0; i < k; i++)
            {
                AssertTrue(dest[i] >= 0 && dest[i] < 4);
                AssertTrue(dest[i] != 1);
            }
            arena.Dispose();
        }

        // ---------------- helpers ----------------

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = (float)(-1);
                Fail[2] = (float)(-1);
                Fail[3] = (float)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void SingleElementAlwaysZeroTest() => RunJob(TestJob.TestType.SingleElementAlwaysZero);
    [Test] public void ZeroWeightNeverPickedTest() => RunJob(TestJob.TestType.ZeroWeightNeverPicked);
    [Test] public void ProportionalityTest() => RunJob(TestJob.TestType.Proportionality);
    [Test] public void InPlaceFillsInRangeTest() => RunJob(TestJob.TestType.InPlaceFillsInRange);

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void WeightedPickValidationThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(1u);

        // Empty weights.
        var empty = arena.floatVec(0);
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in empty, ref rng));

        // Any negative weight.
        var neg = arena.floatVec(3);
        neg[0] = (float)1; neg[1] = (float)(-2); neg[2] = (float)1;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in neg, ref rng));

        // NaN weight (0/0 at runtime; not a compile-time constant).
        var nan = arena.floatVec(2);
        nan[0] = (float)1; nan[1] = (float)0 / (float)0;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in nan, ref rng));

        // +Inf weight (1/0 at runtime).
        var inf = arena.floatVec(2);
        inf[0] = (float)1; inf[1] = (float)1 / (float)0;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in inf, ref rng));

        // All-zero total.
        var zero = arena.floatVec(3);
        zero[0] = (float)0; zero[1] = (float)0; zero[2] = (float)0;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in zero, ref rng));

        arena.Dispose();
    }

    [Test]
    public void WeightedPickInPlaceValidatesUpFrontEvenWhenDestEmpty()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(1u);

        // Invalid (negative) weights must throw even though dest.N == 0 (validation runs first).
        var bad = arena.floatVec(3);
        bad[0] = (float)1; bad[1] = (float)(-1); bad[2] = (float)1;
        var emptyDest = arena.Indices(0);
        Assert.Throws<ArgumentException>(() => Rand.weightedPickInPlace(in bad, ref emptyDest, ref rng));

        // All-zero total likewise throws with an empty destination.
        var zero = arena.floatVec(2);
        zero[0] = (float)0; zero[1] = (float)0;
        var emptyDest2 = arena.Indices(0);
        Assert.Throws<ArgumentException>(() => Rand.weightedPickInPlace(in zero, ref emptyDest2, ref rng));

        arena.Dispose();
    }
}
