using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for the weighted discrete picker (Rand.weightedPick / weightedPickInPlace).
// One template expands to a float and a double build, so statistics use loose tolerances
// that hold for both precisions; the underlying uniform stream is float-valued for both, so a
// fixed seed makes every count deterministic.
//
//   * weightedPick(in weights, ref rng) -> int : unnormalized PMF, linear cumulative scan.
//   * weightedPickInPlace(in weights, ref dest, ref rng) : k picks with replacement; validates ONCE
//     up front (throws even if dest.N == 0 with bad weights).
//
// Burst-compatible computational tests live in TestJob; managed-throw guards are plain [Test]
// methods on the main thread. FIXED seeds only.
public class fProxyRandomWeightedTests
{
    [BurstCompile(CompileSynchronously = true)]
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
        public NativeArray<fProxy> Fail;

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
            var rng = new Random(1234567u);
            var w = new fProxyN(1, Allocator.Temp);
            w[0] = (fProxy)5;
            for (int i = 0; i < 200; i++)
                AssertTrue(Rand.weightedPick(in w, ref rng) == 0);
        }

        // weights {1,0,1}: index 1 (zero weight) is NEVER returned; 0 and 2 ~ equally likely.
        void ZeroWeightNeverPicked()
        {
            var rng = new Random(2468013u);
            var w = new fProxyN(3, Allocator.Temp);
            w[0] = (fProxy)1; w[1] = (fProxy)0; w[2] = (fProxy)1;

            int c0 = 0, c1 = 0, c2 = 0;
            for (int i = 0; i < Draws; i++)
            {
                int idx = Rand.weightedPick(in w, ref rng);
                AssertTrue(idx >= 0 && idx < 3);
                if (idx == 0) c0++; else if (idx == 1) c1++; else c2++;
            }
            AssertTrue(c1 == 0);
            fProxy frac0 = (fProxy)c0 / (fProxy)(c0 + c2);
            AssertClose(frac0, (fProxy)0.5, (fProxy)0.05);        // 0 and 2 ~ equal
        }

        // weights {1,3}: index 1 chosen ~3x as often as index 0.
        void Proportionality()
        {
            var rng = new Random(13572468u);
            var w = new fProxyN(2, Allocator.Temp);
            w[0] = (fProxy)1; w[1] = (fProxy)3;

            int c0 = 0, c1 = 0;
            for (int i = 0; i < Draws; i++)
            {
                int idx = Rand.weightedPick(in w, ref rng);
                if (idx == 0) c0++; else c1++;
            }
            AssertTrue(c0 > 0);
            fProxy ratio = (fProxy)c1 / (fProxy)c0;
            AssertClose(ratio, (fProxy)3, (fProxy)0.35);          // 3:1, loose
        }

        // weightedPickInPlace fills dest.N picks, all valid indices; zero-weight index excluded.
        void InPlaceFillsInRange()
        {
            var rng = new Random(97531864u);
            var w = new fProxyN(4, Allocator.Temp);
            w[0] = (fProxy)2; w[1] = (fProxy)0; w[2] = (fProxy)1; w[3] = (fProxy)1;

            int k = 256;
            var dest = new Indices(k, Allocator.Temp);
            Rand.weightedPickInPlace(in w, ref dest, ref rng);
            AssertTrue(dest.N == k);
            for (int i = 0; i < k; i++)
            {
                AssertTrue(dest[i] >= 0 && dest[i] < 4);
                AssertTrue(dest[i] != 1);
            }
        }

        // ---------------- helpers ----------------

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)(-1);
                Fail[2] = (fProxy)(-1);
                Fail[3] = (fProxy)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
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
        Random rng = new Random(1u);

        // Empty weights.
        var empty = new fProxyN(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in empty, ref rng));

        // Any negative weight.
        var neg = new fProxyN(3, Allocator.Temp);
        neg[0] = (fProxy)1; neg[1] = (fProxy)(-2); neg[2] = (fProxy)1;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in neg, ref rng));

        // NaN weight (0/0 at runtime; not a compile-time constant).
        var nan = new fProxyN(2, Allocator.Temp);
        nan[0] = (fProxy)1; nan[1] = (fProxy)0 / (fProxy)0;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in nan, ref rng));

        // +Inf weight (1/0 at runtime).
        var inf = new fProxyN(2, Allocator.Temp);
        inf[0] = (fProxy)1; inf[1] = (fProxy)1 / (fProxy)0;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in inf, ref rng));

        // All-zero total.
        var zero = new fProxyN(3, Allocator.Temp);
        zero[0] = (fProxy)0; zero[1] = (fProxy)0; zero[2] = (fProxy)0;
        Assert.Throws<ArgumentException>(() => Rand.weightedPick(in zero, ref rng));
    }

    [Test]
    public void WeightedPickInPlaceValidatesUpFrontEvenWhenDestEmpty()
    {
        Random rng = new Random(1u);

        // Invalid (negative) weights must throw even though dest.N == 0 (validation runs first).
        var bad = new fProxyN(3, Allocator.Temp);
        bad[0] = (fProxy)1; bad[1] = (fProxy)(-1); bad[2] = (fProxy)1;
        var emptyDest = new Indices(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Rand.weightedPickInPlace(in bad, ref emptyDest, ref rng));

        // All-zero total likewise throws with an empty destination.
        var zero = new fProxyN(2, Allocator.Temp);
        zero[0] = (fProxy)0; zero[1] = (fProxy)0;
        var emptyDest2 = new Indices(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Rand.weightedPickInPlace(in zero, ref emptyDest2, ref rng));
    }
}
