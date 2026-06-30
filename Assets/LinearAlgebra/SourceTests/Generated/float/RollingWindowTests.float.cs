using System;

using LinearAlgebra;
using LinearAlgebra.Stats;
using LinearAlgebra.Realtime;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class floatRollingWindowTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            PushAndOrderNotFull,
            WrapAroundOrder,
            AsMatrixTimeOrder,
            MovingAverage,
            CovarianceMatchesStatsOP,
            GetSampleAndIndexer,
            ClearResets,
            IndexerFeatureBounds
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.PushAndOrderNotFull: PushAndOrderNotFull(); break;
                case TestType.WrapAroundOrder: WrapAroundOrder(); break;
                case TestType.AsMatrixTimeOrder: AsMatrixTimeOrder(); break;
                case TestType.MovingAverage: MovingAverage(); break;
                case TestType.CovarianceMatchesStatsOP: CovarianceMatchesStatsOP(); break;
                case TestType.GetSampleAndIndexer: GetSampleAndIndexer(); break;
                case TestType.ClearResets: ClearResets(); break;
                case TestType.IndexerFeatureBounds: IndexerFeatureBounds(); break;
            }
        }

        // Positive coverage for the [i,f] indexer's two-axis bounds: every valid (i,f) within
        // Count×Features returns the right sample value (feature axis included, not just feature 0).
        void IndexerFeatureBounds()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(3, 2);

            // Distinct per (sample, feature) values: sample s, feature f -> 10*s + f.
            var s = arena.floatVec(2);
            s[0] = 10f; s[1] = 11f; w.Push(in s); // sample 0
            s[0] = 20f; s[1] = 21f; w.Push(in s); // sample 1
            s[0] = 30f; s[1] = 31f; w.Push(in s); // sample 2

            AssertEqI(3, w.Count);
            AssertEqI(2, w.Features);

            for (int i = 0; i < w.Count; i++)
                for (int f = 0; f < w.Features; f++)
                    AssertClose(w[i, f], (float)(10f * (i + 1) + f), 1E-6f);

            arena.Dispose();
        }

        // Helper: push a single 1-feature sample value. arena is taken by ref — floatVec mutates the
        // arena's tracking list, so an `in` parameter would mutate a defensive copy instead.
        static void Push1(ref floatRollingWindow w, ref Arena arena, float v)
        {
            var s = arena.floatVec(1);
            s[0] = v;
            w.Push(in s);
        }

        // Partly-filled window keeps insertion order; oldest=this[0], newest=this[Count-1].
        void PushAndOrderNotFull()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(4, 2);

            AssertTrue(w.IsEmpty);
            AssertEqI(0, w.Count);

            var s = arena.floatVec(2);
            s[0] = 1f; s[1] = 10f; w.Push(in s);
            s[0] = 2f; s[1] = 20f; w.Push(in s);
            s[0] = 3f; s[1] = 30f; w.Push(in s);

            AssertEqI(3, w.Count);
            AssertTrue(!w.IsFull);
            AssertTrue(!w.IsEmpty);
            AssertEqI(4, w.Capacity);
            AssertEqI(2, w.Features);

            AssertClose(w[0, 0], 1f, 1E-6f); AssertClose(w[0, 1], 10f, 1E-6f); // oldest
            AssertClose(w[2, 0], 3f, 1E-6f); AssertClose(w[2, 1], 30f, 1E-6f); // newest

            arena.Dispose();
        }

        // Once full, Push overwrites the oldest; logical order stays oldest→newest.
        // cap=3, push 1..5 -> window holds [3,4,5].
        void WrapAroundOrder()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(3, 1);

            for (int v = 1; v <= 5; v++)
                Push1(ref w, ref arena, (float)v);

            AssertTrue(w.IsFull);
            AssertEqI(3, w.Count);
            AssertClose(w[0, 0], 3f, 1E-6f);
            AssertClose(w[1, 0], 4f, 1E-6f);
            AssertClose(w[2, 0], 5f, 1E-6f);

            arena.Dispose();
        }

        // AsMatrix is time-ordered (row 0 = oldest); ref form and allocating form agree.
        void AsMatrixTimeOrder()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(3, 1);
            for (int v = 1; v <= 5; v++) // wraps -> [3,4,5]
                Push1(ref w, ref arena, (float)v);

            var M = arena.floatMat(w.Count, w.Features);
            w.AsMatrix(ref M);
            AssertClose(M[0, 0], 3f, 1E-6f);
            AssertClose(M[1, 0], 4f, 1E-6f);
            AssertClose(M[2, 0], 5f, 1E-6f);

            var Ma = w.AsMatrix();
            AssertEqI(3, Ma.M_Rows);
            AssertEqI(1, Ma.N_Cols);
            for (int i = 0; i < 3; i++)
                AssertClose(Ma[i, 0], M[i, 0], 0f);

            arena.Dispose();
        }

        // Mean = per-feature moving average. cap=4 feat=2, push 4 -> mean; then wrap -> mean shifts.
        void MovingAverage()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(4, 2);

            var s = arena.floatVec(2);
            s[0] = 1f; s[1] = 10f; w.Push(in s);
            s[0] = 2f; s[1] = 20f; w.Push(in s);
            s[0] = 3f; s[1] = 30f; w.Push(in s);
            s[0] = 4f; s[1] = 40f; w.Push(in s);

            var m = arena.floatVec(2);
            w.Mean(ref m);
            AssertClose(m[0], 2.5f, 1E-5f);   // mean(1,2,3,4)
            AssertClose(m[1], 25f, 1E-5f);    // mean(10,20,30,40)

            // one more sample evicts the oldest -> window {2..5, 20..50}
            s[0] = 5f; s[1] = 50f; w.Push(in s);
            w.Mean(ref m);
            AssertClose(m[0], 3.5f, 1E-5f);   // mean(2,3,4,5)
            AssertClose(m[1], 35f, 1E-5f);    // mean(20,30,40,50)

            // allocating form agrees
            var ma = w.Mean();
            AssertClose(ma[0], 3.5f, 1E-5f);
            AssertClose(ma[1], 35f, 1E-5f);

            arena.Dispose();
        }

        // window.Covariance() must equal StatsOP.covariance(window.AsMatrix()).
        // Samples {{1,2},{3,6},{5,4}} -> covariance {{4,2},{2,4}} (same oracle as StatsTests).
        void CovarianceMatchesStatsOP()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(3, 2);

            var s = arena.floatVec(2);
            s[0] = 1f; s[1] = 2f; w.Push(in s);
            s[0] = 3f; s[1] = 6f; w.Push(in s);
            s[0] = 5f; s[1] = 4f; w.Push(in s);

            var C = arena.floatMat(2, 2);
            w.Covariance(ref C);

            AssertClose(C[0, 0], 4f, 1E-5f);
            AssertClose(C[0, 1], 2f, 1E-5f);
            AssertClose(C[1, 0], 2f, 1E-5f);
            AssertClose(C[1, 1], 4f, 1E-5f);

            // identical to running StatsOP directly on the materialized matrix
            var viaStats = floatStats_OP.covariance(w.AsMatrix());
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    AssertClose(C[i, j], viaStats[i, j], 1E-5f);

            arena.Dispose();
        }

        // GetSample copies a row; matches the [i,f] indexer.
        void GetSampleAndIndexer()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(3, 2);

            var s = arena.floatVec(2);
            s[0] = 7f; s[1] = 8f; w.Push(in s);
            s[0] = 9f; s[1] = 1f; w.Push(in s);

            var got = arena.floatVec(2);
            w.GetSample(0, ref got);
            AssertClose(got[0], w[0, 0], 0f);
            AssertClose(got[1], w[0, 1], 0f);
            AssertClose(got[0], 7f, 1E-6f);

            w.GetSample(1, ref got);
            AssertClose(got[0], 9f, 1E-6f);
            AssertClose(got[1], 1f, 1E-6f);

            arena.Dispose();
        }

        // Clear empties logically without touching capacity/features.
        void ClearResets()
        {
            var arena = new Arena(Allocator.Persistent);
            var w = arena.floatRollingWindow(3, 1);
            Push1(ref w, ref arena, 1f);
            Push1(ref w, ref arena, 2f);
            AssertEqI(2, w.Count);

            w.Clear();
            AssertEqI(0, w.Count);
            AssertTrue(w.IsEmpty);
            AssertEqI(3, w.Capacity);

            // reusable after clear
            Push1(ref w, ref arena, 9f);
            AssertEqI(1, w.Count);
            AssertClose(w[0, 0], 9f, 1E-6f);

            arena.Dispose();
        }

        // ---- Fail-array diagnostics (layout: [0]=flag, [1]=got, [2]=expected, [3]=diff) ----
        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertEqI(int expected, int got)
        {
            if (expected != got && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)got; Fail[2] = (float)expected; Fail[3] = (float)(got - expected);
            }
            Assert.AreEqual(expected, got);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)(-1); Fail[2] = (float)(-1); Fail[3] = (float)(-1);
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
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void PushAndOrderNotFullTest() => RunJob(TestJob.TestType.PushAndOrderNotFull);
    [Test] public void WrapAroundOrderTest() => RunJob(TestJob.TestType.WrapAroundOrder);
    [Test] public void AsMatrixTimeOrderTest() => RunJob(TestJob.TestType.AsMatrixTimeOrder);
    [Test] public void MovingAverageTest() => RunJob(TestJob.TestType.MovingAverage);
    [Test] public void CovarianceMatchesStatsOPTest() => RunJob(TestJob.TestType.CovarianceMatchesStatsOP);
    [Test] public void GetSampleAndIndexerTest() => RunJob(TestJob.TestType.GetSampleAndIndexer);
    [Test] public void ClearResetsTest() => RunJob(TestJob.TestType.ClearResets);
    [Test] public void IndexerFeatureBoundsTest() => RunJob(TestJob.TestType.IndexerFeatureBounds);

    // ---- Managed throw tests (guard paths) ----

    [Test]
    public void FactoryBadDimsThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        Assert.Throws<ArgumentException>(() => arena.floatRollingWindow(0, 2));
        Assert.Throws<ArgumentException>(() => arena.floatRollingWindow(4, 0));
        arena.Dispose();
    }

    [Test]
    public void PushWrongLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var w = arena.floatRollingWindow(4, 3);
        var bad = arena.floatVec(2);
        Assert.Throws<ArgumentException>(() => w.Push(in bad));
        arena.Dispose();
    }

    [Test]
    public void CovarianceTooFewSamplesThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var w = arena.floatRollingWindow(4, 2);
        var s = arena.floatVec(2);
        s[0] = 1f; s[1] = 2f; w.Push(in s); // only 1 sample
        var C = arena.floatMat(2, 2);
        Assert.Throws<InvalidOperationException>(() => w.Covariance(ref C));
        arena.Dispose();
    }

    [Test]
    public void MeanEmptyThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var w = arena.floatRollingWindow(4, 2);
        var m = arena.floatVec(2);
        Assert.Throws<InvalidOperationException>(() => w.Mean(ref m));
        arena.Dispose();
    }

    [Test]
    public void AsMatrixWrongSizeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var w = arena.floatRollingWindow(4, 2);
        var s = arena.floatVec(2);
        s[0] = 1f; s[1] = 2f; w.Push(in s);
        s[0] = 3f; s[1] = 4f; w.Push(in s);
        var wrong = arena.floatMat(3, 2); // Count is 2, not 3
        Assert.Throws<ArgumentException>(() => w.AsMatrix(ref wrong));
        arena.Dispose();
    }

    // The [i,f] indexer validates BOTH axes via Assume.IndexInsideBounds, which throws
    // ArgumentException when collection checks are enabled. The guard is compiled only under
    // ENABLE_UNITY_COLLECTIONS_CHECKS, so the throw tests are compiled under the same symbol —
    // when checks are off the indexer cannot throw and these tests would be vacuous/false.
#if ENABLE_UNITY_COLLECTIONS_CHECKS
    [Test]
    public void IndexerFeatureOutOfRangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var w = arena.floatRollingWindow(3, 2);
        var s = arena.floatVec(2);
        s[0] = 1f; s[1] = 2f; w.Push(in s);
        s[0] = 3f; s[1] = 4f; w.Push(in s);

        // f == Features (2) is one past the last valid feature index.
        Assert.Throws<ArgumentException>(() => { var _ = w[0, w.Features]; });
        // negative feature index also rejected.
        Assert.Throws<ArgumentException>(() => { var _ = w[0, -1]; });

        arena.Dispose();
    }

    [Test]
    public void IndexerSampleOutOfRangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var w = arena.floatRollingWindow(3, 2);
        var s = arena.floatVec(2);
        s[0] = 1f; s[1] = 2f; w.Push(in s);
        s[0] = 3f; s[1] = 4f; w.Push(in s);

        // i == Count (2) is one past the last valid sample index (not Capacity).
        Assert.Throws<ArgumentException>(() => { var _ = w[w.Count, 0]; });
        // negative sample index also rejected.
        Assert.Throws<ArgumentException>(() => { var _ = w[-1, 0]; });

        arena.Dispose();
    }
#endif
}
