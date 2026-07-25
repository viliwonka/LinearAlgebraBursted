using System;

using BULA;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for Histogram (count-based distribution estimation: histogram / density / cdf / 2D).
//
// Verification is mostly EXACT: counts are integers (RecordEq) and float-valued 2D counts are small
// integers (exact compare), so no tolerance is needed there. The normalized outputs (density / cdf)
// are compared with a per-precision tolerance that scales with Consts.fProxySqrtEps, so the SAME
// expression is loose for float and tight for double (mirroring the sibling Random* tests).
//
// In-job (Burst) tests cover the value behaviors; managed-thread Assert.Throws tests cover the
// argument-validation throw paths (Burst jobs cannot surface managed exceptions cleanly).
//
// NaN / ±Inf handling is the post-review fix under test: non-finite samples are DROPPED, never
// folded into bin 0.
public class fProxyHistogramTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ExplicitCounts,
            NaNInfDropped,
            ZeroedFromGarbage,
            AutoRangeFullSpan,
            AutoRangeConstant,
            AutoRangeAllNaN,
            AutoRangeLeadingNaN,
            DensitySumsToOne,
            DensityDropsBelowOne,
            CdfMonotoneLastExactlyOne,
            CdfMatchesCumulativeCounts,
            CdfAllDropped,
            Histogram2DCells,
            WeightedPickBridge,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExplicitCounts:              ExplicitCounts();              break;
                case TestType.NaNInfDropped:               NaNInfDropped();               break;
                case TestType.ZeroedFromGarbage:           ZeroedFromGarbage();           break;
                case TestType.AutoRangeFullSpan:           AutoRangeFullSpan();           break;
                case TestType.AutoRangeConstant:           AutoRangeConstant();           break;
                case TestType.AutoRangeAllNaN:             AutoRangeAllNaN();             break;
                case TestType.AutoRangeLeadingNaN:         AutoRangeLeadingNaN();         break;
                case TestType.DensitySumsToOne:            DensitySumsToOne();            break;
                case TestType.DensityDropsBelowOne:        DensityDropsBelowOne();        break;
                case TestType.CdfMonotoneLastExactlyOne:   CdfMonotoneLastExactlyOne();   break;
                case TestType.CdfMatchesCumulativeCounts:  CdfMatchesCumulativeCounts();  break;
                case TestType.CdfAllDropped:               CdfAllDropped();               break;
                case TestType.Histogram2DCells:            Histogram2DCells();            break;
                case TestType.WeightedPickBridge:          WeightedPickBridge();          break;
            }
        }

        // =====================================================================
        // histogramInto — explicit range
        // =====================================================================

        // lo=0, hi=10, K=5 -> w=2. Bins [0,2)[2,4)[4,6)[6,8)[8,10]; x==hi -> bin4.
        // data has 9 in-range + 2 out-of-range (-0.5 below, 10.5 above) -> dropped.
        // Hand-computed counts [2,2,1,1,3]; the value at lo (0) lands in bin0; the value at hi (10)
        // lands in the LAST bin (closed upper edge).
        void ExplicitCounts()
        {
            var data = new fProxyN(11, Allocator.Temp);
            data[0] = (fProxy)0;      // ->bin0 (== lo)
            data[1] = (fProxy)1;      // ->bin0
            data[2] = (fProxy)2;      // ->bin1
            data[3] = (fProxy)3.9;    // ->bin1
            data[4] = (fProxy)5;      // ->bin2
            data[5] = (fProxy)7.5;    // ->bin3
            data[6] = (fProxy)8;      // ->bin4
            data[7] = (fProxy)9.999;  // ->bin4
            data[8] = (fProxy)10;     // ->bin4 (== hi, closed upper edge)
            data[9] = (fProxy)(-0.5); // dropped (< lo)
            data[10] = (fProxy)10.5;  // dropped (> hi)

            var counts = new Indices(5, Allocator.Temp);
            for (int b = 0; b < 5; b++) counts[b] = 999;   // garbage; must be overwritten

            Histogram.histogramInto(in data, (fProxy)0, (fProxy)10, ref counts);

            RecordEq(counts[0], 2);
            RecordEq(counts[1], 2);
            RecordEq(counts[2], 1);
            RecordEq(counts[3], 1);
            RecordEq(counts[4], 3);
            RecordEq(Sum(in counts), 9);   // 2 of 11 dropped
        }

        // NaN, +Inf, -Inf injected among the 9 in-range samples: all three are DROPPED. Total counted
        // == finite-in-range count (9), and bin0 is NOT inflated by the non-finite values.
        void NaNInfDropped()
        {
            var data = new fProxyN(12, Allocator.Temp);
            // First 9 values match ExplicitCounts's fixture (same bins); [9..11] are non-finite.
            data[0] = (fProxy)0;
            data[1] = (fProxy)1;
            data[2] = (fProxy)2;
            data[3] = (fProxy)3.9;
            data[4] = (fProxy)5;
            data[5] = (fProxy)7.5;
            data[6] = (fProxy)8;
            data[7] = (fProxy)9.999;
            data[8] = (fProxy)10;
            data[9] = (fProxy)float.NaN;              // dropped
            data[10] = (fProxy)float.PositiveInfinity; // dropped
            data[11] = (fProxy)float.NegativeInfinity; // dropped

            var counts = new Indices(5, Allocator.Temp);
            Histogram.histogramInto(in data, (fProxy)0, (fProxy)10, ref counts);

            RecordEq(counts[0], 2);   // exactly the two finite samples 0 and 1 — not inflated
            RecordEq(counts[1], 2);
            RecordEq(counts[2], 1);
            RecordEq(counts[3], 1);
            RecordEq(counts[4], 3);
            RecordEq(Sum(in counts), 9);   // 3 non-finite dropped
        }

        // counts is zeroed even when no sample lands in a bin and the buffer holds garbage.
        void ZeroedFromGarbage()
        {
            var data = new fProxyN(2, Allocator.Temp);
            data[0] = (fProxy)0.1;   // bin0
            data[1] = (fProxy)0.2;   // bin0

            var counts = new Indices(4, Allocator.Temp);
            for (int b = 0; b < 4; b++) counts[b] = 777;   // garbage

            Histogram.histogramInto(in data, (fProxy)0, (fProxy)4, ref counts);

            RecordEq(counts[0], 2);
            RecordEq(counts[1], 0);   // garbage was cleared
            RecordEq(counts[2], 0);
            RecordEq(counts[3], 0);
        }

        // =====================================================================
        // histogramInto — auto-range overload
        // =====================================================================

        // {1,2,3,4,5}, K=4 -> lo=1, hi=5, w=1. Bins [1,2)[2,3)[3,4)[4,5]; 5==hi -> bin3.
        // Auto-range guarantees NO drops: sum of counts == number of finite samples (5).
        void AutoRangeFullSpan()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);

            var counts = new Indices(4, Allocator.Temp);
            Histogram.histogramInto(in data, ref counts);

            RecordEq(counts[0], 1);
            RecordEq(counts[1], 1);
            RecordEq(counts[2], 1);
            RecordEq(counts[3], 2);   // 4 and 5 (==hi) both here
            RecordEq(Sum(in counts), 5);   // no drops
        }

        // Constant finite data (max == min): all finite samples land in bin0 (no div-by-zero).
        void AutoRangeConstant()
        {
            var data = new fProxyN(3, Allocator.Temp);
            data[0] = (fProxy)3; data[1] = (fProxy)3; data[2] = (fProxy)3;

            var counts = new Indices(4, Allocator.Temp);
            for (int b = 0; b < 4; b++) counts[b] = 5;   // garbage
            Histogram.histogramInto(in data, ref counts);

            RecordEq(counts[0], 3);
            RecordEq(counts[1], 0);
            RecordEq(counts[2], 0);
            RecordEq(counts[3], 0);
        }

        // All-NaN data -> all-zero counts, no throw.
        void AutoRangeAllNaN()
        {
            var data = new fProxyN(4, Allocator.Temp);
            for (int i = 0; i < 4; i++) data[i] = (fProxy)float.NaN;

            var counts = new Indices(4, Allocator.Temp);
            for (int b = 0; b < 4; b++) counts[b] = 9;   // garbage
            Histogram.histogramInto(in data, ref counts);

            for (int b = 0; b < 4; b++) RecordEq(counts[b], 0);
        }

        // Leading NaN does not throw and the finite remainder {1,2,3,4,5} bins exactly as the full-span
        // case (min/max computed over finite samples only).
        void AutoRangeLeadingNaN()
        {
            var data = new fProxyN(6, Allocator.Temp);
            data[0] = (fProxy)float.NaN;
            for (int i = 1; i < 6; i++) data[i] = (fProxy)i;   // 1..5

            var counts = new Indices(4, Allocator.Temp);
            Histogram.histogramInto(in data, ref counts);

            RecordEq(counts[0], 1);
            RecordEq(counts[1], 1);
            RecordEq(counts[2], 1);
            RecordEq(counts[3], 2);
            RecordEq(Sum(in counts), 5);   // 5 finite, none dropped
        }

        // =====================================================================
        // densityInto
        // =====================================================================

        // All samples in range -> Sigma dest[b]*w == 1 (proper density integrating to 1).
        void DensitySumsToOne()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);   // 1..5, all within [1,5]

            int K = 4;
            fProxy lo = (fProxy)1, hi = (fProxy)5;
            fProxy w = (hi - lo) / (fProxy)K;
            var dest = new fProxyN(K, Allocator.Temp);
            Histogram.densityInto(in data, lo, hi, ref dest);

            fProxy integral = (fProxy)0;
            for (int b = 0; b < K; b++) integral += dest[b] * w;

            AssertClose(integral, (fProxy)1, (fProxy)10 * Consts.fProxySqrtEps);
        }

        // Some samples dropped -> integral strictly < 1 (drops reduce the mass).
        // {1,2,3,4,5} over [2,4]: 1 and 5 dropped, 3 of 5 kept -> integral == 3/5 == 0.6.
        void DensityDropsBelowOne()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);

            int K = 4;
            fProxy lo = (fProxy)2, hi = (fProxy)4;
            fProxy w = (hi - lo) / (fProxy)K;
            var dest = new fProxyN(K, Allocator.Temp);
            Histogram.densityInto(in data, lo, hi, ref dest);

            fProxy integral = (fProxy)0;
            for (int b = 0; b < K; b++) integral += dest[b] * w;

            AssertTrue(integral < (fProxy)1);                       // strictly below 1
            AssertClose(integral, (fProxy)0.6, (fProxy)10 * Consts.fProxySqrtEps);  // 3/5 kept
        }

        // =====================================================================
        // cdfInto
        // =====================================================================

        // Monotone non-decreasing; dest[K-1] == 1 EXACTLY (post-fix pins it).
        void CdfMonotoneLastExactlyOne()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);   // counts [1,1,1,2]

            int K = 4;
            var dest = new fProxyN(K, Allocator.Temp);
            Histogram.cdfInto(in data, (fProxy)1, (fProxy)5, ref dest);

            // monotone non-decreasing
            for (int b = 1; b < K; b++)
                AssertTrue(dest[b] >= dest[b - 1]);

            // last bin pinned to bit-exact 1 (assert EXACT equality, not tolerance)
            AssertClose(dest[K - 1], (fProxy)1, (fProxy)0);
        }

        // dest[b] == (cumulative count_i, i<=b) / inRangeTotal, matched against independently computed
        // counts. counts [1,1,1,2], total 5 -> cdf [0.2,0.4,0.6,1.0].
        void CdfMatchesCumulativeCounts()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);

            int K = 4;
            fProxy lo = (fProxy)1, hi = (fProxy)5;

            var counts = new Indices(K, Allocator.Temp);
            Histogram.histogramInto(in data, lo, hi, ref counts);
            int total = Sum(in counts);

            var dest = new fProxyN(K, Allocator.Temp);
            Histogram.cdfInto(in data, lo, hi, ref dest);

            int cum = 0;
            fProxy tol = (fProxy)10 * Consts.fProxySqrtEps;
            for (int b = 0; b < K; b++)
            {
                cum += counts[b];
                AssertClose(dest[b], (fProxy)cum / (fProxy)total, tol);
            }
        }

        // All samples dropped (range disjoint from data) -> all-zero CDF, no throw.
        void CdfAllDropped()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);   // 1..5, all below [10,20]

            int K = 4;
            var dest = new fProxyN(K, Allocator.Temp);
            for (int b = 0; b < K; b++) dest[b] = (fProxy)123;   // garbage
            Histogram.cdfInto(in data, (fProxy)10, (fProxy)20, ref dest);

            for (int b = 0; b < K; b++)
                AssertClose(dest[b], (fProxy)0, (fProxy)0);
        }

        // =====================================================================
        // histogram2DInto
        // =====================================================================

        // Kx=2 over [0,4] (wX=2), Ky=3 over [0,6] (wY=2). counts is Kx x Ky => rows=X bins, cols=Y bins.
        // Paired points land in the documented cells; a pair out of range on X only is dropped; a pair
        // with NaN on Y is dropped.
        void Histogram2DCells()
        {
            var dataX = new fProxyN(6, Allocator.Temp);
            var dataY = new fProxyN(6, Allocator.Temp);
            dataX[0] = (fProxy)0.5; dataY[0] = (fProxy)0.5;   // (bx0,by0)
            dataX[1] = (fProxy)1.0; dataY[1] = (fProxy)5.0;   // (bx0,by2)
            dataX[2] = (fProxy)3.0; dataY[2] = (fProxy)3.0;   // (bx1,by1)
            dataX[3] = (fProxy)4.0; dataY[3] = (fProxy)6.0;   // (bx1,by2) closed upper edge on BOTH axes
            dataX[4] = (fProxy)5.0; dataY[4] = (fProxy)1.0;   // X out of range -> dropped
            dataX[5] = (fProxy)1.0; dataY[5] = (fProxy)float.NaN;     // Y NaN -> dropped

            var counts = new fProxyMxN(2, 3, Allocator.Temp);
            for (int i = 0; i < counts.Length; i++) counts[i] = (fProxy)999;   // garbage

            Histogram.histogram2DInto(in dataX, in dataY,
                (fProxy)0, (fProxy)4, (fProxy)0, (fProxy)6, ref counts);

            // rows = X bins (2), cols = Y bins (3)
            AssertTrue(counts.M_Rows == 2);
            AssertTrue(counts.N_Cols == 3);

            // expected (exact small integer counts):
            // row0: [1,0,1]   row1: [0,1,1]
            AssertClose(counts[0, 0], (fProxy)1, (fProxy)0);
            AssertClose(counts[0, 1], (fProxy)0, (fProxy)0);
            AssertClose(counts[0, 2], (fProxy)1, (fProxy)0);
            AssertClose(counts[1, 0], (fProxy)0, (fProxy)0);
            AssertClose(counts[1, 1], (fProxy)1, (fProxy)0);
            AssertClose(counts[1, 2], (fProxy)1, (fProxy)0);
        }

        // =====================================================================
        // sampling bridge: histogram counts -> weightedPick
        // =====================================================================

        // Build counts via histogramInto, copy them into an fProxyN weight vector, then draw repeatedly
        // with a seeded Random. Every picked bin index must be in [0, K).
        void WeightedPickBridge()
        {
            var data = new fProxyN(5, Allocator.Temp);
            for (int i = 0; i < 5; i++) data[i] = (fProxy)(i + 1);

            int K = 4;
            var counts = new Indices(K, Allocator.Temp);
            Histogram.histogramInto(in data, (fProxy)1, (fProxy)5, ref counts);   // [1,1,1,2]

            var weights = new fProxyN(K, Allocator.Temp);
            for (int b = 0; b < K; b++) weights[b] = (fProxy)counts[b];

            var rng = new Random(20240627u);
            for (int t = 0; t < 128; t++)
            {
                int pick = Rand.weightedPick(in weights, ref rng);
                AssertTrue(pick >= 0 && pick < K);
            }
        }

        // =====================================================================
        // helpers
        // =====================================================================

        int Sum(in Indices counts)
        {
            int s = 0;
            for (int b = 0; b < counts.N; b++) s += counts[b];
            return s;
        }

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

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void HistogramTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
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

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void HistogramIntoValidates()
    {
        var data = new fProxyN(4, Allocator.Temp);
        for (int i = 0; i < 4; i++) data[i] = (fProxy)i;

        // K < 1 (empty counts)
        var empty = new Indices(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Histogram.histogramInto(in data, (fProxy)0, (fProxy)1, ref empty));

        // !(hi > lo): equal, and inverted
        var counts = new Indices(4, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Histogram.histogramInto(in data, (fProxy)1, (fProxy)1, ref counts));
        Assert.Throws<ArgumentException>(
            () => Histogram.histogramInto(in data, (fProxy)5, (fProxy)1, ref counts));

        // auto-range overload also rejects K < 1
        Assert.Throws<ArgumentException>(
            () => Histogram.histogramInto(in data, ref empty));
    }

    [Test]
    public void DensityIntoValidates()
    {
        var data = new fProxyN(4, Allocator.Temp);
        for (int i = 0; i < 4; i++) data[i] = (fProxy)i;
        var dest = new fProxyN(4, Allocator.Temp);

        // hi <= lo
        Assert.Throws<ArgumentException>(
            () => Histogram.densityInto(in data, (fProxy)1, (fProxy)1, ref dest));

        // empty data (cannot normalize)
        var emptyData = new fProxyN(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Histogram.densityInto(in emptyData, (fProxy)0, (fProxy)1, ref dest));

        // K < 1
        var emptyDest = new fProxyN(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Histogram.densityInto(in data, (fProxy)0, (fProxy)1, ref emptyDest));
    }

    [Test]
    public void CdfIntoValidates()
    {
        var data = new fProxyN(4, Allocator.Temp);
        for (int i = 0; i < 4; i++) data[i] = (fProxy)i;
        var dest = new fProxyN(4, Allocator.Temp);

        Assert.Throws<ArgumentException>(
            () => Histogram.cdfInto(in data, (fProxy)2, (fProxy)1, ref dest));

        var emptyDest = new fProxyN(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Histogram.cdfInto(in data, (fProxy)0, (fProxy)1, ref emptyDest));
    }

    [Test]
    public void Histogram2DIntoValidates()
    {
        var dataX = new fProxyN(5, Allocator.Temp);
        var dataY = new fProxyN(4, Allocator.Temp);   // mismatched length
        var counts = new fProxyMxN(2, 2, Allocator.Temp);

        // mismatched dataX / dataY lengths
        Assert.Throws<ArgumentException>(
            () => Histogram.histogram2DInto(in dataX, in dataY,
                (fProxy)0, (fProxy)1, (fProxy)0, (fProxy)1, ref counts));

        // paired (equal-length) but invalid ranges
        var dY = new fProxyN(5, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Histogram.histogram2DInto(in dataX, in dY,
                (fProxy)1, (fProxy)1, (fProxy)0, (fProxy)1, ref counts));   // hiX <= loX
        Assert.Throws<ArgumentException>(
            () => Histogram.histogram2DInto(in dataX, in dY,
                (fProxy)0, (fProxy)1, (fProxy)2, (fProxy)1, ref counts));   // hiY <= loY
    }
}
