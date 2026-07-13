using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for the median / quartile / IQR path of
// Stats.meanMinMaxRange_medianIQRstdDevVariance, including the n==2 regression case (must not
// read out of bounds). Quartiles use the linear-interpolation percentile (numpy 'linear').
public class fProxyFullStatsTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            TwoElements,
            FourElements,
            MedianOddEven,
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.TwoElements:   TwoElements(); break;
                case TestType.FourElements:  FourElements(); break;
                case TestType.MedianOddEven: MedianOddEven(); break;
            }
        }

        // n==2 (the old OOB case). [20,10] -> sorted [10,20]. median=15, q1=12.5, q3=17.5, iqr=5,
        // mean=15, variance(population)=25, stdDev=5, min=10, max=20, range=10.
        void TwoElements()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(2);
            v[0] = (fProxy)20; v[1] = (fProxy)10;   // unsorted on purpose

            var s = Stats.meanMinMaxRange_medianIQRstdDevVariance(in v);

            AssertClose(s.median, (fProxy)15, (fProxy)1E-4);
            AssertClose(s.q1, (fProxy)12.5, (fProxy)1E-4);
            AssertClose(s.q3, (fProxy)17.5, (fProxy)1E-4);
            AssertClose(s.iqr, (fProxy)5, (fProxy)1E-4);
            AssertClose(s.mean, (fProxy)15, (fProxy)1E-4);
            AssertClose(s.variance, (fProxy)25, (fProxy)1E-3);
            AssertClose(s.stdDev, (fProxy)5, (fProxy)1E-3);
            AssertClose(s.min, (fProxy)10, (fProxy)1E-4);
            AssertClose(s.max, (fProxy)20, (fProxy)1E-4);
            AssertClose(s.range, (fProxy)10, (fProxy)1E-4);

            arena.Dispose();
        }

        // [3,1,4,2] -> sorted [1,2,3,4]. median=2.5, q1=1.75, q3=3.25, iqr=1.5, mean=2.5,
        // variance(population)=1.25, stdDev=sqrt(1.25).
        void FourElements()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(4);
            v[0] = (fProxy)3; v[1] = (fProxy)1; v[2] = (fProxy)4; v[3] = (fProxy)2;

            var s = Stats.meanMinMaxRange_medianIQRstdDevVariance(in v);

            AssertClose(s.median, (fProxy)2.5, (fProxy)1E-4);
            AssertClose(s.q1, (fProxy)1.75, (fProxy)1E-4);
            AssertClose(s.q3, (fProxy)3.25, (fProxy)1E-4);
            AssertClose(s.iqr, (fProxy)1.5, (fProxy)1E-4);
            AssertClose(s.mean, (fProxy)2.5, (fProxy)1E-4);
            AssertClose(s.variance, (fProxy)1.25, (fProxy)1E-4);
            AssertClose(s.stdDev, math.sqrt((fProxy)1.25), (fProxy)1E-4);

            arena.Dispose();
        }

        // standalone median: odd [1,2,3] -> 2; even [1,2,3,4] -> 2.5.
        void MedianOddEven()
        {
            var arena = new Arena(Allocator.Persistent);

            var odd = arena.fProxyVec(3);
            odd[0] = (fProxy)3; odd[1] = (fProxy)1; odd[2] = (fProxy)2;
            AssertClose(Stats.median(in odd), (fProxy)2, (fProxy)1E-4);

            var even = arena.fProxyVec(4);
            even[0] = (fProxy)4; even[1] = (fProxy)2; even[2] = (fProxy)1; even[3] = (fProxy)3;
            AssertClose(Stats.median(in even), (fProxy)2.5, (fProxy)1E-4);

            arena.Dispose();
        }

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void FullStatsTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
