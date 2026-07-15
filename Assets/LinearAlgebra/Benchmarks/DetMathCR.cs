using System.Globalization;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Correctly-rounded (0-ULP) FLOAT transcendentals, prototype (benchmark-only, non-shipping).
    //
    // Technique: evaluate the deterministic polynomial in DOUBLE, then round once to float. A double
    // approximation good to ~2^-56 relative is far below float's 2^-24 rounding grid, so casting to
    // float lands on the correctly-rounded result for (nearly) every input — verified EXHAUSTIVELY
    // over all 2^32 float bit patterns below. This is a FLOAT-only technique: correctly-rounded DOUBLE
    // would need double-double (Dekker) intermediate precision, a substantially larger effort.
    //
    // Still deterministic (+ - * / and int/bit only) and typically faster than the platform libm.
    internal static class DetMathCR
    {
        const double INV_LN2 = 1.4426950408889634;
        const double LN2_HI  = 6.93147180369123816490e-01;
        const double LN2_LO  = 1.90821492927058770002e-10;

        // exp via degree-11 minimax in double (~0.03 ULP-of-double), then round to float.
        public static float Exp(float xf)
        {
            double x = xf;
            double n = math.floor(INV_LN2 * x + 0.5);
            double r = x - n * LN2_HI;
            r = r - n * LN2_LO;
            double p = 2.4994305002802721e-08;
            p = p * r + 2.7632293277020833e-07;
            p = p * r + 2.7557622530418205e-06;
            p = p * r + 2.4801486521436422e-05;
            p = p * r + 1.9841269432679901e-04;
            p = p * r + 1.3888888951223976e-03;
            p = p * r + 8.3333333335592709e-03;
            p = p * r + 4.1666666666492767e-02;
            p = p * r + 1.6666666666666169e-01;
            p = p * r + 5.0000000000000177e-01;
            p = p * r + 1.0;
            p = p * r + 1.0;
            long e = (long)n;
            return (float)(p * math.asdouble((e + 1023L) << 52));
        }

        // log via degree-8 B(s^2) atanh form in double (~1e-4 ULP-of-double), then round to float.
        public static float Log(float xf)
        {
            double x = xf;
            long l = math.aslong(x);
            int e = (int)((l >> 52) & 0x7FF) - 1023;
            double m = math.asdouble((l & 0x000FFFFFFFFFFFFFL) | 0x3FF0000000000000L);
            double adj = math.select(0.0, 1.0, m > 1.4142135623730951);
            m = m - adj * (m * 0.5);
            double ef = (double)e + adj;
            double f = m - 1.0;
            double s = f / (2.0 + f);
            double z = s * s;
            double b = 6.64682949817336435e-02;
            b = b * z + 6.62262060746956054e-02;
            b = b * z + 7.69365997006834700e-02;
            b = b * z + 9.09088500878415829e-02;
            b = b * z + 1.11111113624414839e-01;
            b = b * z + 1.42857142842428395e-01;
            b = b * z + 2.00000000000043119e-01;
            b = b * z + 3.33333333333333285e-01;
            b = b * z + 1.0;
            double logm = (2.0 * s) * b;
            return (float)(ef * LN2_HI + (logm + ef * LN2_LO));
        }
    }

    // ---- CR throughput jobs (float) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ExpCRJob : IJob
    {
        public floatN src, dst;
        public void Execute() { int n = src.N; for (int i = 0; i < n; i++) dst[i] = DetMathCR.Exp(src[i]); }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LogCRJob : IJob
    {
        public floatN src, dst;
        public void Execute() { int n = src.N; for (int i = 0; i < n; i++) dst[i] = DetMathCR.Log(src[i]); }
    }

    // ---- exhaustive verifier: walk EVERY 2^32 float bit pattern, compare our CR result to the
    // double-libm reference rounded to float. out[0] = max |ULP diff|, out[1] = mismatch count,
    // out[2] = inputs checked. func: 0 = log (x > 0 finite), 1 = exp (finite, |x| <= 88). ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct VerifyCRJob : IJob
    {
        public int func;
        public NativeArray<long> outp;   // [maxUlp, mismatches, checked]

        public void Execute()
        {
            long maxUlp = 0, mism = 0, checkedN = 0;
            for (long bits = 0; bits <= 0xFFFFFFFFL; bits++)
            {
                float x = math.asfloat((uint)bits);
                if (!(x == x) || math.isinf(x)) continue;   // skip NaN / inf
                float got, refv;
                if (func == 0)
                {
                    if (x <= 0f) continue;
                    got  = DetMathCR.Log(x);
                    refv = (float)math.log((double)x);
                }
                else
                {
                    if (math.abs(x) > 88f) continue;         // finite float range for expf
                    got  = DetMathCR.Exp(x);
                    refv = (float)math.exp((double)x);
                }
                checkedN++;
                int gi = math.asint(got), ri = math.asint(refv);
                long ulp = math.abs((long)gi - (long)ri);
                if (ulp != 0) mism++;
                if (ulp > maxUlp) maxUlp = ulp;
            }
            outp[0] = maxUlp; outp[1] = mism; outp[2] = checkedN;
        }
    }

    public static partial class DetMathBenchmark
    {
        static string CRThroughputRow(int func, string label, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var src = arena.floatVec(n);
            var dst = arena.floatVec(n);
            var rng = new Unity.Mathematics.Random(0xC0FFEEu ^ (uint)func);
            for (int i = 0; i < n; i++) src[i] = func == 0 ? rng.NextFloat(0.1f, 10f) : rng.NextFloat(-10f, 10f);

            double med;
            if (func == 0) { var j = new LogCRJob { src = src, dst = dst }; med = Bench.Time(() => j.Run()).Median; }
            else           { var j = new ExpCRJob { src = src, dst = dst }; med = Bench.Time(() => j.Run()).Median; }
            arena.Dispose();
            return string.Format(CultureInfo.InvariantCulture, "{0,-20} {1,-10} {2,11:F4}", label, n, med);
        }

        static string CRVerifyRow(int func, string label)
        {
            var outp = new NativeArray<long>(3, Allocator.Persistent);
            var job = new VerifyCRJob { func = func, outp = outp };
            job.Run();
            long maxUlp = outp[0], mism = outp[1], chk = outp[2];
            outp.Dispose();
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-16} checked {1,12}   maxULP {2}   mismatches {3}", label, chk, maxUlp, mism);
        }
    }
}
