using System.Globalization;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of DetMathBenchmark (prototype exp + timed IJobs + row builders). The
    // dtype-agnostic harness (size, section, headers) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/DetMathBenchmark.cs.

    // Prototype in-house exp — benchmark-only, NOT shipped. Deterministic by construction: Cody-Waite
    // argument reduction (x = n·ln2 + r), a polynomial for exp(r) on |r| <= ln2/2, and 2^n assembled
    // directly from the IEEE exponent bits. Only + - * and integer/bit ops — no libm/math.exp in the
    // path, so it is bit-identical across CPU architectures under FloatMode.Strict. Overflow/underflow
    // edge cases are NOT handled (inputs are bounded); a shipping DetMath.Exp would add them.
    internal static class DetMathProtoFProxy
    {
        // Accurate: float = cephes expf minimax (~1 ULP); double = Taylor degree 12 (< 1 ULP on the
        // reduced range). exp(r) = 1 + r + r^2·P(r).
        public static fProxy ExpAcc(fProxy x)
        {
            //+skipFor[double]
            const float INV_LN2 = 1.4426950408889634f;
            const float C1 = 0.693359375f;        // ln2 hi
            const float C2 = -2.12194440e-4f;     // ln2 lo  (C1 + C2 == ln2)
            float n = math.floor(INV_LN2 * x + 0.5f);
            float r = x - n * C1;
            r = r - n * C2;
            float p = 1.9875691500e-4f;
            p = p * r + 1.3981999507e-3f;
            p = p * r + 8.3334519073e-3f;
            p = p * r + 4.1665795894e-2f;
            p = p * r + 1.6666665459e-1f;
            p = p * r + 5.0000001201e-1f;
            float mant = (p * r) * r + r + 1.0f;
            int e = (int)n;
            float scale = math.asfloat((e + 127) << 23);   // 2^n
            return mant * scale;
            //-skipFor
            //+emitFor[double]
            //!const double INV_LN2 = 1.4426950408889634;
            //!const double LN2_HI = 6.9314718036912382e-01;
            //!const double LN2_LO = 1.9082149292705877e-10;
            //!double n = math.floor(INV_LN2 * x + 0.5);
            //!double r = x - n * LN2_HI;
            //!r = r - n * LN2_LO;
            //!double p = 2.0876756987868100e-09;   // 1/12!
            //!p = p * r + 2.5052108385441720e-08;  // 1/11!
            //!p = p * r + 2.7557319223985893e-07;  // 1/10!
            //!p = p * r + 2.7557319223985893e-06;  // 1/9!
            //!p = p * r + 2.4801587301587302e-05;  // 1/8!
            //!p = p * r + 1.9841269841269841e-04;  // 1/7!
            //!p = p * r + 1.3888888888888889e-03;  // 1/6!
            //!p = p * r + 8.3333333333333332e-03;  // 1/5!
            //!p = p * r + 4.1666666666666664e-02;  // 1/4!
            //!p = p * r + 1.6666666666666666e-01;  // 1/3!
            //!p = p * r + 5.0000000000000000e-01;  // 1/2!
            //!double mant = (p * r) * r + r + 1.0;  // 1 + r + r^2·P(r)
            //!long e = (long)n;
            //!double scale = math.asdouble((e + 1023L) << 52);   // 2^n
            //!return mant * scale;
            //-emitFor
        }

        // Fast: fewer terms (float degree-3 Taylor ~6e-4; double degree-6 Taylor ~1e-7), single-word
        // ln2 (no hi/lo split). "Slightly inaccurate" tier.
        public static fProxy ExpFast(fProxy x)
        {
            //+skipFor[double]
            const float INV_LN2 = 1.4426950408889634f;
            const float LN2 = 0.6931471805599453f;
            float n = math.floor(INV_LN2 * x + 0.5f);
            float r = x - n * LN2;
            float mant = 1.6666666e-1f;
            mant = mant * r + 0.5f;
            mant = mant * r + 1.0f;
            mant = mant * r + 1.0f;               // 1 + r + r^2/2 + r^3/6
            int e = (int)n;
            float scale = math.asfloat((e + 127) << 23);
            return mant * scale;
            //-skipFor
            //+emitFor[double]
            //!const double INV_LN2 = 1.4426950408889634;
            //!const double LN2 = 0.6931471805599453;
            //!double n = math.floor(INV_LN2 * x + 0.5);
            //!double r = x - n * LN2;
            //!double mant = 1.3888888888888889e-03;   // 1/6!
            //!mant = mant * r + 8.3333333333333332e-03;  // 1/5!
            //!mant = mant * r + 4.1666666666666664e-02;  // 1/4!
            //!mant = mant * r + 1.6666666666666666e-01;  // 1/3!
            //!mant = mant * r + 5.0000000000000000e-01;  // 1/2!
            //!mant = mant * r + 1.0;
            //!mant = mant * r + 1.0;
            //!long e = (long)n;
            //!double scale = math.asdouble((e + 1023L) << 52);
            //!return mant * scale;
            //-emitFor
        }
    }

    // ---- native math.* throughput (batch) ----
    // func: 0=sin 1=cos 2=exp 3=log 4=atan. Switch is OUTSIDE the loop so each function runs a tight,
    // independently-vectorizable body.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MathFuncThroughputJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;
        public int func;

        public void Execute()
        {
            int n = src.N;
            switch (func)
            {
                case 0: for (int i = 0; i < n; i++) dst[i] = math.sin(src[i]);  break;
                case 1: for (int i = 0; i < n; i++) dst[i] = math.cos(src[i]);  break;
                case 2: for (int i = 0; i < n; i++) dst[i] = math.exp(src[i]);  break;
                case 3: for (int i = 0; i < n; i++) dst[i] = math.log(src[i]);  break;
                default: for (int i = 0; i < n; i++) dst[i] = math.atan(src[i]); break;
            }
        }
    }

    // ---- exp comparison: variant {0 math, 1 det-acc, 2 det-fast} x mode {batch, single} ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ExpCompareJobFProxy : IJob
    {
        public fProxyN src;
        public fProxyN dst;    // batch output (length N); dst[0] holds the sink in single mode
        public int variant;
        public int single;     // 0 = batch, 1 = single (dependent chain)

        public void Execute()
        {
            int n = src.N;
            if (single == 0)
            {
                switch (variant)
                {
                    case 0: for (int i = 0; i < n; i++) dst[i] = math.exp(src[i]);                 break;
                    case 1: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.ExpAcc(src[i]); break;
                    default: for (int i = 0; i < n; i++) dst[i] = DetMathProtoFProxy.ExpFast(src[i]); break;
                }
            }
            else
            {
                // Dependent chain: acc feeds the next argument (scaled to a negligible perturbation) so
                // the compiler cannot vectorize across iterations — this measures per-call latency.
                fProxy acc = (fProxy)0;
                fProxy tiny = (fProxy)1e-20;
                switch (variant)
                {
                    case 0: for (int i = 0; i < n; i++) acc = math.exp(src[i] + acc * tiny);                 break;
                    case 1: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.ExpAcc(src[i] + acc * tiny); break;
                    default: for (int i = 0; i < n; i++) acc = DetMathProtoFProxy.ExpFast(src[i] + acc * tiny); break;
                }
                dst[0] = acc;
            }
        }
    }

    public static partial class DetMathBenchmark
    {
        static string MathThroughputFProxy(int func, string label, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var src = arena.fProxyVec(n);
            var dst = arena.fProxyVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)func ^ 0x9E3779B9u);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(0.5f, 3f);

            var job = new MathFuncThroughputJobFProxy { src = src, dst = dst, func = func };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime(label, n, stat);
        }

        static string ExpRowFProxy(int variant, bool single, string label, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var src = arena.fProxyVec(n);
            var dst = arena.fProxyVec(n);

            var rng = new Unity.Mathematics.Random(0xB16B00B5u ^ (uint)variant);
            for (int i = 0; i < n; i++) src[i] = rng.NextFProxy(-10f, 10f);

            var job = new ExpCompareJobFProxy { src = src, dst = dst, variant = variant, single = single ? 1 : 0 };
            var stat = Bench.Time(() => job.Run());

            // Accuracy (batch only — single overwrites just dst[0]): max relative error vs a double
            // System.Math.Exp reference over the whole input. Computed once, untimed.
            double maxRel = 0.0;
            if (!single)
            {
                for (int i = 0; i < n; i++)
                {
                    double refv = System.Math.Exp((double)src[i]);
                    double got  = (double)dst[i];
                    double rel  = System.Math.Abs(got - refv) / System.Math.Abs(refv);
                    if (rel > maxRel) maxRel = rel;
                }
            }
            arena.Dispose();

            double eps = /*+choose[1.1920929e-7|2.220446049250313e-16]*/1.1920929e-7/*-choose*/;
            string relStr = single ? "(chain)" : maxRel.ToString("E3", CultureInfo.InvariantCulture);
            string ulpStr = single ? "-" : (maxRel / eps).ToString("F2", CultureInfo.InvariantCulture);
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-20} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4} {5,13} {6,11}",
                label, n, stat.Min, stat.Median, stat.Mean, relStr, ulpStr);
        }
    }
}
