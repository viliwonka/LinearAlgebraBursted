using System.Globalization;

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Vectorization experiment (benchmark-only, float). The same 10M-element exp, timed through the
    // floatN[i] indexer vs a hoisted raw float*. Finding: the raw pointer is the whole difference —
    // Burst auto-vectorizes the branch-free scalar loop to full SIMD (floor, the integer exponent-bits
    // 2^n, and the select guard all become vector instructions), so no explicit intrinsics are needed.
    // The floatN indexer is opaque to the vectorizer (record-pointer deref through a struct property),
    // which costs ~8x. Holds for Unity's math.exp too (also vectorizes on a raw pointer). Deterministic:
    // each lane is an independent +-*/-and-floor map, no cross-lane reassociation.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public unsafe struct ExpRawPtrJobFloat : IJob
    {
        [NativeDisableUnsafePtrRestriction] public float* src;
        [NativeDisableUnsafePtrRestriction] public float* dst;
        public int n;
        public int useMath;   // 0 = DetMath exp (pure +-*/ & int/bit), 1 = Unity math.exp

        // Same constants as DetMathProtoFloat.ExpAcc (float path).
        const float INV_LN2 = 1.4426950408889634f;
        const float HI = 0.693359375f;
        const float LO = -2.12194440e-4f;
        const float C6 = 1.38368463709141461e-03f;
        const float C5 = 8.37481579955782172e-03f;
        const float C4 = 4.16682255624844372e-02f;
        const float C3 = 1.66664201699263076e-01f;
        const float C2 = 4.99999920798497477e-01f;
        const float C1 = 1.00000003632318291e+00f;
        const float C0 = 1.00000000055416338e+00f;
        const float OV = 88.72284f;
        const float UN = -87.33655f;

        public void Execute()
        {
            if (useMath == 1)
            {
                for (int i = 0; i < n; i++) dst[i] = math.exp(src[i]);
                return;
            }
            for (int i = 0; i < n; i++)
            {
                float x = src[i];
                float nf = math.floor(INV_LN2 * x + 0.5f);
                float r = x - nf * HI;
                r = r - nf * LO;
                float p = C6;
                p = p * r + C5;
                p = p * r + C4;
                p = p * r + C3;
                p = p * r + C2;
                p = p * r + C1;
                p = p * r + C0;
                int e = (int)nf;
                float y = p * math.asfloat((e + 127) << 23);
                y = x > OV ? float.PositiveInfinity : y;
                y = x < UN ? 0f : y;
                dst[i] = y;
            }
        }
    }

    public static partial class DetMathBenchmark
    {
        public static void WideSection(System.Text.StringBuilder sb)
        {
            sb.AppendLine("=== Vectorization experiment: floatN[i] indexer vs hoisted raw float* — 10M float exp (ms) ===");
            sb.AppendLine("    Same 10M exp, timed four ways. our.exp = DetMath (Cody-Waite + deg-6 Horner + ldexp-by-bits");
            sb.AppendLine("    + guard, pure +-*/ & int/bit). The floatN indexer is opaque to Burst's auto-vectorizer;");
            sb.AppendLine("    a hoisted raw float* lets the branch-free loop vectorize to full SIMD (~8x). Holds for");
            sb.AppendLine("    Unity's math.exp too. maxRelErr vs System.Math.Exp (double ref).");
            sb.AppendLine(string.Format("{0,-26} {1,-10} {2,11} {3,11} {4,11} {5,13}",
                "variant", "N", "min(ms)", "med(ms)", "mean(ms)", "maxRelErr"));
            sb.AppendLine(ExpWideRowFloat("our.exp  floatN[i]", N));
            sb.AppendLine(ExpWideRowFloat("our.exp  raw float*", N));
            sb.AppendLine(ExpWideRowFloat("math.exp floatN[i]", N));
            sb.AppendLine(ExpWideRowFloat("math.exp raw float*", N));
            sb.AppendLine();
        }

        static unsafe string ExpWideRowFloat(string label, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var src = arena.floatVec(n);
            var dst = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(0xE7E7u);
            for (int i = 0; i < n; i++) src[i] = rng.NextFloat(-10f, 10f);

            bool useMath = label.Contains("math.exp");
            Bench.Stat stat;
            if (label.Contains("floatN"))
            {
                var job = new ExpCompareJobFloat { src = src, dst = dst, variant = useMath ? 0 : 1, single = 0 };
                stat = Bench.Time(() => job.Run());
            }
            else
            {
                var job = new ExpRawPtrJobFloat { src = src.Data.Ptr, dst = dst.Data.Ptr, n = n, useMath = useMath ? 1 : 0 };
                stat = Bench.Time(() => job.Run());
            }

            // Correctness sanity: max relative error vs the double libm reference.
            double maxRel = 0.0;
            for (int i = 0; i < n; i++)
            {
                double refv = System.Math.Exp((double)src[i]);
                double rel  = System.Math.Abs((double)dst[i] - refv) / System.Math.Abs(refv);
                if (rel > maxRel) maxRel = rel;
            }
            arena.Dispose();

            return string.Format(CultureInfo.InvariantCulture,
                "{0,-26} {1,-10} {2,11:F4} {3,11:F4} {4,11:F4} {5,13}",
                label, n, stat.Min, stat.Median, stat.Mean, maxRel.ToString("E3", CultureInfo.InvariantCulture));
        }
    }
}
