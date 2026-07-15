using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Transcendental benchmark: native Unity.Mathematics math.* vs prototype in-house DetMath exp.
    //
    // Purpose (exploration, not shipping): measure the cost of the platform transcendentals and how a
    // deterministic-by-construction (+ - * / and bit ops only) polynomial exp compares — on BOTH
    // throughput (batch, vectorizable) and per-call latency (single, dependent chain), for float and
    // double, plus the accuracy the prototype actually achieves. See docs/dev/spec-detmath.md.
    //
    // The prototype exp and the timed IJobs are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DetMathBenchmark.fProxy.cs.
    public static partial class DetMathBenchmark
    {
        // "Calculate 10 million at once" — one large array so the batch loop is throughput-bound and
        // per-call overhead is negligible.
        const int N = 10_000_000;

        static readonly string[] MathFuncs = { "sin", "cos", "exp", "log", "atan" };

        static string ExpHeader()
        {
            return string.Format("{0,-20} {1,-10} {2,11} {3,11} {4,11} {5,13} {6,11}",
                "variant", "N", "min(ms)", "med(ms)", "mean(ms)", "maxRelErr", "~ULP");
        }

        public static void Run() => Bench.WriteReport("benchmark-detmath.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Native math.* throughput — 10M elements, batch loop out[i]=math.f(in[i]) (ms) ===");
            sb.AppendLine("    Unity.Mathematics intrinsics (platform libm / hardware). Batch = independent elements,");
            sb.AppendLine("    Burst free to auto-vectorize. Inputs in [0.5,3] (log needs >0, exp stays finite).");
            sb.AppendLine(Bench.HeaderTime());
            for (int f = 0; f < MathFuncs.Length; f++) sb.AppendLine(MathThroughputFloat(f, MathFuncs[f] + "-f", N));
            for (int f = 0; f < MathFuncs.Length; f++) sb.AppendLine(MathThroughputDouble(f, MathFuncs[f] + "-d", N));
            sb.AppendLine();

            sb.AppendLine("=== exp: native math.exp vs DetMath prototype — 10M, batch + single (ms), max rel err ===");
            sb.AppendLine("    batch  = out[i]=exp(in[i]), independent (throughput, vectorizable).");
            sb.AppendLine("    single = dependent chain acc=exp(in[i]+acc*tiny) (per-call latency, no vectorization).");
            sb.AppendLine("    DetMath = Cody-Waite reduction + poly + ldexp via exponent bits; +-*/ and int/bit only,");
            sb.AppendLine("    no libm call in the path (cross-arch deterministic by construction). acc = accurate poly,");
            sb.AppendLine("    fast = fewer terms. relErr vs System.Math.Exp double reference over inputs in [-10,10].");
            sb.AppendLine(ExpHeader());
            // variant: 0 = math.exp, 1 = DetMath accurate, 2 = DetMath fast.  single: false=batch, true=single.
            sb.AppendLine(ExpRowFloat(0, false, "math.exp   batch-f", N));
            sb.AppendLine(ExpRowFloat(0, true,  "math.exp   single-f", N));
            sb.AppendLine(ExpRowFloat(1, false, "det.acc    batch-f", N));
            sb.AppendLine(ExpRowFloat(1, true,  "det.acc    single-f", N));
            sb.AppendLine(ExpRowFloat(2, false, "det.fast   batch-f", N));
            sb.AppendLine(ExpRowFloat(2, true,  "det.fast   single-f", N));
            sb.AppendLine(ExpRowDouble(0, false, "math.exp   batch-d", N));
            sb.AppendLine(ExpRowDouble(0, true,  "math.exp   single-d", N));
            sb.AppendLine(ExpRowDouble(1, false, "det.acc    batch-d", N));
            sb.AppendLine(ExpRowDouble(1, true,  "det.acc    single-d", N));
            sb.AppendLine(ExpRowDouble(2, false, "det.fast   batch-d", N));
            sb.AppendLine(ExpRowDouble(2, true,  "det.fast   single-d", N));
            sb.AppendLine();
        }
    }
}
