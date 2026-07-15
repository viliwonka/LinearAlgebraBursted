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
            sb.AppendLine("    DetMath = Cody-Waite reduction + minimax poly + ldexp via exponent bits; +-*/ and int/bit");
            sb.AppendLine("    only, no libm call in the path (cross-arch deterministic by construction). acc = accurate");
            sb.AppendLine("    minimax; Horner = sequential, Estrin = balanced-tree regroup (shorter dependency chain →");
            sb.AppendLine("    lower latency); fast = fewer terms. relErr vs System.Math.Exp double ref, inputs [-10,10].");
            sb.AppendLine(ExpHeader());
            // variant: 0 = math.exp, 1 = det.acc Horner, 2 = det.acc Estrin, 3 = det.fast.  single: batch/latency.
            sb.AppendLine(ExpRowFloat(0, false, "math.exp   batch-f", N));
            sb.AppendLine(ExpRowFloat(0, true,  "math.exp   single-f", N));
            sb.AppendLine(ExpRowFloat(1, false, "det.acc.H  batch-f", N));
            sb.AppendLine(ExpRowFloat(1, true,  "det.acc.H  single-f", N));
            sb.AppendLine(ExpRowFloat(2, false, "det.acc.E  batch-f", N));
            sb.AppendLine(ExpRowFloat(2, true,  "det.acc.E  single-f", N));
            sb.AppendLine(ExpRowFloat(3, false, "det.fast   batch-f", N));
            sb.AppendLine(ExpRowFloat(3, true,  "det.fast   single-f", N));
            sb.AppendLine(ExpRowDouble(0, false, "math.exp   batch-d", N));
            sb.AppendLine(ExpRowDouble(0, true,  "math.exp   single-d", N));
            sb.AppendLine(ExpRowDouble(1, false, "det.acc.H  batch-d", N));
            sb.AppendLine(ExpRowDouble(1, true,  "det.acc.H  single-d", N));
            sb.AppendLine(ExpRowDouble(2, false, "det.acc.E  batch-d", N));
            sb.AppendLine(ExpRowDouble(2, true,  "det.acc.E  single-d", N));
            sb.AppendLine(ExpRowDouble(3, false, "det.fast   batch-d", N));
            sb.AppendLine(ExpRowDouble(3, true,  "det.fast   single-d", N));
            sb.AppendLine();

            sb.AppendLine("=== sin/cos: native math vs DetMath minimax — 10M, batch + single (ms), max ABS err ===");
            sb.AppendLine("    Cody-Waite pi/2 reduction + odd/even minimax (sin=r*P(r^2), cos=Q(r^2)) + branch-free");
            sb.AppendLine("    quadrant select. Deterministic (+-*/ & int only). absErr vs System.Math, inputs [-10,10].");
            sb.AppendLine(ExpHeader());
            // variant: 0 math.sin, 1 det.sin, 2 math.cos, 3 det.cos.
            sb.AppendLine(TrigRowFloat(0, false, "math.sin   batch-f", N));
            sb.AppendLine(TrigRowFloat(0, true,  "math.sin   single-f", N));
            sb.AppendLine(TrigRowFloat(1, false, "det.sin    batch-f", N));
            sb.AppendLine(TrigRowFloat(1, true,  "det.sin    single-f", N));
            sb.AppendLine(TrigRowFloat(2, false, "math.cos   batch-f", N));
            sb.AppendLine(TrigRowFloat(3, false, "det.cos    batch-f", N));
            sb.AppendLine(TrigRowDouble(0, false, "math.sin   batch-d", N));
            sb.AppendLine(TrigRowDouble(0, true,  "math.sin   single-d", N));
            sb.AppendLine(TrigRowDouble(1, false, "det.sin    batch-d", N));
            sb.AppendLine(TrigRowDouble(1, true,  "det.sin    single-d", N));
            sb.AppendLine(TrigRowDouble(2, false, "math.cos   batch-d", N));
            sb.AppendLine(TrigRowDouble(3, false, "det.cos    batch-d", N));
            sb.AppendLine();

            sb.AppendLine("=== log: native math.log vs DetMath minimax — 10M, batch + single (ms), max ABS err ===");
            sb.AppendLine("    x = m*2^e via exponent bits, m centered to [sqrt2/2,sqrt2), log(m)=2s*B(s^2), s=(m-1)/(m+1).");
            sb.AppendLine("    Deterministic (+-*/ & int/bit only). absErr vs System.Math.Log, inputs (0.1,10].");
            sb.AppendLine(ExpHeader());
            sb.AppendLine(LogRowFloat(0, false, "math.log   batch-f", N));
            sb.AppendLine(LogRowFloat(0, true,  "math.log   single-f", N));
            sb.AppendLine(LogRowFloat(1, false, "det.log    batch-f", N));
            sb.AppendLine(LogRowFloat(1, true,  "det.log    single-f", N));
            sb.AppendLine(LogRowDouble(0, false, "math.log   batch-d", N));
            sb.AppendLine(LogRowDouble(0, true,  "math.log   single-d", N));
            sb.AppendLine(LogRowDouble(1, false, "det.log    batch-d", N));
            sb.AppendLine(LogRowDouble(1, true,  "det.log    single-d", N));
            sb.AppendLine();

            sb.AppendLine("=== atan: native math.atan vs DetMath minimax — 10M, batch + single (ms), max ABS err ===");
            sb.AppendLine("    Fold |x|>1 to [0,1] (atan(x)=pi/2-atan(1/x)), split at tan(pi/8) (atan=pi/4+atan((x-1)/");
            sb.AppendLine("    (x+1))), odd minimax atan(xr)=xr*P(xr^2). Deterministic, branch-free. absErr vs System.Math.");
            sb.AppendLine(ExpHeader());
            sb.AppendLine(AtanRowFloat(0, false, "math.atan  batch-f", N));
            sb.AppendLine(AtanRowFloat(0, true,  "math.atan  single-f", N));
            sb.AppendLine(AtanRowFloat(1, false, "det.atan   batch-f", N));
            sb.AppendLine(AtanRowFloat(1, true,  "det.atan   single-f", N));
            sb.AppendLine(AtanRowDouble(0, false, "math.atan  batch-d", N));
            sb.AppendLine(AtanRowDouble(0, true,  "math.atan  single-d", N));
            sb.AppendLine(AtanRowDouble(1, false, "det.atan   batch-d", N));
            sb.AppendLine(AtanRowDouble(1, true,  "det.atan   single-d", N));
            sb.AppendLine();
        }
    }
}
