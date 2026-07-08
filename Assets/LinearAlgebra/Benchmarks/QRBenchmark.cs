using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // QR Householder factorization (also forms Q explicitly). Each Execute copies a pristine source
    // into the working matrix and factors it, so every timed sample does identical work
    // (decompInPlace overwrites its input). The O(N^2) copy against an O(N^3) factorization is
    // < 1% for N >= 128 and is included in the reported time.
    //
    // Hand-written harness half. The timed IJob (QRJob{Float,Double}) and build+measure method
    // (Bench{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QRBenchmark.fProxy.cs.
    public static partial class QRBenchmark
    {
        // (4/3) N^3 is the standard leading term for square Householder QR (approximate; forming Q
        // explicitly adds more, so GFLOP/s here is a lower bound on real work).
        static double Flops(int n) => (4.0 / 3.0) * n * (double)n * n;

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-qr.txt.
        public static void Run() => Bench.WriteReport("benchmark-qr.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== QR Householder factorization (time = copy-in + decompInPlace, forms Q) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n, Flops(n)));
            sb.AppendLine();
        }
    }
}
