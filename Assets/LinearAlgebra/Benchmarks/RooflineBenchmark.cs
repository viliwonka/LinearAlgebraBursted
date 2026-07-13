using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Machine roofline probe — NOT a library-kernel benchmark. Two ceilings every kernel number
    // in the other sections should be read against:
    //   A/B: compute-bound width-4 mul+add chains in registers (zero memory traffic) under
    //        FloatMode.Default (the library's mode: no FMA contraction) and FloatMode.Fast
    //        (FMA allowed) — the gap between A and B is what Strict-mode determinism costs at
    //        the ALU limit; B approximates the machine's SIMD FLOP ceiling per core.
    //   C:   memory-bound STREAM triad over buffers far beyond L3 — sustained single-core GB/s.
    //
    // Standalone: run with Tools/benchmark.ps1 -Method LinearAlgebra.Benchmarks.RooflineBenchmark.Run
    // (not part of AllBenchmarks).
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per
    // dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/RooflineBenchmark.fProxy.cs.
    public static partial class RooflineBenchmark
    {
        // N column below = millions (of scalar flops for A/B, of elements for C).
        static readonly int[] Millions = { 1, 4, 8, 16, 32, 64 };

        public static void Run() => Bench.WriteReport("benchmark-roofline.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Roofline A: compute-bound, FloatMode.Default — 8 independent width-4 chains, a = a*s + b (N = millions of scalar flops) ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchComputeFloat(m, fast: false));
            foreach (var m in Millions) sb.AppendLine(BenchComputeDouble(m, fast: false));
            sb.AppendLine();

            sb.AppendLine("=== Roofline B: compute-bound, FloatMode.Fast — same chains, FMA contraction allowed ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchComputeFloat(m, fast: true));
            foreach (var m in Millions) sb.AppendLine(BenchComputeDouble(m, fast: true));
            sb.AppendLine();

            sb.AppendLine("=== Roofline C: memory-bound STREAM triad y = x*s + y (N = millions of ELEMENTS; last column reads GB/s, 2 reads + 1 write nominal) ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchMemoryFloat(m));
            foreach (var m in Millions) sb.AppendLine(BenchMemoryDouble(m));
            sb.AppendLine();
        }
    }
}
