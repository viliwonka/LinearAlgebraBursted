namespace LinearAlgebra.Benchmarks
{
    // Default -executeMethod entry point (see Tools/benchmark.ps1). Runs every kernel section into one
    // combined report (TestResults/benchmark-all.txt). For an A/B run of a single kernel, target that
    // kernel's own Run instead, e.g. LinearAlgebra.Benchmarks.QRBenchmark.Run.
    //
    // Reading the report: GFLOP/s should stay roughly flat across N for a cache-friendly kernel; a
    // sharp drop at large N is a cache cliff. float ~2x double on the same kernel means the inner loop
    // vectorises; float ~= double means it does not (headroom). See the per-file headers for each
    // kernel's FLOP model.
    public static class AllBenchmarks
    {
        public static void Run()
        {
            Bench.WriteReport("benchmark-all.txt", sb =>
            {
                KernelBenchmark.Section(sb);
                GemmBenchmark.Section(sb);
                LUBenchmark.Section(sb);
                CholeskyBenchmark.Section(sb);
                QRBenchmark.Section(sb);
                QRVariantsBenchmark.Section(sb);
                TallWideSolveBenchmark.Section(sb);
                DirectSolveBenchmark.Section(sb);
                MultiRhsSolveBenchmark.Section(sb);
                SmallSizeBenchmark.Section(sb);
                EigenSvdBenchmark.Section(sb);
                SvdSolversBenchmark.Section(sb);
                LOBPCGBenchmark.Section(sb);
                IterativeBenchmark.Section(sb);
                SparseSolverBenchmark.Section(sb);
                PCGBenchmark.Section(sb);
                LargeSparseBenchmark.Section(sb);
                KMeansBenchmark.Section(sb);
                FFTBenchmark.Section(sb);
                LPBenchmark.Section(sb);

                sb.AppendLine("GFLOP/s~ uses each kernel's leading-term flop count (approximate): GEMM 2N^3, " +
                              "LU (2/3)N^3, Cholesky (1/3)N^3, QR (4/3)N^3, tall/wide QR-LQ 2cols^2(rows-cols/3).");
            });
        }
    }
}
