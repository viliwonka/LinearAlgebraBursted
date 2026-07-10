using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Cholesky factorization A = L L^T of a symmetric positive-definite matrix. A is taken `in`
    // (never mutated) and L is overwritten each run, so the SPD input is built once and every timed
    // sample does identical work. The input is symmetric + diagonally dominant, which guarantees SPD.
    //
    // The pivoted (rank-revealing) variant P^T A P = L L^T does the same (1/3)N^3 factor work plus
    // rank-revealing bookkeeping (largest-diagonal pivot search, symmetric row/col swaps, and the
    // Schur update's symmetric mirror), so its GFLOP/s is below the plain factorization by design.
    // The n x n working copy is a pre-built workspace; the O(n) Pivot is the only per-Execute alloc.
    //
    // Hand-written harness half. The timed IJobs (CholJob/CholPivotJob {Float,Double}) and
    // build+measure methods (Bench/Pivot {Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/CholeskyBenchmark.fProxy.cs.
    public static partial class CholeskyBenchmark
    {
        // (1/3) N^3 is the standard leading term for Cholesky (half the work of LU).
        static double Flops(int n) => (1.0 / 3.0) * n * (double)n * n;

        // (2/3) N^3 is the standard leading term for LU factorization (duplicated from LUBenchmark's
        // private Flops -- not accessible cross-class -- so the face-off's GFLOP/s column is
        // apples-to-apples against the CHO/CHOP rows above).
        static double LUFlops(int n) => (2.0 / 3.0) * n * (double)n * n;

        // Sizes for the face-off section only -- 256/512/1024 bracket the measured blocking
        // crossover (see Consts.floatCholPivotBlockMinN/doubleCholPivotBlockMinN); 64/128 are skipped
        // to keep this section's runtime down (it already triples the per-size job count of the
        // sections above: CHO + CHOP + LU, all decompInPlace).
        static readonly int[] FaceOffSizes = { 256, 512, 1024 };

        // Single-kernel entry point for A/B runs: writes TestResults/benchmark-chol.txt.
        public static void Run() => Bench.WriteReport("benchmark-chol.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Cholesky factorization A = L L^T (SPD input) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(BenchDouble(n, Flops(n)));
            sb.AppendLine();

            sb.AppendLine("=== Pivoted (rank-revealing) Cholesky P^T A P = L L^T (full-rank SPD input) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(PivotFloat(n, Flops(n)));
            foreach (var n in Bench.Sizes) sb.AppendLine(PivotDouble(n, Flops(n)));
            sb.AppendLine();

            // Face-off: CHO vs CHOP vs LU, all decompInPlace (destructive), same SPD input per size.
            // GFLOP/s is comparable ACROSS the three sub-tables (each uses its own true leading-term
            // flop count), but CHOP's includes rank-revealing bookkeeping (pivot search + symmetric
            // swaps) on top of the same (1/3)N^3 arithmetic as CHO, so a lower CHOP GFLOP/s than CHO at
            // equal N is expected, not a regression.
            sb.AppendLine("=== Face-off: CHO vs CHOP vs LU (decompInPlace, SPD input, N=256/512/1024) ===");
            sb.AppendLine("--- CHO.decompInPlace ---");
            sb.AppendLine(Bench.Header());
            foreach (var n in FaceOffSizes) sb.AppendLine(FaceOffCholFloat(n, Flops(n)));
            foreach (var n in FaceOffSizes) sb.AppendLine(FaceOffCholDouble(n, Flops(n)));
            sb.AppendLine("--- CHOP.decomp (in-place, L aliases A) ---");
            sb.AppendLine(Bench.Header());
            foreach (var n in FaceOffSizes) sb.AppendLine(FaceOffCholPivotFloat(n, Flops(n)));
            foreach (var n in FaceOffSizes) sb.AppendLine(FaceOffCholPivotDouble(n, Flops(n)));
            sb.AppendLine("--- LU.decompInPlace ---");
            sb.AppendLine(Bench.Header());
            foreach (var n in FaceOffSizes) sb.AppendLine(FaceOffLUFloat(n, LUFlops(n)));
            foreach (var n in FaceOffSizes) sb.AppendLine(FaceOffLUDouble(n, LUFlops(n)));
            sb.AppendLine();
        }
    }
}
