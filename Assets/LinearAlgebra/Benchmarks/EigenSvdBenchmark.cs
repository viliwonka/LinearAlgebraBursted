using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Production spectral algorithms: Golub-Kahan SVD (full + values-only), Householder symmetric
    // eigen (values + vectors), and the general-matrix QR-iteration eigenvalues (elmhes + Francis hqr).
    // Their cost is data-dependent (iteration / sweep count), so only ms is reported. Each Execute
    // copies a pristine source so every timed sample does identical (and identically-converging) work.
    //
    // Both float and double variants are benched so the float/double timing ratio diagnoses
    // SIMD vectorisation: a vectorised float path should run ~1.5-2x faster than double;
    // non-vectorised paths run at roughly equal speed.
    //
    // The deprecated one-sided-Jacobi svdDecomposition and cyclic-Jacobi decompInPlace are NOT
    // benched here: they are [Obsolete], redundant with the above, and O(sweeps*n^3) with strided
    // column access. Their historical comparison numbers live in git history if ever needed again.
    //
    // Hand-written harness half. The timed IJobs (SvdValues/EigSym/EigSymVec/EigQR/SvdGK Job
    // {Float,Double}) and build+measure methods (SvdGK/SvdVals/EigSym/EigSymVec/EigQR {Float,Double})
    // are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/EigenSvdBenchmark.fProxy.cs.
    public static partial class EigenSvdBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-eigensvd.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Golub-Kahan full SVD (thin; bidiag + implicit-shift QR, ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdGKFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdGKDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== SVD singular values only (values, Golub-Kahan bidiagonal; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdValsFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(SvdValsDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Householder symmetric eigenvalues (valuesSymmetric; values only, ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(EigSymFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(EigSymDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Householder symmetric eigen + vectors (symmetric; ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(EigSymVecFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(EigSymVecDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== General eigenvalues, QR iteration (valuesQR; iterative, ms only) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in Bench.Sizes) sb.AppendLine(EigQRFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(EigQRDouble(n));
            sb.AppendLine();
        }
    }
}
