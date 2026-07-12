using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // The end-to-end "solve Ax=b" entry points, as opposed to decompositions.md's factorization-only
    // benchmarks: LU.decompSolve, LU.decompInPlace+decompSolveTransA, CHO.decomp+decompSolve,
    // QR.solveInPlace (square). Each Execute copies a
    // pristine source into the working buffers (factorization/solve destroys them), so every timed
    // sample does identical work. solvers.md notes the triangular-solve step itself is O(n^2), dominated
    // by the O(n^3) factorization in every case here — these numbers are effectively the factorization
    // cost plus a negligible solve. The QR-cache variant isolates the Temp-alloc-elimination win.
    //
    // Hand-written harness half. The timed IJobs (LuSolve/CholSolve/QrSquareSolve/QrSquareSolveCache Job
    // {Float,Double}) and build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DirectSolveBenchmark.fProxy.cs.
    public static partial class DirectSolveBenchmark
    {
        // One representative square size — the O(n^2) triangular solve is negligible next to the
        // O(n^3) factorization at this scale (see decompositions.md / solvers.md).
        const int N = 1024;

        public static void Run() => Bench.WriteReport("benchmark-directsolve.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Direct solve Ax=b, square N=" + N + " (factor + triangular solve, ms) ===");
            sb.AppendLine(Bench.HeaderTime());
            sb.AppendLine(LuSolveFloat(N));
            sb.AppendLine(LuSolveDouble(N));
            // TransA row (LU.decompInPlace + LU.decompSolveTransA, compact form): expected roughly on
            // par with the forward LU row above -- both triangular passes are axpy-shaped (right-
            // looking) in their own direction, so neither should out-vectorise the other; the O(n^3)
            // factorization dominates either way at this N (see the class doc comment).
            sb.AppendLine(LuSolveTransAFloat(N));
            sb.AppendLine(LuSolveTransADouble(N));
            sb.AppendLine(CholSolveFloat(N));
            sb.AppendLine(CholSolveDouble(N));
            sb.AppendLine(QrSolveFloat(N));
            sb.AppendLine(QrSolveDouble(N));
            sb.AppendLine(QrSolveCacheFloat(N));
            sb.AppendLine(QrSolveCacheDouble(N));
            sb.AppendLine();
        }
    }
}
