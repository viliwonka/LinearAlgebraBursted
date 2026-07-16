using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Eigen.lobpcg (k smallest eigenpairs of a symmetric operator) — not covered by
    // EigenSvdBenchmark.cs (that file predates LOBPCG's generalization). Dense SPD input
    // A = MᵀM + I (same recipe as IterativeBenchmark.cs — guarantees SPD, well-conditioned smallest
    // eigenvalues clustered near 1). tol is set near machine-epsilon so every timed sample runs the
    // full maxIter budget (deterministic timing, mirroring the fixed-K convention used by the other
    // iterative-solver benchmarks); iterations/converged/maxResidual are reported alongside timing to
    // show how far that fixed budget actually gets, not just how fast.

    // info is written into a length-1 NativeArray, not a plain struct field: IJob.Run() executes on
    // an internal copy of the job struct, so a plain value-type field written inside Execute() is NOT
    // visible on the caller's job variable afterwards (only pointer-backed data — NativeArray/Arena
    // buffers — survives the copy). Every other benchmark in this folder sidesteps this by only ever
    // reading back Arena-backed buffers (floatN/floatMxN); LOBPCGInfo has no such buffer to piggyback
    // on, so it gets its own one-element NativeArray output.

    // This is the hand-written harness half. The timed IJob (LobpcgJob{Float,Double}) and the
    // build+measure method (Bench{Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LOBPCGBenchmark.fProxy.cs.
    public static partial class LOBPCGBenchmark
    {
        const int N = 512;
        const int K = 4;         // smallest eigenpairs requested
        const int MaxIter = 50;  // fixed budget for deterministic timing

        public static void Run() => Bench.WriteReport("benchmark-lobpcg.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine(string.Format("=== Eigen.lobpcg, dense SPD (A = M^T M + I), k={0} smallest, maxIter={1} (ms) ===", K, MaxIter));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,11} {3,11} {4,10} {5,10} {6,14}",
                "dtype", "N", "min(ms)", "med(ms)", "iters", "converged", "maxResidual"));
            sb.AppendLine(BenchFloat(N, K, MaxIter));
            sb.AppendLine(BenchDouble(N, K, MaxIter));
            sb.AppendLine(BenchFloat(1024, K, MaxIter));
            sb.AppendLine(BenchDouble(1024, K, MaxIter));
            sb.AppendLine();

            sb.AppendLine(string.Format("=== lobpcg preconditioner face-off, sparse BSR, k={0} smallest, solve to tol=sqrt(eps) ===", K));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,7} {6,10} {7,14}",
                "dtype", "N", "precond", "med(ms)", "min(ms)", "iters", "converged", "maxResidual"));
            sb.AppendLine(BenchSparsePrecondFloat(true, 4, 256, 0f, 0u, K));
            sb.AppendLine(BenchSparsePrecondDouble(true, 4, 256, 0f, 0u, K));
            sb.AppendLine(BenchSparsePrecondFloat(false, 120, 3, 0.30f, 0xC004Du, K));
            sb.AppendLine(BenchSparsePrecondDouble(false, 120, 3, 0.30f, 0xC004Du, K));
            sb.AppendLine();
        }
    }
}
