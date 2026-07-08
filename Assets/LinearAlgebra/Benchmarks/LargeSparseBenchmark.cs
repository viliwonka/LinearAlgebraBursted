using System.Text;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic table formatters for LargeSparseBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can call them.
    public static class LargeSparseFmt
    {
        public static string StatusName(int s) => s == 0 ? "Converged" : s == 1 ? "MaxIter" : s == 2 ? "Breakdown" : "THREW";

        public static void LobRow(StringBuilder sb, string dtype, string grid, string precond, int guard, double[] o) =>
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-10} {3,-6} {4,-12} {5,7} {6,6} {7,13:E3} {8,13:E3}",
                dtype, grid, precond, guard, StatusName((int)o[0]), (int)o[1], (int)o[2], o[3], o[4]));

        public static double[] Snap(NativeArray<double> o) => new[] { o[0], o[1], o[2], o[3], o[4] };

        public static string Row(string dtype, string size, string solver, Bench.Stat st, double residual) =>
            string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-12} {3,11:F4} {4,11:F4} {5,14:E3}", dtype, size, solver, st.Median, st.Min, residual);

        public static string EigRow(string dtype, string size, string solver, Bench.Stat st, double[] info) =>
            string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-12} {3,11:F4} {4,11:F4} {5,8} {6,10} {7,14:E3}", dtype, size, solver, st.Median, st.Min, (int)info[0], (int)info[1], info[2]);
    }

    // LARGE sparse solvers at the scale where they are the only option: N up to 10240 at ~1.5% block
    // fill, built via the sparse gallery with NO dense twin (a dense 10240^2 float matrix is ~420 MB and
    // its O(N^3) factor/eig is minutes -- exactly the regime where iterative sparse solvers earn their keep).
    // Every Krylov timing is a FIXED K iterations at tol=0 (deterministic timing; the residual column shows
    // how converged, not just how fast). All workspace is pre-allocated ONCE and reused across timed
    // samples; every solve runs inside a [BurstCompile] IJob.
    //   1. spMV throughput; 2. square SPD (cg/pcg/minres); 3. square non-symmetric (biCGStab);
    //   4. tall rectangular least-squares m=2n (cgls/lsqr/lsmr); 5. Lanczos (throughput);
    //   6. LOBPCG smallest-k eigenpairs on a spread-spectrum grid Laplacian (precond x guard levers).
    //
    // Hand-written harness half. The timed IJobs, the LOBPCG report helper, the residual helper, and the
    // per-family build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LargeSparseBenchmark.fProxy.cs.
    public static partial class LargeSparseBenchmark
    {
        const int BR = 4;
        static readonly int[] Ns = { 2048, 5120, 10240 };
        const float Density = 0.015f;
        const int K = 40;
        const int SpmvReps = 50;
        const int LanczosSteps = 32;

        // LOBPCG smallest-eigenpair rows run on SQUARE 2D grid Laplacians (fProxyLaplacian2D, BR = grid),
        // whose spectrum is SPREAD (with exact multiplicities). Reported metric is ITERATIONS (deterministic;
        // one from-scratch solve per row), swept over preconditioner (none / block-Jacobi) x guard (0 / 8).
        static readonly int[] EigGrids = { 32, 64, 96 };   // n = g*g -> 1024, 4096, 9216; block size BR = g
        const int LobpcgK = 8;
        const int LobpcgGuard = 8;
        const int LobpcgMaxIter = 800;

        public static void Run() => Bench.WriteReport("benchmark-largesparse.txt", Section);

        public static void RunLobpcg() => Bench.WriteReport("benchmark-largesparse-lobpcg.txt", LobpcgSection);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== LARGE sparse solvers (BSR, ~1.5% block fill, b=4), N up to 10240 -- no dense form ===");
            sb.AppendLine("Krylov rows: K=" + K + " fixed iterations, tol=0 (deterministic timing); residual = ||Ax-b||/||b|| after K.");
            sb.AppendLine("At N=10240 a dense matrix is ~420 MB (float) and O(N^3) dense factor/eig is minutes -- these solvers");
            sb.AppendLine("touch only the ~1.5% nonzero blocks, so this scale is exactly where dense is not an option.");
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-12} {3,11} {4,11} {5,14}", "dtype", "size", "solver", "med(ms)", "min(ms)", "residual"));
            BenchKrylovFloat(sb, BR, Ns, Density, K, SpmvReps);
            sb.AppendLine();
            BenchKrylovDouble(sb, BR, Ns, Density, K, SpmvReps);
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-12} {3,11} {4,11} {5,8} {6,10} {7,14}", "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "converged", "maxResid"));
            BenchEigenFloat(sb, BR, Ns, Density, LanczosSteps);
            sb.AppendLine();
            BenchEigenDouble(sb, BR, Ns, Density, LanczosSteps);
            sb.AppendLine();
            LobpcgSection(sb);
        }

        // LOBPCG smallest-k eigenpairs of a large sparse 2D grid Laplacian, sweeping preconditioner
        // (none / block-Jacobi) and guard-vector count (0 / LobpcgGuard). Reported metric is ITERATIONS,
        // not wall-clock (the BR=grid dense-block encoding would dominate any timing and say more about
        // the encoding than the solver). Findings: block-Jacobi cuts iterations ~30%; guards cut them ~2x
        // (at higher per-iteration cost, so guards are a robustness/iteration lever, not a wall-clock win);
        // the two stack. orthoErr confirms the output stays orthonormal in every config.
        public static void LobpcgSection(StringBuilder sb)
        {
            sb.AppendLine("=== LOBPCG smallest-" + LobpcgK + " eigenpairs, square 2D grid Laplacian (spread spectrum) ===");
            sb.AppendLine("Levers: preconditioner (none / block-Jacobi) x guard (0 / " + LobpcgGuard + "). maxIter=" + LobpcgMaxIter + ", tol=sqrt(eps).");
            sb.AppendLine("Metric is ITERATIONS (deterministic, one solve/row); wall-clock omitted -- it is dominated by the");
            sb.AppendLine("BR=grid dense-block encoding, not the solver. orthoErr = max_ij |X_i.X_j - d_ij| over the k wanted.");
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-10} {3,-6} {4,-12} {5,7} {6,6} {7,13} {8,13}",
                "dtype", "grid(n)", "precond", "guard", "status", "iters", "conv", "maxResid", "orthoErr"));
            BenchLobpcgFloat(sb, EigGrids, LobpcgK, LobpcgGuard, LobpcgMaxIter);
            sb.AppendLine();
            BenchLobpcgDouble(sb, EigGrids, LobpcgK, LobpcgGuard, LobpcgMaxIter);
            sb.AppendLine();
        }
    }
}
