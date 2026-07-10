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

        // Krylov R3b: grew a wall-clock column (med(ms), from Bench.Time -- see
        // SpLobpcgJobFProxy's doc comment for why every sample re-zeroes ws.X first).
        public static void LobRow(StringBuilder sb, string dtype, string grid, string precond, int guard, Bench.Stat st, double[] o) =>
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-10} {3,-6} {4,11:F4} {5,-12} {6,7} {7,6} {8,13:E3} {9,13:E3}",
                dtype, grid, precond, guard, st.Median, StatusName((int)o[0]), (int)o[1], (int)o[2], o[3], o[4]));

        public static double[] Snap(NativeArray<double> o) => new[] { o[0], o[1], o[2], o[3], o[4] };

        // iters/status: the LAST timed sample's SolveInfo/LstsqInfo (fixed K=40, tol=0 -> every
        // timed sample of the same job is deterministic, so any one of them is representative).
        // Exposes the benchmark-hygiene case a wall-clock-only column hides: a solver that exits via
        // a breakdown guard partway through K iterations looks "fast" for having done LESS work, not
        // more of it -- see docs/draft-spec-krylov-optimization.md's benchmark hygiene note.
        public static string Row(string dtype, string size, string solver, Bench.Stat st, double residual, int iters, int status) =>
            string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-12} {2,-12} {3,11:F4} {4,11:F4} {5,14:E3} {6,6} {7,12}", dtype, size, solver, st.Median, st.Min, residual, iters, StatusName(status));

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
    //   4. tall rectangular least-squares m=2n (cgls/lsqr/lsmr); 5. b=1 scalar stencil SPD
    //   (cg/pcg/minres, low-fill R1-fusion visibility); 6. Lanczos (throughput);
    //   7. LOBPCG smallest-k eigenpairs on a spread-spectrum grid Laplacian (precond x guard levers).
    // Every Krylov row also reports the LAST timed sample's iterations+status (fixed K/tol=0 can exit
    // early via a breakdown guard, which looks "fast" for doing less work -- see StencilSection).
    //
    // Hand-written harness half. The timed IJobs, the LOBPCG report helper, the residual helper, and the
    // per-family build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LargeSparseBenchmark.fProxy.cs.
    public static partial class LargeSparseBenchmark
    {
        const int BR = 4;
        // N=2048 dropped (Q7 budget ruling): the float==double diagnostic and the BR=4 fill-level
        // trend are already fully visible at 5120/10240 -- keeping a 3rd, smallest, LEAST informative
        // size added rows without adding information. Freed budget pays for the new stencil section
        // below (BenchStencilFloat/Double) and the iters+status columns on every Krylov row.
        static readonly int[] Ns = { 5120, 10240 };
        // Krylov R3 (Q7 budget ruling): the b=1 stencil section runs at N=10240 only (see
        // StencilSection) -- paid for by dropping this size there, not by an SpmvReps-style knob
        // (spMV x50's own throughput row was deleted, see BenchKrylovFProxy's comment).
        static readonly int[] StencilNs = { 10240 };
        const float Density = 0.015f;
        const int K = 40;
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

        static string KrylovHeader() =>
            string.Format("{0,-7} {1,-12} {2,-12} {3,11} {4,11} {5,14} {6,6} {7,12}",
                "dtype", "size", "solver", "med(ms)", "min(ms)", "residual", "iters", "status");

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== LARGE sparse solvers (BSR, ~1.5% block fill, b=4), N up to 10240 -- no dense form ===");
            sb.AppendLine("Krylov rows: K=" + K + " fixed iterations, tol=0 (deterministic timing); residual = ||Ax-b||/||b|| after K.");
            sb.AppendLine("iters/status: the last timed sample's SolveInfo/LstsqInfo -- a fixed-K/tol=0 row that exits via a");
            sb.AppendLine("breakdown guard did LESS work than one that ran the full K, even though both report the same K budget.");
            sb.AppendLine("At N=10240 a dense matrix is ~420 MB (float) and O(N^3) dense factor/eig is minutes -- these solvers");
            sb.AppendLine("touch only the ~1.5% nonzero blocks, so this scale is exactly where dense is not an option.");
            sb.AppendLine();
            sb.AppendLine(KrylovHeader());
            BenchKrylovFloat(sb, BR, Ns, Density, K);
            sb.AppendLine();
            BenchKrylovDouble(sb, BR, Ns, Density, K);
            sb.AppendLine();
            StencilSection(sb);
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-12} {3,11} {4,11} {5,8} {6,10} {7,14}", "dtype", "N", "solver", "med(ms)", "min(ms)", "iters", "converged", "maxResid"));
            BenchEigenFloat(sb, BR, Ns, Density, LanczosSteps);
            sb.AppendLine();
            BenchEigenDouble(sb, BR, Ns, Density, LanczosSteps);
            sb.AppendLine();
            LobpcgSection(sb);
        }

        // b=1 (scalar BSR) stencil section (R1 fusion spec, Q7): fProxyLaplacian2D(1, N) is a genuine
        // SCALAR (BR=1) tridiagonal SPD system (diag=4, off-diag=-1, nnz ~= 3N) -- the low-fill regime
        // where vector-op sweeps are the largest fraction of per-iteration traffic (vs BR=4/1.5% fill
        // above, where spMV dominates), so R1's fusion should be most visible here. Only CG/PCG-Jacobi/
        // MINRES/PCG-SSOR run (SPD-only; BiCGStab/CGLS/LSQR/LSMR need non-symmetric/rectangular
        // operators this generator does not produce). N=10240 only (StencilNs, Krylov R3 Q7
        // budget ruling) -- see StencilNs's own comment; the BR=4 section above still sweeps both
        // Ns for direct float==double comparison at more than one size.
        static void StencilSection(StringBuilder sb)
        {
            sb.AppendLine("=== b=1 stencil (scalar BSR, fProxyLaplacian2D(1,N): tridiag SPD, nnz~=3N) -- vector-op fusion visibility ===");
            sb.AppendLine();
            sb.AppendLine(KrylovHeader());
            BenchStencilFloat(sb, StencilNs, K);
            sb.AppendLine();
            BenchStencilDouble(sb, StencilNs, K);
            sb.AppendLine();
        }

        // LOBPCG smallest-k eigenpairs of a large sparse 2D grid Laplacian, sweeping preconditioner
        // (none / block-Jacobi / SSOR -- Krylov R3b) and guard-vector count (0, and block-Jacobi
        // also at LobpcgGuard -- the "none"+guard combination was dropped to pay for the new SSOR
        // row, see BenchLobpcgFProxy's comment; none/g0 -> blockJac/g0 -> blockJac/gG still shows
        // both the precond-alone and the precond+guard-stacking stories). Reported metrics are
        // ITERATIONS (deterministic, same seed every timed sample) AND wall-clock (R3b: added to
        // test whether SSOR's iteration cut wins wall despite its OWN apply costing 2-4x
        // block-Jacobi's per R3 -- LOBPCG's per-iteration cost may be dominated by Rayleigh-Ritz
        // work rather than the preconditioner apply, which the earlier "wall-clock omitted, it's
        // dominated by the BR=grid dense-block encoding" note undersold). orthoErr = max_ij
        // |X_i.X_j - d_ij| over the k wanted.
        public static void LobpcgSection(StringBuilder sb)
        {
            sb.AppendLine("=== LOBPCG smallest-" + LobpcgK + " eigenpairs, square 2D grid Laplacian (spread spectrum) ===");
            sb.AppendLine("Levers: preconditioner (none / block-Jacobi / SSOR) x guard (0, and block-Jacobi also at " + LobpcgGuard + "). maxIter=" + LobpcgMaxIter + ", tol=sqrt(eps).");
            sb.AppendLine("Metrics are ITERATIONS (deterministic, one solve/row) AND wall-clock (med(ms), Bench.Time: 1 warmup + 4 timed,");
            sb.AppendLine("ws.X re-zeroed every sample so each is a fair cold start). orthoErr = max_ij |X_i.X_j - d_ij| over the k wanted.");
            sb.AppendLine();
            sb.AppendLine(string.Format("{0,-7} {1,-12} {2,-10} {3,-6} {4,11} {5,-12} {6,7} {7,6} {8,13} {9,13}",
                "dtype", "grid(n)", "precond", "guard", "med(ms)", "status", "iters", "conv", "maxResid", "orthoErr"));
            BenchLobpcgFloat(sb, EigGrids, LobpcgK, LobpcgGuard, LobpcgMaxIter);
            sb.AppendLine();
            BenchLobpcgDouble(sb, EigGrids, LobpcgK, LobpcgGuard, LobpcgMaxIter);
            sb.AppendLine();
        }
    }
}
