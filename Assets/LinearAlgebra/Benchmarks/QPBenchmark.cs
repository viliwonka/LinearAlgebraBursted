using System.Globalization;
using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic config + table formatters for QPBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes and row writer.
    public static class QPBenchmarkFmt
    {
        // Random SPD QP sizes: n variables, m = n/2 general (LessEqual) constraints -- mirroring
        // LPBenchmarkFmt.SolveVarsN's "wide, comfortably feasible" construction. Three sizes, capped at
        // 192 (T = m+n = 288 rows in the unified working-set system), so the whole section -- six solves
        // total (three sizes x two dtypes), each through the FULL public QP.solve facade (validation +
        // phase-1 LP + active-set core) -- stays well under a minute. The per-iteration cost
        // (one QR of nxk + one Cholesky + GEMMs, O(n^3) worst case) puts even
        // the n=192 row's worst-case iteration cost at a few million flops, and a well-conditioned random
        // SPD Q with a comfortably-feasible polytope converges in a handful to a few dozen iterations
        // (consistent with the HS/BruteForce oracle tests, all under 20-30 iterations at far smaller n).
        public static readonly int[] SizesN = { 16, 64, 192 };

        public static string StatusName(QPStatus s) => s == QPStatus.Optimal ? "Optimal"
            : s == QPStatus.Infeasible ? "Infeasible" : s == QPStatus.Unbounded ? "Unbounded" : "MaxIter";

        public static string Header() => string.Format("{0,-7} {1,-6} {2,-6} {3,11} {4,11} {5,7} {6,14} {7,10} {8,14}",
            "dtype", "n", "m", "med(ms)", "min(ms)", "iters", "KKT resid", "status", "objective");

        // `status` crosses the hand-written/template assembly boundary as a raw int (the job writes
        // `(int)info.status`, Burst-legal enum-to-int cast) rather than the enum itself, for the exact
        // CS0012 reason LPBenchmarkFmt.InfeasRow's own doc comment explains (the generated
        // TemplateSourceBenchmarks firstpass compile has its own LOCAL QPStatus, distinct from this
        // hand-written assembly's) -- StatusName (this assembly's real QPStatus) belongs here, one cast
        // away from the raw int.
        public static string Row(string dtype, int n, int m, Bench.Stat st, int iters, double kkt, int status, double obj) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,11:F4} {4,11:F4} {5,7} {6,14:E4} {7,10} {8,14:E4}",
                dtype, n, m, st.Median, st.Min, iters, kkt, StatusName((QPStatus)status), obj);
    }

    // Formatter for Section 2 (the loop-isolating core benchmark: an extra `reduced` column naming the
    // incremental-vs-batch variant, and no KKT column -- correctness is Section 1's / the test suite's
    // job; this section is purely the stage-2 A/B timing gate).
    public static class QPCoreLoopFmt
    {
        public static string Header() => string.Format("{0,-7} {1,-6} {2,-6} {3,-6} {4,11} {5,11} {6,7} {7,10} {8,14}",
            "dtype", "reduced", "n", "m", "med(ms)", "min(ms)", "iters", "status", "objective");

        public static string Row(string dtype, string reduced, int n, int m, Bench.Stat st, int iters, int status, double obj) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-6} {4,11:F4} {5,11:F4} {6,7} {7,10} {8,14:E4}",
                dtype, reduced, n, m, st.Median, st.Min, iters, QPBenchmarkFmt.StatusName((QPStatus)status), obj);
    }

    // ================================================================================================
    // Convex quadratic programming benchmark.
    //
    //   Section 1 (QP.solve, random SPD QP): random dense feasible+bounded convex QPs
    //     (Q symmetric PSD via Rand.spdInPlace with a modest condition
    //     number ~10, A >= 0, b = A x0 + slack so a comfortably-feasible region exists, x boxed in
    //     [0, 3]) solved through the FULL PUBLIC facade -- QP.solve, no caller-supplied starting point,
    //     so every row exercises phase 1 (the LP-powered feasible start) as
    //     well as the active-set loop. The KKT-residual column is recomputed FRESH from the returned x
    //     inside the timed job using only PUBLIC data (Q, c, A, b, senses, xl, xu, x) -- see
    //     QpSolveJobFProxy's own comment for exactly what it captures: full primal feasibility (general
    //     rows AND box, exact) plus a box-only projected-gradient stationarity proxy (the job has no
    //     access to qpActiveSetCore's internal working-set machinery to recover exact multipliers for
    //     active GENERAL rows, so this is an honest, clearly-scoped proxy, not a claim of an exact
    //     multiplier-based KKT residual -- the hand-written test suite's KKT-oracle checks, which DO
    //     have that internal access via InternalsVisibleTo, are the source of truth for correctness).
    //
    // Every solve runs inside a [BurstCompile] IJob; timing is IJob.Run() (native code, not Mono). Hand-
    // written harness half. The timed IJob and the per-section build+measure method are code-generated
    // per dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/QPBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class QPBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-qp.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Convex quadratic programming (QP.solve: dense HiGHS-style primal null-space active set) ===");
            sb.AppendLine("Section 1: random SPD QP (m = n/2 LessEqual rows, box bounds [0,3], Q well-conditioned via");
            sb.AppendLine("Rand.spdInPlace) solved through the public facade with NO caller-supplied start -- every row");
            sb.AppendLine("exercises phase 1 (LP-powered feasible start) as well as the active-set loop. KKT resid is an");
            sb.AppendLine("HONEST, freshly-recomputed proxy (exact primal feasibility + box-only projected-gradient");
            sb.AppendLine("stationarity) using only public data -- see QpSolveJobFProxy's own comment for scope.");
            sb.AppendLine("Section 2: the active-set loop ALONE (qpActiveSetCore from a supplied feasible x0, no");
            sb.AppendLine("phase-1 LP), timed with the incremental reduced space vs from-scratch every iteration --");
            sb.AppendLine("the stage-2 up/downdate A/B gate, undiluted by phase 1. iters must match between rows.");
            sb.AppendLine();

            SectionSolveFloat(sb);
            SectionSolveDouble(sb);
            SectionCoreLoopFloat(sb);
            SectionCoreLoopDouble(sb);
        }
    }
}
