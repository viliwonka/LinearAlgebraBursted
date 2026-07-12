using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatters for MPCBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes/seeds and row writers.
    public static class MPCBenchmarkFmt
    {
        // (N, n, m, coldReps): (10,4,1)/(20,6,2)/(40,12,4) are the original gamedev-scale rows
        // (unchanged, coldReps=20 as before). (20,16,4) and (30,24,8) are quadrotor-MPC scale (pose+vel+
        // attitude+rates+rotor states land n around 24; IMU+GPS+visual-pose+rangefinder-fused inputs
        // land m around 8) -- (30,24,8) gives d=N*m=240 condensed decision variables (270 in the
        // box+soft-wall variant, +N slack columns), deliberately AT the ~250-300-variable
        // condensed-formulation boundary the design research flagged for revisit; this row is the
        // empirical probe of that boundary. coldReps is reduced (5) for this row only -- Section 2's
        // cold, no-warm-start solve is the slowest per-call operation here -- every other row's coldReps
        // is unchanged.
        public static readonly (int N, int n, int m, int coldReps)[] Sizes =
        {
            (10, 4, 1, 20),
            (20, 6, 2, 20),
            (40, 12, 4, 20),
            (20, 16, 4, 20),
            (30, 24, 8, 5),
        };
        public static readonly uint[] Seeds = { 111u, 222u, 333u, 444u, 555u };

        // Frames burned off UNTIMED before every Section-1 measurement -- past the cold-start/active-set
        // churn transient MPCTests.fProxy.cs's own WarmStartChurnBound test shows collapsing within a
        // handful of frames (that test's own steadyFrame constant is 8; 5 is a conservative margin given
        // this benchmark's smaller, better-conditioned random plants).
        public const int WarmPrewarmFrames = 5;
        public const int ConstructReps = 5;

        public static string StatusName(MPCStatus s) => s switch
        {
            MPCStatus.Optimal => "Optimal",
            MPCStatus.MaxIterations => "MaxIter",
            MPCStatus.Fallback => "Fallback",
            _ => "Unknown",
        };

        public static string Header() => string.Format("{0,-7} {1,-10} {2,4} {3,4} {4,4} {5,5} {6,11} {7,11} {8,7} {9,10} {10,10}",
            "dtype", "variant", "N", "n", "m", "reps", "med(ms)", "min(ms)", "iters", "asChanges", "status");

        // med/min are PER-SOLVE (job time / reps) -- reps=1 for Section 1 (Bench.Time's own median across
        // already-past-the-churn frames IS the per-frame time). iters/asChanges/status are the honest
        // MPCInfo fields -- a budget-limited solve is visible here, not masked.
        public static string Row(string dtype, string variant, int N, int n, int m, int reps, Bench.Stat st,
                                 int iters, int changes, int status) =>
            string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-10} {2,4} {3,4} {4,4} {5,5} {6,11:F4} {7,11:F4} {8,7} {9,10} {10,10}",
                dtype, variant, N, n, m, reps, st.Median / reps, st.Min / reps, iters, changes, StatusName((MPCStatus)status));

        public static string ConstructionHeader() => string.Format("{0,-7} {1,4} {2,4} {3,4} {4,5} {5,11} {6,11}",
            "dtype", "N", "n", "m", "reps", "med(ms)", "min(ms)");

        public static string ConstructionRow(string dtype, int N, int n, int m, int reps, Bench.Stat st) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,4} {2,4} {3,4} {4,5} {5,11:F4} {6,11:F4}",
                dtype, N, n, m, reps, st.Median / reps, st.Min / reps);
    }

    // ================================================================================================
    // Linear MPC (MPC.solve / fProxyMPCState). Sections:
    //   1. Warm steady-state per-frame cost (the headline): a receding-horizon loop with the first
    //      WarmPrewarmFrames frames burned off UNTIMED (cold-start + active-set-churn transient), then
    //      Bench.Time's own 1 warmup + 4 timed calls -- each ONE receding-horizon frame -- report the
    //      per-frame median. Box-only and box+soft-wall variants at each size. Sizes span gamedev-scale
    //      (10,4,1)/(20,6,2)/(40,12,4) up to quadrotor-MPC scale (20,16,4)/(30,24,8) -- the last row's
    //      d=N*m=240 condensed decision variables (270 with the soft wall's slack columns) sits AT the
    //      ~250-300-variable condensed-formulation boundary the design research flagged for revisit; it
    //      is the empirical probe of that boundary.
    //   2. Cold solve cost (fresh warm-start carry every call -- z/uPlan zeroed, wstatus all Inactive,
    //      populated=false, so every rep pays the first-ever-solve cost) at the same sizes, for contrast
    //      against Section 1's warm numbers (coldReps reduced for the largest row only).
    //   3. fProxyMPCState construction cost (one-shot: the terminal-DARE solve + Phi/Gamma/H condensing).
    //
    // iters/activeSetChanges/status are always the honest MPCInfo fields -- a solve that returns early on
    // the iteration budget must be visible, not masquerade as fast.
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per dtype
    // from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/MPCBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class MPCBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-mpc.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Linear MPC (MPC.solve / fProxyMPCState) ===");
            sb.AppendLine("Section 1: warm steady-state per-frame cost (the headline) -- a receding-horizon loop,");
            sb.AppendLine("first 5 frames burned off untimed (cold-start + active-set churn), then the per-frame");
            sb.AppendLine("median reported; box-only and box+soft-wall variants. Sizes span gamedev-scale");
            sb.AppendLine("(10,4,1)/(20,6,2)/(40,12,4) up to quadrotor-MPC scale (20,16,4)/(30,24,8) -- the last");
            sb.AppendLine("row's d=N*m=240 condensed decision variables (270 with the soft wall's slack columns)");
            sb.AppendLine("deliberately sits AT the ~250-300-variable condensed-formulation boundary the design");
            sb.AppendLine("research flagged for revisit; this row is the empirical probe of that boundary.");
            sb.AppendLine("Section 2: cold solve cost (fresh warm-start carry every call) at the same sizes, for");
            sb.AppendLine("contrast (coldReps reduced for the largest row only). Section 3: fProxyMPCState");
            sb.AppendLine("construction cost (terminal DARE + condensing), one-shot. iters/asChanges/status are");
            sb.AppendLine("always the honest MPCInfo fields -- a budget-limited solve is visible, not masked.");
            sb.AppendLine();

            sb.AppendLine("--- 1. Warm steady-state per-frame cost ---");
            sb.AppendLine(MPCBenchmarkFmt.Header());
            for (int i = 0; i < MPCBenchmarkFmt.Sizes.Length; i++)
            {
                var (N, n, m, _) = MPCBenchmarkFmt.Sizes[i];
                sb.AppendLine(WarmFrameFloat(N, n, m, MPCBenchmarkFmt.Seeds[i], false));
                sb.AppendLine(WarmFrameFloat(N, n, m, MPCBenchmarkFmt.Seeds[i], true));
            }
            for (int i = 0; i < MPCBenchmarkFmt.Sizes.Length; i++)
            {
                var (N, n, m, _) = MPCBenchmarkFmt.Sizes[i];
                sb.AppendLine(WarmFrameDouble(N, n, m, MPCBenchmarkFmt.Seeds[i], false));
                sb.AppendLine(WarmFrameDouble(N, n, m, MPCBenchmarkFmt.Seeds[i], true));
            }

            sb.AppendLine();
            sb.AppendLine("--- 2. Cold solve cost (fresh warm-start carry every call) ---");
            sb.AppendLine(MPCBenchmarkFmt.Header());
            for (int i = 0; i < MPCBenchmarkFmt.Sizes.Length; i++)
            {
                var (N, n, m, coldReps) = MPCBenchmarkFmt.Sizes[i];
                sb.AppendLine(ColdSolveFloat(N, n, m, MPCBenchmarkFmt.Seeds[i], coldReps, false));
                sb.AppendLine(ColdSolveFloat(N, n, m, MPCBenchmarkFmt.Seeds[i], coldReps, true));
            }
            for (int i = 0; i < MPCBenchmarkFmt.Sizes.Length; i++)
            {
                var (N, n, m, coldReps) = MPCBenchmarkFmt.Sizes[i];
                sb.AppendLine(ColdSolveDouble(N, n, m, MPCBenchmarkFmt.Seeds[i], coldReps, false));
                sb.AppendLine(ColdSolveDouble(N, n, m, MPCBenchmarkFmt.Seeds[i], coldReps, true));
            }

            sb.AppendLine();
            sb.AppendLine("--- 3. fProxyMPCState construction cost (one-shot) ---");
            sb.AppendLine(MPCBenchmarkFmt.ConstructionHeader());
            for (int i = 0; i < MPCBenchmarkFmt.Sizes.Length; i++)
            {
                var (N, n, m, _) = MPCBenchmarkFmt.Sizes[i];
                sb.AppendLine(ConstructFloat(N, n, m, MPCBenchmarkFmt.Seeds[i], MPCBenchmarkFmt.ConstructReps));
            }
            for (int i = 0; i < MPCBenchmarkFmt.Sizes.Length; i++)
            {
                var (N, n, m, _) = MPCBenchmarkFmt.Sizes[i];
                sb.AppendLine(ConstructDouble(N, n, m, MPCBenchmarkFmt.Seeds[i], MPCBenchmarkFmt.ConstructReps));
            }
        }
    }
}
