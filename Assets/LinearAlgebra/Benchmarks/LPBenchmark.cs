using System.Globalization;
using System.Text;

namespace BULA.Benchmarks
{
    // Shared, dtype-agnostic config + table formatters for LPBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes and row writers.
    public static class LPBenchmarkFmt
    {
        // LP.solve sizes: n variables, m = n/2 inequality constraints (a wide, interior feasible region).
        // 384 exercises the revised/dual backends at a size where the
        // interior point and both revised-simplex engines all stay well within budget.
        public static readonly int[] SolveVarsN = { 24, 48, 96, 192, 384 };

        // LAD sizes: m observations, NCoef coefficients. LAD-via-LP has m equality constraints, so the
        // reformulated LP grows with m -- kept modest here precisely to show the scaling gap against IRLS.
        public static readonly int[] LadRowsM = { 48, 96, 192, 384 };
        public const int NCoef = 4;

        // Fast-route-only LAD sweep (Section 2b): brackets the literature's Barrodale-Roberts vs
        // Frisch-Newton crossover (Portnoy & Koenker 1997, "The Gaussian Hare and the Laplacian
        // Tortoise", crossover cited ~1e3-1e4 observations) with one point below LadRowsM's range
        // (m=8, near NCoef=4) and three points spanning past it (1024, 4096, 16384). ONLY LP.ladFN /
        // LP.ladBR / ladIRLS run at these sizes -- the LP-reformulation backends (interior/revised/
        // dual, via LadJobFProxy) build an O(m x m)-scaled structure and
        // are both far over budget at m>=1024 and uninteresting at m=8; see SectionLadFProxy's own
        // budget comment for the estimate.
        public static readonly int[] LadFastRowsM = { 8, 16, 1024, 4096, 16384 };

        // Shared by Section 4 (dense covering LP, dual-favorable) and Section 5 (infeasibility
        // detection): both stay modest relative to SolveVarsN's top end because Section 4's primal phase
        // 1 (every row starts infeasible under the all-logical basis) and Section 5's degenerate
        // contradiction can need materially more pivots than Section 1's feasibility-friendly
        // construction at the same n.
        public static readonly int[] MidVarsN = { 48, 96, 192 };

        // Sparse LAD sizes: m observations over a tall BSR design (~8 nonzeros/row), SparseLadCoef
        // coefficients. m spans past where the dense m x m interior-point normal matrix is practical --
        // the matrix-free interior point never forms it. The dense baseline runs only up to
        // SparseLadDenseCap (above it the dense normal matrix is hundreds of MB+ and the O(m^3) factor
        // dominates).
        public static readonly int[] SparseLadRowsM = { 512 };   // trimmed to a single trusted size until timings are nailed
        public const int SparseLadCoef = 32;
        public const int SparseLadDenseCap = 512;   // dense interior baseline only up to here

        public static string StatusName(LPStatus s) => s == LPStatus.Optimal ? "Optimal"
            : s == LPStatus.Infeasible ? "Infeasible" : s == LPStatus.Unbounded ? "Unbounded" : "MaxIter";

        public static string SolveHeader() => string.Format("{0,-7} {1,-6} {2,-6} {3,-14} {4,11} {5,11} {6,7} {7,14}",
            "dtype", "n", "m", "method", "med(ms)", "min(ms)", "iters", "objective");

        public static string SolveRow(string dtype, int n, int m, string method, Bench.Stat st, int iters, double obj) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-14} {4,11:F4} {5,11:F4} {6,7} {7,14:E4}",
                dtype, n, m, method, st.Median, st.Min, iters, obj);

        public static string LadHeader() => string.Format("{0,-7} {1,-6} {2,-6} {3,-16} {4,11} {5,11} {6,7} {7,14}",
            "dtype", "m", "n", "method", "med(ms)", "min(ms)", "iters", "L1 residual");

        public static string LadRow(string dtype, int m, int n, string method, Bench.Stat st, int iters, double l1) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-16} {4,11:F4} {5,11:F4} {6,7} {7,14:E4}",
                dtype, m, n, method, st.Median, st.Min, iters, l1);

        // Same column layout as SolveHeader/SolveRow but the last column is the terminal status instead
        // of the objective -- Section 5 (infeasibility detection) needs to show WHICH backends actually
        // report Infeasible vs exhaust MaxIterations, which an objective number can't communicate
        // (Infeasible/MaxIterations both leave objective meaningless -- see LPInfo's own doc comment).
        //
        // Takes `int status`, not LPStatus: the job (root cause 1's fix) already writes `(int)info.status`
        // into a NativeArray<int> output -- an enum-to-int cast is Burst-legal -- and the template passes
        // that raw int straight through. This int IS the cross-assembly bridge: the raw
        // TemplateSourceBenchmarks firstpass compile has its own LOCAL LPStatus, distinct from this
        // hand-written assembly's (same reason the OP TemplateSource firstpass needs its own proxy
        // structs -- see LP.RevisedSimplex.fProxy.cs's file header), so passing the ENUM directly across
        // that boundary is a CS0012 "add a reference to assembly BurstLinearAlgebra" error; an earlier
        // version worked around it by formatting the status name to a string inside the template, which
        // was rejected as the wrong place to do that mapping -- StatusName (this assembly's own real
        // LPStatus) belongs here instead, one cast away from the raw int.
        public static string InfeasHeader() => string.Format("{0,-7} {1,-6} {2,-6} {3,-14} {4,11} {5,11} {6,7} {7,14}",
            "dtype", "n", "m", "method", "med(ms)", "min(ms)", "iters", "status");

        public static string InfeasRow(string dtype, int n, int m, string method, Bench.Stat st, int iters, int status) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-14} {4,11:F4} {5,11:F4} {6,7} {7,14}",
                dtype, n, m, method, st.Median, st.Min, iters, StatusName((LPStatus)status));

        // Section 6 (warm re-solve chain): iters column = TOTAL dual-simplex pivots across the K
        // warm re-solves (the seeding cold solve excluded), objective = last re-solve's (three-mode
        // agreement check on the identical perturbation sequence).
        public static readonly int[] WarmVarsN = { 96, 192, 384 };

        public static string WarmHeader() => string.Format("{0,-7} {1,-6} {2,-6} {3,-14} {4,11} {5,11} {6,9} {7,14}",
            "dtype", "n", "m", "mode", "med(ms)", "min(ms)", "warmIters", "objective");

        public static string WarmRow(string dtype, int n, int m, string mode, Bench.Stat st, int iters, double obj) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-6} {3,-14} {4,11:F4} {5,11:F4} {6,9} {7,14:E4}",
                dtype, n, m, mode, st.Median, st.Min, iters, obj);
    }

    // ================================================================================================
    // Linear programming + least-absolute-deviation benchmark.
    //
    //   Section 1 (LP.solve): random dense feasible+bounded LPs (A >= 0, b = A x0 + slack so x0 is
    //     feasible). THREE backends compared head to head -- Mehrotra interior
    //     point, the bounded-variable revised primal simplex, and the dual
    //     revised simplex -- on the IDENTICAL problem; the objective column shows all three agree, the
    //     iters column is directly comparable pivot-for-pivot.
    //
    //   Section 2 (LAD): random overdetermined regression b = A x_true + noise with periodic gross
    //     outliers. Exact L1 fit via all THREE LP.lad/LP.solve backends, PLUS the two reformulation-free
    //     exact engines -- LP.ladFN (matrix-free Frisch-Newton dual interior point over the raw m x n
    //     design -- no LP reformulation, no m x m matrix, an n x n
    //     weighted normal solve per Newton step) and LP.ladBR (Barrodale-Roberts specialized simplex
    //     over the same raw design -- an exact VERTEX solution,
    //     ~O(n) iterations regardless of m via its weighted-median long step) -- vs the fast approximate
    //     IRLS (Optimize.ladIRLS). LP.lad's own default backend is RevisedSimplex, since LAD's standard
    //     form is exactly the bounded-variable shape revised simplex targets. The
    //     L1-residual column shows all six reach essentially the same minimum; the timing shows
    //     LAD-via-LP grows with the number of observations (its constraint count) while
    //     ladFN/ladBR/IRLS stay practical much further -- ladFN's and ladBR's iters columns are directly comparable to IRLS's (all
    //     three priced per iteration; ladFN/ladBR are exact, IRLS is approximate).
    //
    //   Section 2b (LAD, fast routes only): the same construction extended to a wider m range
    //     (LadFastRowsM: 8, 16, 1024, 4096, 16384) with ONLY ladFN/ladBR/IRLS run -- the LP-
    //     reformulation backends are far over budget at m>=1024. Exists to bracket the Barrodale-
    //     Roberts vs Frisch-Newton crossover the literature (Portnoy & Koenker 1997) predicts around
    //     m in [1e3,1e4].
    //
    //   Section 3 (sparse LAD): the same L1 fit over a tall block-sparse (BSR) design, solved by the
    //     matrix-free Frisch-Newton (streams the stored blocks; its normal matrix is n x n in the
    //     coefficient count, never m x m), vs the dense LP.lad baseline where it still fits.
    //
    //   Section 4 (dense covering LP): A,b,c >= 0 by construction --
    //     deliberately DUAL-FAVORABLE: every nonneg cost column is already dual-feasible at the
    //     all-logical start (y=0 -> d_j=c_j>=0), so dual simplex needs no phase 1 at all, while every row
    //     starts primal-INFEASIBLE (0 doesn't satisfy Ax>=b), forcing a real phase 1 on the revised
    //     primal. The fairness counterpoint to Section 1's primal-friendly construction, for the
    //     primal-vs-dual-default question.
    //
    //   Section 5 (infeasibility detection): Section 1's feasible construction plus ONE contradictory
    //     row (row 0 duplicated as a >= row with rhs b0+10 -- A0.x can never be both <= b0 and >= b0+10,
    //     infeasible by construction with no subtler failure mode to get wrong). Reports a STATUS column
    //     instead of an objective: the exact backends (revised/dual simplex) should all report
    //     Infeasible; interior point has no exact infeasibility certificate (that needs a homogeneous
    //     self-dual embedding -- see LP.InteriorPoint.fProxy.cs's own doc comment) and is EXPECTED to
    //     exhaust MaxIterations instead -- the table reports that honestly rather than masking it.
    //
    //   Section 6 (warm re-solve chain): 1 cold seed + K=16 rhs-perturbed re-solves on the same
    //     instance -- cold every time vs ref LPBasis vs ref
    //     LPBasis + LP cache (factor/weight persistence); identical perturbation sequence per mode,
    //     so warmIters and objective are directly comparable.
    //
    // Every solve runs inside a [BurstCompile] IJob; timing is IJob.Run() (native code, not Mono).
    // Hand-written harness half. The timed IJobs and the per-section build+measure methods are code-
    // generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LPBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class LPBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-lp.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Linear programming (interior point vs revised/dual simplex) + least absolute deviation ===");
            sb.AppendLine("Section 1: random dense feasible/bounded LPs, min cx s.t. Ax<=b, x>=0 -- interior point vs");
            sb.AppendLine("revised primal simplex vs dual simplex, all on the SAME problem (objective");
            sb.AppendLine("column shows all three agree). Section 2: LAD");
            sb.AppendLine("(L1) regression with gross outliers -- exact LP.lad (all three LP.solve backends, LAD's own");
            sb.AppendLine("default is RevisedSimplex) plus the two reformulation-free exact engines, LP.ladFN (matrix-free");
            sb.AppendLine("Frisch-Newton dual interior point, no LP reformulation) and LP.ladBR (Barrodale-Roberts");
            sb.AppendLine("specialized simplex, exact vertex solution, ~O(n) iterations via its weighted-median long");
            sb.AppendLine("step), vs fast approximate Optimize.ladIRLS. L1-residual column shows");
            sb.AppendLine("agreement; timing shows LAD-via-LP grows with observation count while ladFN/ladBR/IRLS");
            sb.AppendLine("stay practical much further, ladFN and");
            sb.AppendLine("IRLS both staying a fixed n x n solve per iteration throughout, ladBR's iters staying ~O(n).");
            sb.AppendLine("Section 3: SPARSE LAD -- the same L1 fit over a tall block-sparse");
            sb.AppendLine("(BSR) design, solved by the matrix-free interior point (never forms the m x m normal");
            sb.AppendLine("matrix), vs the dense LP.lad baseline where it still fits. Same core drives sparse");
            sb.AppendLine("LP.solve, so it is representative. Section 4: the DENSE covering LP (min cx s.t.");
            sb.AppendLine("Ax>=b, x>=0), all three LP.solve backends -- deliberately dual-favorable (every row starts");
            sb.AppendLine("primal-infeasible, forcing a real phase 1 on the primal backends, while every column");
            sb.AppendLine("starts dual-feasible) -- the fairness counterpoint to Section 1 for the primal-vs-dual");
            sb.AppendLine("default question. Section 5: infeasibility detection -- Section 1's construction plus one");
            sb.AppendLine("contradictory row, all three backends; a STATUS column (not objective) shows which backends");
            sb.AppendLine("actually detect Infeasible vs exhaust MaxIterations (interior point has no exact");
            sb.AppendLine("infeasibility certificate, so MaxIterations there is expected, not a bug).");
            sb.AppendLine("Section 6: warm re-solve chain -- 1 cold seed +");
            sb.AppendLine("K=16 rhs-perturbed re-solves on the same instance: cold every time vs ref LPBasis vs");
            sb.AppendLine("ref LPBasis + fProxyLPCache (factor/weight persistence); identical perturbation");
            sb.AppendLine("sequence per mode, so warmIters and objective are directly comparable.");
            sb.AppendLine();

            SectionSolveFloat(sb);
            SectionSolveDouble(sb);
            SectionLadFloat(sb);
            SectionLadDouble(sb);
            SectionSparseLadFloat(sb);
            SectionSparseLadDouble(sb);
            SectionDenseCoveringFloat(sb);
            SectionDenseCoveringDouble(sb);
            SectionInfeasibleFloat(sb);
            SectionInfeasibleDouble(sb);
            SectionWarmResolveFloat(sb);
            SectionWarmResolveDouble(sb);
        }
    }
}
