using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatters for LPBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes and row writers.
    public static class LPBenchmarkFmt
    {
        // LP.solve sizes: n variables, m = n/2 inequality constraints (a wide, interior feasible region).
        // 384 added alongside the revised/dual backends (docs/spec-revised-simplex.md): tableau simplex
        // is still practical there (~120ms/solve, double) so it stays in the size list rather than
        // getting its own cap, unlike the LAD/infeasibility sections below.
        public static readonly int[] SolveVarsN = { 24, 48, 96, 192, 384 };

        // LAD sizes: m observations, NCoef coefficients. LAD-via-LP has m equality constraints, so its
        // tableau grows with m -- kept modest here precisely to show the scaling gap against IRLS. 384
        // added to show the revised/dual backends' win over the tableau at a size where the tableau
        // itself is no longer practical -- LadSimplexCap stops the tableau-simplex row there (measured
        // ~101ms already at m=192, double; the O(m*nCols) per-pivot tableau update would make m=384 the
        // slow tail of this whole benchmark for no informative reason, exactly like SparseLadDenseCap
        // stops the dense interior-point baseline in Section 3 below).
        public static readonly int[] LadRowsM = { 48, 96, 192, 384 };
        public const int NCoef = 4;
        public const int LadSimplexCap = 192;   // tableau-simplex LAD row only up to here

        // Fast-route-only LAD sweep (Section 2b): brackets the literature's Barrodale-Roberts vs
        // Frisch-Newton crossover (Portnoy & Koenker 1997, "The Gaussian Hare and the Laplacian
        // Tortoise", crossover cited ~1e3-1e4 observations) with one point below LadRowsM's range
        // (m=8, near NCoef=4) and three points spanning past it (1024, 4096, 16384). ONLY LP.ladFN /
        // LP.ladBR / ladIRLS run at these sizes -- the LP-reformulation backends (simplex/interior/
        // revised/dual, via LadJobFProxy) build an O(m) tableau or an O(m x m)-scaled structure and
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
    }

    // ================================================================================================
    // Linear programming + least-absolute-deviation benchmark.
    //
    //   Section 1 (LP.solve): random dense feasible+bounded LPs (min cᵀx s.t. A x <= b, x >= 0, A >= 0,
    //     b = A x0 + slack so x0 is feasible). FOUR backends compared head to head -- two-phase tableau
    //     simplex, Mehrotra interior point, and (docs/spec-revised-simplex.md) the bounded-variable
    //     revised primal simplex and dual revised simplex -- on the IDENTICAL problem; the objective
    //     column shows all four agree, the iters column is directly comparable pivot-for-pivot.
    //
    //   Section 2 (LAD): random overdetermined regression b = A x_true + noise with periodic gross
    //     outliers. Exact L1 fit via all FOUR LP.lad/LP.solve backends, PLUS the two reformulation-free
    //     exact engines -- LP.ladFN (matrix-free Frisch-Newton dual interior point over the raw m x n
    //     design, docs/spec-lad-frisch-newton.md -- no LP reformulation, no m x m matrix, an n x n
    //     weighted normal solve per Newton step) and LP.ladBR (Barrodale-Roberts specialized simplex
    //     over the same raw design, docs/spec-lad-barrodale-roberts.md -- an exact VERTEX solution,
    //     ~O(n) iterations regardless of m via its weighted-median long step) -- vs the fast approximate
    //     IRLS (Optimize.ladIRLS). LP.lad's own default backend is RevisedSimplex (not the tableau),
    //     since LAD's standard form is exactly the bounded-variable shape revised simplex targets. The
    //     L1-residual column shows all seven reach essentially the same minimum; the timing shows
    //     LAD-via-tableau-simplex grows with the number of observations (its constraint count, capped
    //     at LadSimplexCap for exactly that reason) while revised/dual/ladFN/ladBR/IRLS stay practical
    //     much further -- ladFN's and ladBR's iters columns are directly comparable to IRLS's (all
    //     three priced per iteration; ladFN/ladBR are exact, IRLS is approximate).
    //
    //   Section 2b (LAD, fast routes only): the same construction extended to a wider m range
    //     (LadFastRowsM: 8, 16, 1024, 4096, 16384) with ONLY ladFN/ladBR/IRLS run -- the LP-
    //     reformulation backends are far over budget at m>=1024. Exists to bracket the Barrodale-
    //     Roberts vs Frisch-Newton crossover the literature (Portnoy & Koenker 1997) predicts around
    //     m in [1e3,1e4].
    //
    //   Section 4 (dense covering LP): min cᵀx s.t. A x >= b, x >= 0 with A,b,c >= 0 by construction --
    //     deliberately DUAL-FAVORABLE: every nonneg cost column is already dual-feasible at the
    //     all-logical start (y=0 -> d_j=c_j>=0), so dual simplex needs no phase 1 at all, while every row
    //     starts primal-INFEASIBLE (0 doesn't satisfy Ax>=b), forcing a real phase 1 on the tableau AND
    //     revised primal. The fairness counterpoint to Section 1's primal-friendly construction, for the
    //     primal-vs-dual-default question.
    //
    //   Section 5 (infeasibility detection): Section 1's feasible construction plus ONE contradictory
    //     row (row 0 duplicated as a >= row with rhs b0+10 -- A0.x can never be both <= b0 and >= b0+10,
    //     infeasible by construction with no subtler failure mode to get wrong). Reports a STATUS column
    //     instead of an objective: the exact backends (tableau/revised/dual simplex) should all report
    //     Infeasible; interior point has no exact infeasibility certificate (that needs a homogeneous
    //     self-dual embedding -- see LP.InteriorPoint.fProxy.cs's own doc comment) and is EXPECTED to
    //     exhaust MaxIterations instead -- the table reports that honestly rather than masking it.
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
            sb.AppendLine("=== Linear programming (simplex vs interior point vs revised/dual simplex) + least absolute deviation ===");
            sb.AppendLine("Section 1: random dense feasible/bounded LPs, min cx s.t. Ax<=b, x>=0 -- tableau simplex vs");
            sb.AppendLine("interior point vs revised primal simplex vs dual simplex, all on the SAME problem (objective");
            sb.AppendLine("column shows all four agree). Section 2: LAD");
            sb.AppendLine("(L1) regression with gross outliers -- exact LP.lad (all four LP.solve backends, LAD's own");
            sb.AppendLine("default is RevisedSimplex) plus the two reformulation-free exact engines, LP.ladFN (matrix-free");
            sb.AppendLine("Frisch-Newton dual interior point, no LP reformulation) and LP.ladBR (Barrodale-Roberts");
            sb.AppendLine("specialized simplex, exact vertex solution, ~O(n) iterations via its weighted-median long");
            sb.AppendLine("step), vs fast approximate Optimize.ladIRLS. L1-residual column shows");
            sb.AppendLine("agreement; timing shows tableau-simplex LAD grows with observation count (capped past");
            sb.AppendLine("LadSimplexCap) while revised/dual simplex/ladFN/ladBR/IRLS stay practical much further, ladFN and");
            sb.AppendLine("IRLS both staying a fixed n x n solve per iteration throughout, ladBR's iters staying ~O(n).");
            sb.AppendLine("Section 3: SPARSE LAD -- the same L1 fit over a tall block-sparse");
            sb.AppendLine("(BSR) design, solved by the matrix-free interior point (never forms the m x m normal");
            sb.AppendLine("matrix), vs the dense LP.lad baseline where it still fits. Same core drives sparse");
            sb.AppendLine("LP.solve, so it is representative. Section 4: the DENSE covering LP (min cx s.t.");
            sb.AppendLine("Ax>=b, x>=0), all four LP.solve backends -- deliberately dual-favorable (every row starts");
            sb.AppendLine("primal-infeasible, forcing a real phase 1 on the primal backends, while every column");
            sb.AppendLine("starts dual-feasible) -- the fairness counterpoint to Section 1 for the primal-vs-dual");
            sb.AppendLine("default question. Section 5: infeasibility detection -- Section 1's construction plus one");
            sb.AppendLine("contradictory row, all four backends; a STATUS column (not objective) shows which backends");
            sb.AppendLine("actually detect Infeasible vs exhaust MaxIterations (interior point has no exact");
            sb.AppendLine("infeasibility certificate, so MaxIterations there is expected, not a bug).");
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
        }
    }
}
