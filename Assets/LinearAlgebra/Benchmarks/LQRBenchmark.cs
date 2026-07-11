using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatter for LQRBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes/seeds and row writer.
    public static class LQRBenchmarkFmt
    {
        // n in {4, 12} x m in {2, 4} per docs/spec-lqr.md, REPEATED 100x inside the timed job (one
        // solve is ~2-50us -- single-invocation samples sit in timer-noise territory; med(ms)/min(ms)
        // columns report PER-SOLVE time = job time / reps). Plus n in {64, 128} single-shot rows to
        // locate the SDA-vs-recursion wall-clock crossover the spec's small-n finding left open.
        // The two `marginal` rows use a near-unit-circle spectrum (diag [0.90,0.98)) -- the regime
        // where the plain recursion's linear convergence collapses and SDA's quadratic convergence
        // is supposed to pay; the well-damped rows measured them comparable, so this is SDA's
        // justification cell.
        public static readonly (int n, int m, int reps, bool marginal)[] Sizes =
            { (4, 2, 100, false), (4, 4, 100, false), (12, 2, 100, false), (12, 4, 100, false),
              (64, 4, 1, false), (128, 4, 1, false), (12, 4, 100, true), (64, 4, 1, true) };
        public static readonly uint[] Seeds = { 11u, 22u, 33u, 44u, 55u, 66u, 77u, 88u };

        // `status` crosses the hand-written/template assembly boundary as a raw int -- same CS0012
        // reason as LPBenchmarkFmt.InfeasRow's own doc comment (LQRStatus is defined in Control.Info.cs,
        // a "singularFile" that's ALSO part of the TemplateSource firstpass compile, so the generated
        // TemplateSourceBenchmarks firstpass compile has its own LOCAL LQRStatus, distinct from this
        // hand-written assembly's).
        public static string StatusName(LQRStatus s) => s switch
        {
            LQRStatus.Converged => "Converged",
            LQRStatus.MaxIterations => "MaxIter",
            LQRStatus.Diverged => "Diverged",
            _ => "Unknown",
        };

        public static string Header() => string.Format("{0,-7} {1,-16} {2,4} {3,4} {4,5} {5,11} {6,11} {7,6} {8,10}",
            "dtype", "variant", "n", "m", "reps", "med(ms)", "min(ms)", "iters", "status");

        // med/min are PER-SOLVE (job time / reps).
        public static string Row(string dtype, string variant, int n, int m, int reps, Bench.Stat st, int iters, int status) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-16} {2,4} {3,4} {4,5} {5,11:F4} {6,11:F4} {7,6} {8,10}",
                dtype, variant, n, m, reps, st.Median / reps, st.Min / reps, iters, StatusName((LQRStatus)status));
    }

    // ================================================================================================
    // Discrete-time LQR (Control.lqr / Control.lqrSchedule -- docs/spec-lqr.md). Three variants at each
    // (n, m):
    //   - cold-SDA: structure-preserving doubling from scratch (the plain Control.lqr overload).
    //   - cold-recursion: the plain fixed-point Riccati recursion, ALSO cold-started (S seeded at zero)
    //     -- the naive baseline SDA/warm are compared against, reached via the warm overload with a
    //     force-populated fresh state (see LQRBenchmark.fProxy.cs's header for why).
    //   - warm: the same plain recursion, but re-solved from the PRIOR converged S after a ~1e-3
    //     relative perturbation of A -- the per-frame re-linearization use case.
    //
    // n in {4, 12}, m in {2, 4}, both dtypes. Expect everything in microseconds (that is the point of a
    // gamedev-scale n<=12/m<=4 problem) with cold-recursion the clear loser: warm and cold-SDA should
    // both be much cheaper (SDA via quadratic doubling convergence in ~10-25 steps regardless of start;
    // warm because a 1e-3 perturbation only needs a handful of plain-recursion steps from an
    // already-near-fixed-point S).
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per dtype
    // from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/LQRBenchmark.fProxy.cs.
    // ================================================================================================
    public static partial class LQRBenchmark
    {
        public static void Run() => Bench.WriteReport("benchmark-lqr.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Discrete-time LQR (Control.lqr): cold SDA vs cold plain-recursion vs warm ===");
            sb.AppendLine("cold-recursion is the naive baseline (plain fixed-point Riccati iteration, S seeded");
            sb.AppendLine("at zero); cold-SDA (structure-preserving doubling) and warm (1e-3-relative A");
            sb.AppendLine("perturbation, re-solved from the prior converged S) should both be much cheaper.");
            sb.AppendLine("Everything here is expected in microseconds -- that IS the point at this");
            sb.AppendLine("gamedev-scale n<=12/m<=4 problem size.");
            sb.AppendLine(LQRBenchmarkFmt.Header());

            for (int i = 0; i < LQRBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, reps, marginal) = LQRBenchmarkFmt.Sizes[i];
                uint seed = LQRBenchmarkFmt.Seeds[i];
                sb.AppendLine(ColdSdaFloat(n, m, reps, seed, marginal));
                sb.AppendLine(ColdRecursionFloat(n, m, reps, seed, marginal));
                sb.AppendLine(WarmFloat(n, m, reps, seed, marginal));
            }
            for (int i = 0; i < LQRBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m, reps, marginal) = LQRBenchmarkFmt.Sizes[i];
                uint seed = LQRBenchmarkFmt.Seeds[i];
                sb.AppendLine(ColdSdaDouble(n, m, reps, seed, marginal));
                sb.AppendLine(ColdRecursionDouble(n, m, reps, seed, marginal));
                sb.AppendLine(WarmDouble(n, m, reps, seed, marginal));
            }
        }
    }
}
