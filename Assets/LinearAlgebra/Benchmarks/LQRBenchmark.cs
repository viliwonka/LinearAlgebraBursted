using System.Globalization;
using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // Shared, dtype-agnostic config + table formatter for LQRBenchmark. Public so the code-generated
    // per-dtype build methods (in a separate template assembly) can reach the sizes/seeds and row writer.
    public static class LQRBenchmarkFmt
    {
        // n in {4, 12}, m in {2, 4} per docs/spec-lqr.md's benchmark section -- all 4 combinations.
        public static readonly (int n, int m)[] Sizes = { (4, 2), (4, 4), (12, 2), (12, 4) };
        public static readonly uint[] Seeds = { 11u, 22u, 33u, 44u };

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

        public static string Header() => string.Format("{0,-7} {1,-16} {2,4} {3,4} {4,11} {5,11} {6,6} {7,10}",
            "dtype", "variant", "n", "m", "med(ms)", "min(ms)", "iters", "status");

        public static string Row(string dtype, string variant, int n, int m, Bench.Stat st, int iters, int status) =>
            string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-16} {2,4} {3,4} {4,11:F4} {5,11:F4} {6,6} {7,10}",
                dtype, variant, n, m, st.Median, st.Min, iters, StatusName((LQRStatus)status));
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
                var (n, m) = LQRBenchmarkFmt.Sizes[i];
                uint seed = LQRBenchmarkFmt.Seeds[i];
                sb.AppendLine(ColdSdaFloat(n, m, seed));
                sb.AppendLine(ColdRecursionFloat(n, m, seed));
                sb.AppendLine(WarmFloat(n, m, seed));
            }
            for (int i = 0; i < LQRBenchmarkFmt.Sizes.Length; i++)
            {
                var (n, m) = LQRBenchmarkFmt.Sizes[i];
                uint seed = LQRBenchmarkFmt.Seeds[i];
                sb.AppendLine(ColdSdaDouble(n, m, seed));
                sb.AppendLine(ColdRecursionDouble(n, m, seed));
                sb.AppendLine(WarmDouble(n, m, seed));
            }
        }
    }
}
