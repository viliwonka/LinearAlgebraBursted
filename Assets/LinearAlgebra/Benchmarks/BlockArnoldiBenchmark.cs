using System.Text;

namespace BULA.Benchmarks
{
    // Block-Arnoldi (bgmres / bgcrodr) timing over dense nonsymmetric diagonally-dominant systems
    // (A = randMat(-1,1) + 2n*I), at three (n, s) points chosen to span the block-Arnoldi step's cost
    // regimes: LQ/MGS-heavy (restart*s comparable to or larger than n) down to matvec-dominated
    // (restart*s << n). Each point runs a "distinct" full-row-rank RHS block (no deflation) and a
    // "dup" block (row s-1 forced equal to row 0, forcing deflation every Arnoldi step) -- task #70's
    // benchmark for replacing the block-Arnoldi step's LQRP.decomp + LQRPRankFloored with a lean
    // allocation-free rank-revealing row-orthonormalizer (Krylov.RowOrthoRankFloored).
    //
    // Hand-written harness half. The timed IJobs and build+measure method (Bench{Float,Double}) are
    // code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/BlockArnoldiBenchmark.fProxy.cs.
    public static partial class BlockArnoldiBenchmark
    {
        // (n, s, restart, maxIter, recycle) -- restart*s/n roughly 5, 1.25, 0.3 across the three points.
        static readonly (int n, int s, int restart, int maxIter, int recycle)[] Points =
        {
            (128, 16, 20, 400, 10),
            (256, 16, 20, 400, 10),
            (512, 8,  20, 400, 10),
        };

        public static void Run() => Bench.WriteReport("benchmark-block-arnoldi.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Block-Arnoldi (bgmres/bgcrodr), dense nonsymmetric A = randMat + 2n*I ===");
            sb.AppendLine(string.Format("{0,-7}{1,-6}{2,-4}{3,-10}{4,-9}{5,10}{6,12}{7,8}{8,8}{9,14}",
                "dtype", "N", "s", "RHS", "solver", "med(ms)", "min(ms)", "iters", "minAct", "status"));
            foreach (var p in Points)
                sb.Append(BenchFloat(p.n, p.s, p.restart, p.maxIter, p.recycle));
            foreach (var p in Points)
                sb.Append(BenchDouble(p.n, p.s, p.restart, p.maxIter, p.recycle));
            sb.AppendLine();
        }
    }
}
