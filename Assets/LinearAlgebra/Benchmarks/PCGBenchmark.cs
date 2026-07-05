using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
    // Block-Jacobi Preconditioned Conjugate Gradient (Solvers.pcg) over a representative BSR system —
    // the one square iterative solver SparseSolverBenchmark.cs doesn't already cover (that file
    // benchmarks plain cg/minres/biCGStab/cgls/lsqr but predates pcg). The system is a block-tridiagonal
    // SPD matrix (block size BR, a common 1D FEM/heat-equation stencil): diagonally-dominant diagonal
    // blocks + small symmetric off-diagonal coupling to the immediate neighbor block only, so it is
    // genuinely sparse (nnzb = 3*nb-2) without needing SparseSolverBenchmark's randomized block-pattern
    // machinery. maxIterations is FIXED with tolerance=0 (mirrors SparseSolverBenchmark's convention),
    // so every timed sample runs exactly K iterations — deterministic timing; the residual column shows
    // convergence, not just speed. Plain cg (unpreconditioned) on the SAME system is included alongside
    // for a direct preconditioning-overhead-vs-iteration-savings comparison.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgBsrJobFloat : IJob
    {
        public floatBSR A;
        public floatBlockJacobi M;
        public floatN b, x, r, p, Ap, z;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Solvers.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgBsrJobDouble : IJob
    {
        public doubleBSR A;
        public doubleBlockJacobi M;
        public doubleN b, x, r, p, Ap, z;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0.0;
            Solvers.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0.0);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CgBsrJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Solvers.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CgBsrJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0.0;
            Solvers.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0.0);
        }
    }

    public static class PCGBenchmark
    {
        const int BR = 3;         // block size (matches SparseSolverBenchmark's FEM/cloth/PD workhorse)
        const int NB = 256;       // number of blocks -> N = 768
        const int K = 40;         // fixed iteration budget, tol=0 (deterministic timing)

        public static void Run() => Bench.WriteReport("benchmark-pcg.txt", Section);

        static void BuildTridiagBlockSPDFloat(ref Arena arena, out floatBSR sparse, out int n)
        {
            n = NB * BR;
            int nnzb = NB + 2 * (NB - 1);
            var builder = arena.floatBSRBuilder(NB, NB, BR, BR, nnzb);
            var rng = new Random(0x51ED270Bu);

            for (int i = 0; i < NB; i++)
            {
                var Di = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Di[r, c] = (r == c ? BR * 8f : 0f) + rng.NextFloat(-0.1f, 0.1f);
                builder.AddBlock(i, i, in Di);

                if (i > 0)
                {
                    var off = arena.floatMat(BR, BR);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            off[r, c] = rng.NextFloat(-0.3f, 0.3f);
                    builder.AddBlock(i, i - 1, in off);

                    var offT = arena.floatMat(BR, BR);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            offT[r, c] = off[c, r];
                    builder.AddBlock(i - 1, i, in offT);
                }
            }

            sparse = builder.ToBSR(ref arena);
        }

        static void BuildTridiagBlockSPDDouble(ref Arena arena, out doubleBSR sparse, out int n)
        {
            n = NB * BR;
            int nnzb = NB + 2 * (NB - 1);
            var builder = arena.doubleBSRBuilder(NB, NB, BR, BR, nnzb);
            var rng = new Random(0x51ED270Bu);

            for (int i = 0; i < NB; i++)
            {
                var Di = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Di[r, c] = (r == c ? BR * 8.0 : 0.0) + rng.NextFloat(-0.1f, 0.1f);
                builder.AddBlock(i, i, in Di);

                if (i > 0)
                {
                    var off = arena.doubleMat(BR, BR);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            off[r, c] = rng.NextFloat(-0.3f, 0.3f);
                    builder.AddBlock(i, i - 1, in off);

                    var offT = arena.doubleMat(BR, BR);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            offT[r, c] = off[c, r];
                    builder.AddBlock(i - 1, i, in offT);
                }
            }

            sparse = builder.ToBSR(ref arena);
        }

        static double Residual(in floatBSR A, in floatN x, in floatN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double diff = (double)Ax[i] - (double)b[i];
                num += diff * diff;
                den += (double)b[i] * (double)b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        static double Residual(in doubleBSR A, in doubleN x, in doubleN b)
        {
            var Ax = BSR.spMV(in A, in x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double diff = Ax[i] - b[i];
                num += diff * diff;
                den += b[i] * b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine(string.Format("=== Block-Jacobi PCG vs plain CG, block-tridiagonal SPD BSR (b={0}, nb={1}, K={2}, tol=0) ===", BR, NB, K));
            sb.AppendLine(string.Format("{0,-7} {1,-6} {2,-12} {3,11} {4,11} {5,14}",
                "dtype", "N", "solver", "med(ms)", "min(ms)", "residual"));
            sb.AppendLine(BenchFloat());
            sb.AppendLine(BenchDouble());
            sb.AppendLine();
        }

        static string Row(string dtype, int n, string solver, Bench.Stat st, double residual) =>
            string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,-12} {3,11:F4} {4,11:F4} {5,14:E3}",
                dtype, n, solver, st.Median, st.Min, residual);

        static string BenchFloat()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildTridiagBlockSPDFloat(ref arena, out var A, out int n);
            var M = arena.floatBlockJacobi(in A);
            var b = arena.floatRandomVec(n, -1f, 1f, 0xC001Du);

            var xCg = arena.floatVec(n); var rCg = arena.floatVec(n); var pCg = arena.floatVec(n); var ApCg = arena.floatVec(n);
            var cgJob = new CgBsrJobFloat { A = A, b = b, x = xCg, r = rCg, p = pCg, Ap = ApCg, K = K };
            var cgStat = Bench.Time(() => cgJob.Run());
            var sb = new StringBuilder();
            sb.AppendLine(Row("float", n, "CG", cgStat, Residual(in A, in xCg, in b)));

            var xPcg = arena.floatVec(n); var rPcg = arena.floatVec(n); var pPcg = arena.floatVec(n); var ApPcg = arena.floatVec(n); var zPcg = arena.floatVec(n);
            var pcgJob = new PcgBsrJobFloat { A = A, M = M, b = b, x = xPcg, r = rPcg, p = pPcg, Ap = ApPcg, z = zPcg, K = K };
            var pcgStat = Bench.Time(() => pcgJob.Run());
            sb.Append(Row("float", n, "PCG-Jacobi", pcgStat, Residual(in A, in xPcg, in b)));

            arena.Dispose();
            return sb.ToString();
        }

        static string BenchDouble()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildTridiagBlockSPDDouble(ref arena, out var A, out int n);
            var M = arena.doubleBlockJacobi(in A);
            var b = arena.doubleRandomVec(n, -1.0, 1.0, 0xC001Du);

            var xCg = arena.doubleVec(n); var rCg = arena.doubleVec(n); var pCg = arena.doubleVec(n); var ApCg = arena.doubleVec(n);
            var cgJob = new CgBsrJobDouble { A = A, b = b, x = xCg, r = rCg, p = pCg, Ap = ApCg, K = K };
            var cgStat = Bench.Time(() => cgJob.Run());
            var sb = new StringBuilder();
            sb.AppendLine(Row("double", n, "CG", cgStat, Residual(in A, in xCg, in b)));

            var xPcg = arena.doubleVec(n); var rPcg = arena.doubleVec(n); var pPcg = arena.doubleVec(n); var ApPcg = arena.doubleVec(n); var zPcg = arena.doubleVec(n);
            var pcgJob = new PcgBsrJobDouble { A = A, M = M, b = b, x = xPcg, r = rPcg, p = pPcg, Ap = ApPcg, z = zPcg, K = K };
            var pcgStat = Bench.Time(() => pcgJob.Run());
            sb.Append(Row("double", n, "PCG-Jacobi", pcgStat, Residual(in A, in xPcg, in b)));

            arena.Dispose();
            return sb.ToString();
        }
    }
}
