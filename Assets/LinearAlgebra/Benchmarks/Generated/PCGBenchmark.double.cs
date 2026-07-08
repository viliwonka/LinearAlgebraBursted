using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of PCGBenchmark (timed IJobs + system builder + residual + measure).
    // The dtype-agnostic harness (BR/NB/K constants, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/PCGBenchmark.cs.
    //
    // NOTE: the block-tridiagonal builder intentionally draws its noise with rng.NextFloat in BOTH
    // dtypes (not NextDouble) so the float and double systems are seeded from the identical stream;
    // do not "correct" those to NextDouble.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct PcgBsrJobDouble : IJob
    {
        public doubleBSR A;
        public doubleBlockJacobi M;
        public doubleN b, x, r, p, Ap, z;
        public int K;

        public void Execute()
        {
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, K, 0f);
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
            for (int i = 0; i < x.N; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    public static partial class PCGBenchmark
    {
        static void BuildTridiagBlockSPDDouble(ref Arena arena, int NB, int BR, out doubleBSR sparse, out int n)
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
                        Di[r, c] = (r == c ? BR * 8f : 0f) + rng.NextFloat(-0.1f, 0.1f);
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

        static double Residual(in doubleBSR A, in doubleN x, in doubleN b)
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

        static string BenchDouble(int BR, int NB, int K)
        {
            const string fmt = "{0,-7} {1,-6} {2,-12} {3,11:F4} {4,11:F4} {5,14:E3}";
            var arena = new Arena(Allocator.Persistent);
            BuildTridiagBlockSPDDouble(ref arena, NB, BR, out var A, out int n);
            var M = arena.doubleBlockJacobi(in A);
            var b = arena.doubleRandomVec(n, -1f, 1f, 0xC001Du);

            var xCg = arena.doubleVec(n); var rCg = arena.doubleVec(n); var pCg = arena.doubleVec(n); var ApCg = arena.doubleVec(n);
            var cgJob = new CgBsrJobDouble { A = A, b = b, x = xCg, r = rCg, p = pCg, Ap = ApCg, K = K };
            var cgStat = Bench.Time(() => cgJob.Run());
            var sb = new StringBuilder();
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "double", n, "CG", cgStat.Median, cgStat.Min, Residual(in A, in xCg, in b)));

            var xPcg = arena.doubleVec(n); var rPcg = arena.doubleVec(n); var pPcg = arena.doubleVec(n); var ApPcg = arena.doubleVec(n); var zPcg = arena.doubleVec(n);
            var pcgJob = new PcgBsrJobDouble { A = A, M = M, b = b, x = xPcg, r = rPcg, p = pPcg, Ap = ApPcg, z = zPcg, K = K };
            var pcgStat = Bench.Time(() => pcgJob.Run());
            sb.Append(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "double", n, "PCG-Jacobi", pcgStat.Median, pcgStat.Min, Residual(in A, in xPcg, in b)));

            arena.Dispose();
            return sb.ToString();
        }
    }
}
