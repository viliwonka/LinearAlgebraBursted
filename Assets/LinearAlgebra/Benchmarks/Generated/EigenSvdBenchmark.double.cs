using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of EigenSvdBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/EigenSvdBenchmark.cs.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdValuesJobDouble : IJob
    {
        public doubleMxN A;     // not modified (values works on a Temp copy)
        public doubleN S;

        public void Execute() => SVD.values(in A, ref S);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN E;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.valuesSymmetric(ref A, ref E);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymVecJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN E;
        public doubleMxN V;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.symmetric(ref A, ref E, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigQRJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN Re;
        public doubleN Im;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.valuesQR(ref A, ref Re, ref Im, 100);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdGKJobDouble : IJob
    {
        public doubleMxN A;     // input, not modified (thin takes A `in`)
        public doubleMxN U;
        public doubleN S;
        public doubleMxN V;

        public void Execute() => SVD.thin(in A, ref U, ref S, ref V);
    }

    public static partial class EigenSvdBenchmark
    {
        static string SvdGKDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var U = arena.doubleMat(n, n);
            var S = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1f, 1f);

            var job = new SvdGKJobDouble { A = A, U = U, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string SvdValsDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var S = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1f, 1f);

            var job = new SvdValuesJobDouble { A = A, S = S };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigSymDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var E = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigSymJobDouble { A = A, Src = Src, E = E };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigSymVecDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var E = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }

            var job = new EigSymVecJobDouble { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        static string EigQRDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var Re = arena.doubleVec(n);
            var Im = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1f, 1f);

            var job = new EigQRJobDouble { A = A, Src = Src, Re = Re, Im = Im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
