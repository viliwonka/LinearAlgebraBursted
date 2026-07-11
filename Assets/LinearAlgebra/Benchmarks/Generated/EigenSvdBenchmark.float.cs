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
    public struct SvdValuesJobFloat : IJob
    {
        public floatMxN A;     // not modified (values works on a Temp copy)
        public floatN S;

        public void Execute() => SVD.values(in A, ref S);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN E;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.valuesSymmetricInPlace(ref A, ref E);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymVecJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN E;
        public floatMxN V;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.symmetricInPlace(ref A, ref E, ref V);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigQRJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN Re;
        public floatN Im;

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
    public struct SvdGKJobFloat : IJob
    {
        public floatMxN A;     // input, not modified (thin takes A `in`)
        public floatMxN U;
        public floatN S;
        public floatMxN V;

        public void Execute() => SVD.thin(in A, ref U, ref S, ref V);
    }

    public static partial class EigenSvdBenchmark
    {
        static string SvdGKFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);

            var job = new SvdGKJobFloat { A = A, U = U, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string SvdValsFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var S = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);

            var job = new SvdValuesJobFloat { A = A, S = S };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigSymFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var E = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigSymJobFloat { A = A, Src = Src, E = E };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigSymVecFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var E = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }

            var job = new EigSymVecJobFloat { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string EigQRFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var Re = arena.floatVec(n);
            var Im = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);

            var job = new EigQRJobFloat { A = A, Src = Src, Re = Re, Im = Im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }
    }
}
