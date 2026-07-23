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
    public struct SvdValuesJobFProxy : IJob
    {
        public fProxyMxN A;     // not modified (values works on a Temp copy)
        public fProxyN S;

        public void Execute() => SVD.values(in A, ref S);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct EigSymJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Src;
        public fProxyN E;

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
    public struct EigSymVecJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Src;
        public fProxyN E;
        public fProxyMxN V;

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
    public struct EigQRJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyMxN Src;
        public fProxyN Re;
        public fProxyN Im;

        public void Execute()
        {
            int n = A.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = Src[r, c];
            Eigen.valuesQRInPlace(ref A, ref Re, ref Im, 100);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SvdGKJobFProxy : IJob
    {
        public fProxyMxN A;     // input, not modified (thin takes A `in`)
        public fProxyMxN U;
        public fProxyN S;
        public fProxyMxN V;

        public void Execute() => SVD.thin(in A, ref U, ref S, ref V);
    }

    public static partial class EigenSvdBenchmark
    {
        static string SvdGKFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var U = new fProxyMxN(n, n, Allocator.Persistent);
            var S = new fProxyN(n, Allocator.Persistent);
            var V = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);

            var job = new SvdGKJobFProxy { A = A, U = U, S = S, V = V };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); U.Dispose(); S.Dispose(); V.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }

        static string SvdValsFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var S = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFProxy(-1f, 1f);

            var job = new SvdValuesJobFProxy { A = A, S = S };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); S.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }

        static string EigSymFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var E = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;              // exactly symmetric
                }

            var job = new EigSymJobFProxy { A = A, Src = Src, E = E };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); E.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }

        static string EigSymVecFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var E = new fProxyN(n, Allocator.Persistent);
            var V = new fProxyMxN(n, n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    fProxy v = rng.NextFProxy(-1f, 1f);
                    Src[i, j] = v;
                    Src[j, i] = v;
                }

            var job = new EigSymVecJobFProxy { A = A, Src = Src, E = E, V = V };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); E.Dispose(); V.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }

        static string EigQRFProxy(int n)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var Src = new fProxyMxN(n, n, Allocator.Persistent);
            var Re = new fProxyN(n, Allocator.Persistent);
            var Im = new fProxyN(n, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFProxy(-1f, 1f);

            var job = new EigQRJobFProxy { A = A, Src = Src, Re = Re, Im = Im };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); Src.Dispose(); Re.Dispose(); Im.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }
    }
}
