using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of KernelBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Kern selectors, Reductions table, REPS constants, Run, Section) is
    // hand-written in Assets/LinearAlgebra/Benchmarks/KernelBenchmark.cs. See that file for the
    // measurement rationale (reduction vectorisation, ping-pong overflow guard, GFLOP/s ratio tell).

    // ---- Level-1 reduction microbench ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ReduceJobDouble : IJob
    {
        public doubleN a, b, sink;
        public int kind, reps;
        public void Execute()
        {
            int n = a.N;
            double acc = 0f;
            for (int k = 0; k < reps; k++)
            {
                switch (kind)
                {
                    case Kern.VecDot: acc += Blas.dot(a, b);  break;
                    case Kern.L1:     acc += Norms.L1(a);     break;
                    case Kern.L2:     acc += Norms.L2(a);     break;
                    case Kern.LInf:   acc += Norms.LInf(a);   break;
                    case Kern.Sum:    acc += Stats.sum(a);    break;
                }
                a[k % n] += acc * (double)1e-30;
            }
            sink[0] = acc;
        }
    }

    // ---- Level-2 matrix-vector microbench: ping-pong x<->y ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatVecJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN x, y;
        public int kind, reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if (kind == Kern.Gemv)
                {
                    if ((k & 1) == 0) Blas.dot(in A, in x, ref y);   // y = A x
                    else              Blas.dot(in A, in y, ref x);
                }
                else // VecMat: Aᵀx via vector*matrix
                {
                    if ((k & 1) == 0) Blas.dot(in x, in A, ref y);   // y = Aᵀ x
                    else              Blas.dot(in y, in A, ref x);
                }
            }
        }
    }

    // ---- Level-1 axpy microbench: ping-pong which vector is updated ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct AxpyJobDouble : IJob
    {
        public doubleN x, y;
        public double alpha;
        public int reps;
        public unsafe void Execute()
        {
            int n = x.N;
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) UnsafeOP.axpy(y.Data.Ptr, x.Data.Ptr, alpha, n); // y += a x
                else              UnsafeOP.axpy(x.Data.Ptr, y.Data.Ptr, alpha, n);
            }
        }
    }

    public static partial class KernelBenchmark
    {
        // ---- Level-1 reduction runners ----
        static string ReduceDouble(int n, int kind, double flopPerElem, int reps)
        {
            var arena = new Arena(Allocator.Persistent);
            var a = arena.doubleVec(n);
            var b = arena.doubleVec(n);
            var sink = arena.doubleVec(1);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { a[i] = rng.NextDouble(-1f, 1f); b[i] = rng.NextDouble(-1f, 1f); }

            var job = new ReduceJobDouble { a = a, b = b, sink = sink, kind = kind, reps = reps };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, reps * flopPerElem * n);
        }

        // ---- Level-1 axpy runners (2 flops/elem: one multiply + one add) ----
        static string AxpyDouble(int n, int reps)
        {
            var arena = new Arena(Allocator.Persistent);
            var x = arena.doubleVec(n);
            var y = arena.doubleVec(n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { x[i] = rng.NextDouble(-1f, 1f); y[i] = rng.NextDouble(-1f, 1f); }

            var job = new AxpyJobDouble { x = x, y = y, alpha = (double)0.001, reps = reps };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, reps * 2.0 * n);
        }

        // ---- Level-2 matrix-vector runners (2 N^2 flops/call) ----
        static string MatVecDouble(int n, int kind, int reps)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var x = arena.doubleVec(n);
            var y = arena.doubleVec(n);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            double s = (double)1 / Unity.Mathematics.math.sqrt(n); // norm-preserving scale: keeps the ping-pong finite
            for (int i = 0; i < n; i++)
            {
                x[i] = rng.NextDouble(-1f, 1f);
                for (int j = 0; j < n; j++) A[i, j] = rng.NextDouble(-1f, 1f) * s;
            }

            var job = new MatVecJobDouble { A = A, x = x, y = y, kind = kind, reps = reps };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, reps * 2.0 * n * n);
        }
    }
}
