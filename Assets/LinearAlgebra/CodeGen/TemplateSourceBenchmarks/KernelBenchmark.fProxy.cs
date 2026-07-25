using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;
using BULA.Internal;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of KernelBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Kern selectors, Reductions table, REPS constants, Run, Section) is
    // hand-written in Assets/LinearAlgebra/Benchmarks/KernelBenchmark.cs. See that file for the
    // measurement rationale (reduction vectorisation, ping-pong overflow guard, GFLOP/s ratio tell).

    // ---- Level-1 reduction microbench ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ReduceJobFProxy : IJob
    {
        public fProxyN a, b, sink;
        public int kind, reps;
        public void Execute()
        {
            int n = a.N;
            fProxy acc = 0f;
            for (int k = 0; k < reps; k++)
            {
                switch (kind)
                {
                    case Kern.VecDot: acc += Blas.dot(a, b);  break;
                    case Kern.L1:     acc += Norms.L1(a);     break;
                    case Kern.L2:     acc += Norms.L2(a);     break;
                    case Kern.LInf:   acc += Norms.LInf(a);   break;
                    case Kern.Sum:    acc += Stats.sum(a);    break;
                    case Kern.Max:    acc += Stats.max(a);    break;
                    case Kern.Min:    acc += Stats.min(a);    break;
                }
                a[k % n] += acc * (fProxy)1e-30;
            }
            sink[0] = acc;
        }
    }

    // ---- Level-2 matrix-vector microbench: ping-pong x<->y ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatVecJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN x, y;
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
    public struct AxpyJobFProxy : IJob
    {
        public fProxyN x, y;
        public fProxy alpha;
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
        static string ReduceFProxy(int n, int kind, double flopPerElem, int reps)
        {
            var a = new fProxyN(n, Allocator.Persistent);
            var b = new fProxyN(n, Allocator.Persistent);
            var sink = new fProxyN(1, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { a[i] = rng.NextFProxy(-1f, 1f); b[i] = rng.NextFProxy(-1f, 1f); }

            var job = new ReduceJobFProxy { a = a, b = b, sink = sink, kind = kind, reps = reps };
            var stat = Bench.Time(() => job.Run());

            a.Dispose(); b.Dispose(); sink.Dispose();
            return Bench.Row("fProxy", n, stat, reps * flopPerElem * n);
        }

        // ---- Level-1 axpy runners (2 flops/elem: one multiply + one add) ----
        static string AxpyFProxy(int n, int reps)
        {
            var x = new fProxyN(n, Allocator.Persistent);
            var y = new fProxyN(n, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++) { x[i] = rng.NextFProxy(-1f, 1f); y[i] = rng.NextFProxy(-1f, 1f); }

            var job = new AxpyJobFProxy { x = x, y = y, alpha = (fProxy)0.001, reps = reps };
            var stat = Bench.Time(() => job.Run());

            x.Dispose(); y.Dispose();
            return Bench.Row("fProxy", n, stat, reps * 2.0 * n);
        }

        // ---- Level-2 matrix-vector runners (2 N^2 flops/call) ----
        static string MatVecFProxy(int n, int kind, int reps)
        {
            var A = new fProxyMxN(n, n, Allocator.Persistent);
            var x = new fProxyN(n, Allocator.Persistent);
            var y = new fProxyN(n, Allocator.Persistent);
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            fProxy s = (fProxy)1 / Unity.Mathematics.math.sqrt(n); // norm-preserving scale: keeps the ping-pong finite
            for (int i = 0; i < n; i++)
            {
                x[i] = rng.NextFProxy(-1f, 1f);
                for (int j = 0; j < n; j++) A[i, j] = rng.NextFProxy(-1f, 1f) * s;
            }

            var job = new MatVecJobFProxy { A = A, x = x, y = y, kind = kind, reps = reps };
            var stat = Bench.Time(() => job.Run());

            A.Dispose(); x.Dispose(); y.Dispose();
            return Bench.Row("fProxy", n, stat, reps * 2.0 * n * n);
        }
    }
}
