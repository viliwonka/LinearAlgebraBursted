using System.Runtime.CompilerServices;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites the fProxy4 token -> float4 / double4 (real Unity.Mathematics
// types), so its native operators + field access resolve directly. See docs/dev/spec-alias-simd-proxies.md.
using fProxy4 = Unity.Mathematics.float4;
//-deleteThis

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of RooflineBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (Section, size list) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/RooflineBenchmark.cs.

    public static class RooflineKernelsFProxy
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy4 Splat(fProxy v)
        {
            fProxy4 r = default;
            r.x = v; r.y = v; r.z = v; r.w = v;
            return r;
        }

        // 8 independent width-4 chains, each stepping a = a*s + b: enough parallel chains to hide
        // FP latency, so this measures ISSUE THROUGHPUT, not dependency latency. s/b are runtime
        // values (1 and 0 at every call site) so the compiler can neither fold the chain nor
        // change any value — no overflow, no denormals, at any step count. Inlined into each
        // timed job so it compiles under THAT job's FloatMode (Default = no FMA contraction,
        // Fast = FMA allowed). One step = 8 chains x (mul + add) x 4 lanes = 64 scalar flops.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy ComputeChains(int steps, fProxy scale, fProxy bias)
        {
            fProxy4 s = Splat(scale);
            fProxy4 b = Splat(bias);
            fProxy4 a0 = Splat((fProxy)1.01);
            fProxy4 a1 = Splat((fProxy)1.02);
            fProxy4 a2 = Splat((fProxy)1.03);
            fProxy4 a3 = Splat((fProxy)1.04);
            fProxy4 a4 = Splat((fProxy)1.05);
            fProxy4 a5 = Splat((fProxy)1.06);
            fProxy4 a6 = Splat((fProxy)1.07);
            fProxy4 a7 = Splat((fProxy)1.08);

            for (int i = 0; i < steps; i++)
            {
                a0 = a0 * s + b;
                a1 = a1 * s + b;
                a2 = a2 * s + b;
                a3 = a3 * s + b;
                a4 = a4 * s + b;
                a5 = a5 * s + b;
                a6 = a6 * s + b;
                a7 = a7 * s + b;
            }

            fProxy4 r = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            return (r.x + r.y) + (r.z + r.w);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RooflineComputeJobFProxy : IJob
    {
        public int Steps;
        public fProxy Scale;
        public fProxy Bias;
        public fProxyN Result;   // written so the chains cannot be dead-code-eliminated

        public void Execute() => Result[0] = RooflineKernelsFProxy.ComputeChains(Steps, Scale, Bias);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Fast)]
    public struct RooflineComputeFastJobFProxy : IJob
    {
        public int Steps;
        public fProxy Scale;
        public fProxy Bias;
        public fProxyN Result;

        public void Execute() => Result[0] = RooflineKernelsFProxy.ComputeChains(Steps, Scale, Bias);
    }

    // STREAM-style triad y = x*s + y over width-4 vectors: 2 loads + 1 store per element with
    // trivial math — pure bandwidth at any N past the caches.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public unsafe struct RooflineMemoryJobFProxy : IJob
    {
        public fProxyN X;   // read stream
        public fProxyN Y;   // read-modify-write stream
        public fProxy Scale;

        public void Execute()
        {
            var px = (fProxy4*)X.Data.Ptr;
            var py = (fProxy4*)Y.Data.Ptr;
            fProxy4 s = RooflineKernelsFProxy.Splat(Scale);
            int nQ = X.N >> 2;
            for (int q = 0; q < nQ; q++)
                py[q] = px[q] * s + py[q];
        }
    }

    public static partial class RooflineBenchmark
    {
        static string BenchComputeFProxy(int millionFlops, bool fast)
        {
            int steps = (int)((long)millionFlops * 1000000 / 64);
            var res = new fProxyN(4, Allocator.Persistent);

            Bench.Stat stat;
            if (fast)
            {
                var job = new RooflineComputeFastJobFProxy { Steps = steps, Scale = (fProxy)1, Bias = (fProxy)0, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            else
            {
                var job = new RooflineComputeJobFProxy { Steps = steps, Scale = (fProxy)1, Bias = (fProxy)0, Result = res };
                stat = Bench.Time(() => job.Run());
            }

            res.Dispose();
            return Bench.Row("fProxy", millionFlops, stat, (double)millionFlops * 1e6);
        }

        static string BenchMemoryFProxy(int millionElems)
        {
            int n = (int)((long)millionElems * 1000000);
            var X = new fProxyN(n, Allocator.Persistent);
            var Y = new fProxyN(n, Allocator.Persistent);
            X.fillInPlace((fProxy)1);
            Y.fillInPlace((fProxy)1);

            var job = new RooflineMemoryJobFProxy { X = X, Y = Y, Scale = (fProxy)1 };
            var stat = Bench.Time(() => job.Run());

            X.Dispose();
            Y.Dispose();

            // The "flops" argument is fed nominal BYTES MOVED (2 reads + 1 write per element), so
            // the report's last column reads as GB/s for these rows.
            double bytes = 3.0 * n * UnsafeUtility.SizeOf<fProxy>();
            return Bench.Row("fProxy", millionElems, stat, bytes);
        }
    }
}
