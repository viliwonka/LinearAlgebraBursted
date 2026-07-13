using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebra.Benchmarks
{
    // Machine roofline probe — NOT a library-kernel benchmark. Two ceilings every kernel number
    // in the other sections should be read against:
    //   A/B: compute-bound width-4 mul+add chains in registers (zero memory traffic) under
    //        FloatMode.Default (the library's mode: no FMA contraction) and FloatMode.Fast
    //        (FMA allowed) — the gap between A and B is what Strict-mode determinism costs at
    //        the ALU limit; B approximates the machine's SIMD FLOP ceiling per core.
    //   C:   memory-bound STREAM triad over buffers far beyond L3 — sustained single-core GB/s.
    //
    // Standalone: run with Tools/benchmark.ps1 -Method LinearAlgebra.Benchmarks.RooflineBenchmark.Run
    // (not part of AllBenchmarks).
    //
    // Hand-written harness half. The timed IJobs and build+measure methods are code-generated per
    // dtype from Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/RooflineBenchmark.fProxy.cs.
    // 8-lane experiment (hand-written, both dtypes explicit): the same chain shape as sections
    // A/B but on 4x2 vector types — two adjacent width-4 fields per value. Probes whether Burst's
    // SLP pass fuses the two halves into one 256-bit op (which would double float throughput over
    // float4 kernels) or leaves them as two 128-bit ops. One step = 8 chains x (mul + add) x
    // 8 lanes = 128 scalar flops.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    struct RooflineComputeF4x2Job : IJob
    {
        public int Steps;
        public float Scale;
        public float Bias;
        public NativeArray<float> Result;

        public void Execute() => Result[0] = RooflineWideKernels.ChainsF4x2(Steps, Scale, Bias);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Fast)]
    struct RooflineComputeF4x2FastJob : IJob
    {
        public int Steps;
        public float Scale;
        public float Bias;
        public NativeArray<float> Result;

        public void Execute() => Result[0] = RooflineWideKernels.ChainsF4x2(Steps, Scale, Bias);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    struct RooflineComputeD4x2Job : IJob
    {
        public int Steps;
        public double Scale;
        public double Bias;
        public NativeArray<double> Result;

        public void Execute() => Result[0] = RooflineWideKernels.ChainsD4x2(Steps, Scale, Bias);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Fast)]
    struct RooflineComputeD4x2FastJob : IJob
    {
        public int Steps;
        public double Scale;
        public double Bias;
        public NativeArray<double> Result;

        public void Execute() => Result[0] = RooflineWideKernels.ChainsD4x2(Steps, Scale, Bias);
    }

    // True 256-bit probe via Burst's AVX intrinsics (what a hand-rolled or MaxMath-style float8
    // would compile to): 8 chains of v256 mul+add, and an FMA variant. Float only — double at
    // 256 bits is what double4 already is.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    struct RooflineComputeV256Job : IJob
    {
        public int Steps;
        public float Scale;
        public float Bias;
        public NativeArray<float> Result;

        public void Execute()
        {
            if (Unity.Burst.Intrinsics.X86.Avx.IsAvxSupported)
                Result[0] = RooflineWideKernels.ChainsV256(Steps, Scale, Bias, fma: false);
            else
                Result[0] = 0f;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    struct RooflineComputeV256FmaJob : IJob
    {
        public int Steps;
        public float Scale;
        public float Bias;
        public NativeArray<float> Result;

        public void Execute()
        {
            if (Unity.Burst.Intrinsics.X86.Fma.IsFmaSupported)
                Result[0] = RooflineWideKernels.ChainsV256(Steps, Scale, Bias, fma: true);
            else
                Result[0] = 0f;
        }
    }

    static class RooflineWideKernels
    {
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static float ChainsV256(int steps, float scale, float bias, bool fma)
        {
            var s = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(scale);
            var b = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(bias);
            var a0 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.01f);
            var a1 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.02f);
            var a2 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.03f);
            var a3 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.04f);
            var a4 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.05f);
            var a5 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.06f);
            var a6 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.07f);
            var a7 = Unity.Burst.Intrinsics.X86.Avx.mm256_set1_ps(1.08f);

            if (fma)
            {
                for (int i = 0; i < steps; i++)
                {
                    a0 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a0, s, b);
                    a1 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a1, s, b);
                    a2 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a2, s, b);
                    a3 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a3, s, b);
                    a4 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a4, s, b);
                    a5 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a5, s, b);
                    a6 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a6, s, b);
                    a7 = Unity.Burst.Intrinsics.X86.Fma.mm256_fmadd_ps(a7, s, b);
                }
            }
            else
            {
                for (int i = 0; i < steps; i++)
                {
                    a0 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a0, s), b);
                    a1 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a1, s), b);
                    a2 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a2, s), b);
                    a3 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a3, s), b);
                    a4 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a4, s), b);
                    a5 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a5, s), b);
                    a6 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a6, s), b);
                    a7 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(a7, s), b);
                }
            }

            var r = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(
                        Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(
                            Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(a0, a1),
                            Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(a2, a3)),
                        Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(
                            Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(a4, a5),
                            Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(a6, a7)));

            var lo = Unity.Burst.Intrinsics.X86.Avx.mm256_castps256_ps128(r);
            var hi = Unity.Burst.Intrinsics.X86.Avx.mm256_extractf128_ps(r, 1);
            var q = Unity.Burst.Intrinsics.X86.Sse.add_ps(lo, hi);
            return (q.Float0 + q.Float1) + (q.Float2 + q.Float3);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static float ChainsF4x2(int steps, float scale, float bias)
        {
            float4x2 s = new float4x2(scale);
            float4x2 b = new float4x2(bias);
            float4x2 a0 = new float4x2(1.01f);
            float4x2 a1 = new float4x2(1.02f);
            float4x2 a2 = new float4x2(1.03f);
            float4x2 a3 = new float4x2(1.04f);
            float4x2 a4 = new float4x2(1.05f);
            float4x2 a5 = new float4x2(1.06f);
            float4x2 a6 = new float4x2(1.07f);
            float4x2 a7 = new float4x2(1.08f);

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

            float4x2 r = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            float4 q = r.c0 + r.c1;
            return (q.x + q.y) + (q.z + q.w);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static double ChainsD4x2(int steps, double scale, double bias)
        {
            double4x2 s = new double4x2(scale);
            double4x2 b = new double4x2(bias);
            double4x2 a0 = new double4x2(1.01);
            double4x2 a1 = new double4x2(1.02);
            double4x2 a2 = new double4x2(1.03);
            double4x2 a3 = new double4x2(1.04);
            double4x2 a4 = new double4x2(1.05);
            double4x2 a5 = new double4x2(1.06);
            double4x2 a6 = new double4x2(1.07);
            double4x2 a7 = new double4x2(1.08);

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

            double4x2 r = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            double4 q = r.c0 + r.c1;
            return (q.x + q.y) + (q.z + q.w);
        }
    }

    public static partial class RooflineBenchmark
    {
        // N column below = millions (of scalar flops for A/B, of elements for C).
        static readonly int[] Millions = { 1, 4, 8, 16, 32, 64 };

        public static void Run() => Bench.WriteReport("benchmark-roofline.txt", Section);

        static string BenchCompute4x2Float(int millionFlops, bool fast)
        {
            int steps = (int)((long)millionFlops * 1000000 / 128);
            var res = new NativeArray<float>(1, Allocator.Persistent);
            Bench.Stat stat;
            if (fast)
            {
                var job = new RooflineComputeF4x2FastJob { Steps = steps, Scale = 1f, Bias = 0f, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            else
            {
                var job = new RooflineComputeF4x2Job { Steps = steps, Scale = 1f, Bias = 0f, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            res.Dispose();
            return Bench.Row("float", millionFlops, stat, (double)millionFlops * 1e6);
        }

        static string BenchComputeV256(int millionFlops, bool fma)
        {
            int steps = (int)((long)millionFlops * 1000000 / 128);
            var res = new NativeArray<float>(1, Allocator.Persistent);
            Bench.Stat stat;
            if (fma)
            {
                var job = new RooflineComputeV256FmaJob { Steps = steps, Scale = 1f, Bias = 0f, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            else
            {
                var job = new RooflineComputeV256Job { Steps = steps, Scale = 1f, Bias = 0f, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            res.Dispose();
            return Bench.Row("float", millionFlops, stat, (double)millionFlops * 1e6);
        }

        static string BenchCompute4x2Double(int millionFlops, bool fast)
        {
            int steps = (int)((long)millionFlops * 1000000 / 128);
            var res = new NativeArray<double>(1, Allocator.Persistent);
            Bench.Stat stat;
            if (fast)
            {
                var job = new RooflineComputeD4x2FastJob { Steps = steps, Scale = 1.0, Bias = 0.0, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            else
            {
                var job = new RooflineComputeD4x2Job { Steps = steps, Scale = 1.0, Bias = 0.0, Result = res };
                stat = Bench.Time(() => job.Run());
            }
            res.Dispose();
            return Bench.Row("double", millionFlops, stat, (double)millionFlops * 1e6);
        }

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Roofline A: compute-bound, FloatMode.Default — 8 independent width-4 chains, a = a*s + b (N = millions of scalar flops) ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchComputeFloat(m, fast: false));
            foreach (var m in Millions) sb.AppendLine(BenchComputeDouble(m, fast: false));
            sb.AppendLine();

            sb.AppendLine("=== Roofline B: compute-bound, FloatMode.Fast — same chains, FMA contraction allowed ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchComputeFloat(m, fast: true));
            foreach (var m in Millions) sb.AppendLine(BenchComputeDouble(m, fast: true));
            sb.AppendLine();

            sb.AppendLine("=== Roofline C: memory-bound STREAM triad y = x*s + y (N = millions of ELEMENTS; last column reads GB/s, 2 reads + 1 write nominal) ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchMemoryFloat(m));
            foreach (var m in Millions) sb.AppendLine(BenchMemoryDouble(m));
            sb.AppendLine();

            sb.AppendLine("=== Roofline D: compute-bound, FloatMode.Default — same chains on 4x2 types (8 lanes/value; does SLP fuse the halves to 256-bit?) ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchCompute4x2Float(m, fast: false));
            foreach (var m in Millions) sb.AppendLine(BenchCompute4x2Double(m, fast: false));
            sb.AppendLine();

            sb.AppendLine("=== Roofline E: compute-bound, FloatMode.Fast — 4x2 chains, FMA contraction allowed ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchCompute4x2Float(m, fast: true));
            foreach (var m in Millions) sb.AppendLine(BenchCompute4x2Double(m, fast: true));
            sb.AppendLine();

            sb.AppendLine("=== Roofline F: compute-bound, explicit AVX v256 intrinsics — 8 chains, mm256 mul+add (float8 done by hand) ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchComputeV256(m, fma: false));
            sb.AppendLine();

            sb.AppendLine("=== Roofline G: compute-bound, explicit AVX v256 intrinsics — 8 chains, mm256_fmadd ===");
            sb.AppendLine(Bench.Header());
            foreach (var m in Millions) sb.AppendLine(BenchComputeV256(m, fma: true));
            sb.AppendLine();
        }
    }
}
