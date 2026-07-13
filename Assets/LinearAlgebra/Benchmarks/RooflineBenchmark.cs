using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
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

    static partial class RooflineWideKernels
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

    // Real-kernel-shape probe: dot product (the library's width-4 reduction family) — the
    // current Blas.dot route vs a hand-rolled v256 version at the same Strict semantics
    // (separate mul + add, fixed chain tree). Reps batched so tiny sizes are measurable;
    // results accumulated so nothing dead-code-eliminates.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    struct RooflineDotLibFloatJob : IJob
    {
        public floatN A;
        public floatN B;
        public int Reps;
        public NativeArray<float> Result;

        public void Execute()
        {
            float acc = 0f;
            for (int r = 0; r < Reps; r++)
                acc += Blas.dot(A, B);
            Result[0] = acc;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    struct RooflineDotLibDoubleJob : IJob
    {
        public doubleN A;
        public doubleN B;
        public int Reps;
        public NativeArray<double> Result;

        public void Execute()
        {
            double acc = 0.0;
            for (int r = 0; r < Reps; r++)
                acc += Blas.dot(A, B);
            Result[0] = acc;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    unsafe struct RooflineDotV256Job : IJob
    {
        [ReadOnly] public NativeArray<float> A;
        [ReadOnly] public NativeArray<float> B;
        public int Reps;
        public NativeArray<float> Result;

        public void Execute()
        {
            if (!Unity.Burst.Intrinsics.X86.Avx.IsAvxSupported)
            {
                Result[0] = 0f;
                return;
            }

            float acc = 0f;
            for (int r = 0; r < Reps; r++)
                acc += RooflineWideKernels.DotV256((float*)A.GetUnsafeReadOnlyPtr(),
                                                   (float*)B.GetUnsafeReadOnlyPtr(), A.Length);
            Result[0] = acc;
        }
    }

    static partial class RooflineWideKernels
    {
        // v256 dot: 4 independent 8-lane accumulator chains (32 lane-chains total), separate
        // mul + add (the library's Strict semantics — no FMA, no reassociation), fixed
        // reduction tree, ascending scalar tail.
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotV256(float* a, float* b, int n)
        {
            var pa = (Unity.Burst.Intrinsics.v256*)a;
            var pb = (Unity.Burst.Intrinsics.v256*)b;
            int nO = n >> 3;

            var acc0 = Unity.Burst.Intrinsics.X86.Avx.mm256_setzero_ps();
            var acc1 = Unity.Burst.Intrinsics.X86.Avx.mm256_setzero_ps();
            var acc2 = Unity.Burst.Intrinsics.X86.Avx.mm256_setzero_ps();
            var acc3 = Unity.Burst.Intrinsics.X86.Avx.mm256_setzero_ps();

            int o = 0;
            for (; o + 4 <= nO; o += 4)
            {
                acc0 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc0, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(pa[o],     pb[o]));
                acc1 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc1, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(pa[o + 1], pb[o + 1]));
                acc2 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc2, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(pa[o + 2], pb[o + 2]));
                acc3 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc3, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(pa[o + 3], pb[o + 3]));
            }
            for (; o < nO; o++)
                acc0 = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc0, Unity.Burst.Intrinsics.X86.Avx.mm256_mul_ps(pa[o], pb[o]));

            var acc = Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(
                          Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc0, acc1),
                          Unity.Burst.Intrinsics.X86.Avx.mm256_add_ps(acc2, acc3));
            var lo = Unity.Burst.Intrinsics.X86.Avx.mm256_castps256_ps128(acc);
            var hi = Unity.Burst.Intrinsics.X86.Avx.mm256_extractf128_ps(acc, 1);
            var q = Unity.Burst.Intrinsics.X86.Sse.add_ps(lo, hi);
            float s = (q.Float0 + q.Float1) + (q.Float2 + q.Float3);

            for (int i = nO << 3; i < n; i++)
                s += a[i] * b[i];
            return s;
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

        // Dot-shape rows: N column = THOUSANDS of elements; reps batched to ~8M elements
        // processed per timed sample regardless of N.
        static readonly int[] DotKElems = { 1, 4, 16, 64, 256, 1024, 4096 };

        static int DotReps(int n) => math.max(1, 8388608 / n);

        static string BenchDotLibFloat(int kElems)
        {
            int n = kElems * 1024;
            int reps = DotReps(n);
            var A = new floatN(n, Allocator.Persistent);
            var B = new floatN(n, Allocator.Persistent);
            A.fillInPlace(0.001f);
            B.fillInPlace(0.001f);
            var res = new NativeArray<float>(1, Allocator.Persistent);
            var job = new RooflineDotLibFloatJob { A = A, B = B, Reps = reps, Result = res };
            var stat = Bench.Time(() => job.Run());
            res.Dispose(); A.Dispose(); B.Dispose();
            return Bench.Row("float", kElems, stat, 2.0 * n * reps);
        }

        static string BenchDotLibDouble(int kElems)
        {
            int n = kElems * 1024;
            int reps = DotReps(n);
            var A = new doubleN(n, Allocator.Persistent);
            var B = new doubleN(n, Allocator.Persistent);
            A.fillInPlace(0.001);
            B.fillInPlace(0.001);
            var res = new NativeArray<double>(1, Allocator.Persistent);
            var job = new RooflineDotLibDoubleJob { A = A, B = B, Reps = reps, Result = res };
            var stat = Bench.Time(() => job.Run());
            res.Dispose(); A.Dispose(); B.Dispose();
            return Bench.Row("double", kElems, stat, 2.0 * n * reps);
        }

        static string BenchDotV256(int kElems)
        {
            int n = kElems * 1024;
            int reps = DotReps(n);
            var A = new NativeArray<float>(n, Allocator.Persistent);
            var B = new NativeArray<float>(n, Allocator.Persistent);
            for (int i = 0; i < n; i++) { A[i] = 0.001f; B[i] = 0.001f; }
            var res = new NativeArray<float>(1, Allocator.Persistent);
            var job = new RooflineDotV256Job { A = A, B = B, Reps = reps, Result = res };
            var stat = Bench.Time(() => job.Run());
            res.Dispose(); A.Dispose(); B.Dispose();
            return Bench.Row("float", kElems, stat, 2.0 * n * reps);
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

            sb.AppendLine("=== Roofline H: dot product, library route (Blas.dot, width-4 two-chain kernel; N = THOUSANDS of elements, ~8M processed per sample) ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in DotKElems) sb.AppendLine(BenchDotLibFloat(k));
            foreach (var k in DotKElems) sb.AppendLine(BenchDotLibDouble(k));
            sb.AppendLine();

            sb.AppendLine("=== Roofline I: dot product, v256 intrinsics (4x 8-lane chains, same Strict semantics; float) ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in DotKElems) sb.AppendLine(BenchDotV256(k));
            sb.AppendLine();
        }
    }
}
