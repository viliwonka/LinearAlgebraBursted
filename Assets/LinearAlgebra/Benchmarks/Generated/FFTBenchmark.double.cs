using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of FFTBenchmark (timed IJobs + build+measure methods). The
    // dtype-agnostic harness (size lists, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/FFTBenchmark.cs.

    // ---- radix-2 in-place FFT ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftJobDouble : IJob
    {
        public doubleN re;       // working buffer — overwritten each Execute
        public doubleN im;       // working buffer — overwritten each Execute
        public doubleN srcRe;    // pristine copy of input re
        public doubleN srcIm;    // pristine copy of input im

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            FFT.fft(ref re, ref im);
        }
    }

    // ---- real-input half-spectrum rfft ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftJobDouble : IJob
    {
        public doubleN real;   // NOT modified (rfft takes `in`)
        public doubleN re;     // output — overwritten each Execute
        public doubleN im;     // output — overwritten each Execute

        public void Execute() => FFT.rfft(in real, ref re, ref im);
    }

    // ---- table-indexed FFT ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftTableJobDouble : IJob
    {
        public doubleN re;
        public doubleN im;
        public doubleN srcRe;
        public doubleN srcIm;
        public doubleFFTCache ws;

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            FFT.fft(ref re, ref im, in ws);
        }
    }

    // ---- table FFT WITH in-job table build (one-shot cost, build clocked in Burst) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftBuildRunJobDouble : IJob
    {
        public doubleN re;
        public doubleN im;
        public doubleN srcRe;
        public doubleN srcIm;
        public int n;

        // Builds the workspace from scratch (the cos/sin table build, Burst-compiled) then runs one
        // transform — the true one-shot cost of the table path, vs the reuse rows that build once.
        public void Execute()
        {
            var a = new Arena(Allocator.Persistent);
            var ws = a.doubleFFTCache(n);
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            FFT.fft(ref re, ref im, in ws);
            a.Dispose();
        }
    }

    // ---- table-indexed real-input rfft ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftTableJobDouble : IJob
    {
        public doubleN real;
        public doubleN re;
        public doubleN im;
        public doubleFFTCache ws;   // built once, outside the timed loop

        public void Execute() => FFT.rfft(in real, ref re, ref im, in ws);
    }

    // ---- direct DFT ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DftJobDouble : IJob
    {
        public doubleN inRe;     // NOT modified
        public doubleN inIm;     // NOT modified
        public doubleN outRe;
        public doubleN outIm;

        public void Execute() => FFT.dft(in inRe, in inIm, ref outRe, ref outIm);
    }

    public static partial class FFTBenchmark
    {
        // ---- FFT helpers ----
        static string FftDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.doubleVec(n);
            var im    = arena.doubleVec(n);
            var srcRe = arena.doubleVec(n);
            var srcIm = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextDouble(-1f, 1f);
                srcIm[i] = rng.NextDouble(-1f, 1f);
            }

            var job = new FftJobDouble { re = re, im = im, srcRe = srcRe, srcIm = srcIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- table FFT helpers ----
        static string FftTableDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.doubleVec(n);
            var im    = arena.doubleVec(n);
            var srcRe = arena.doubleVec(n);
            var srcIm = arena.doubleVec(n);
            var ws    = arena.doubleFFTCache(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextDouble(-1f, 1f);
                srcIm[i] = rng.NextDouble(-1f, 1f);
            }

            var job = new FftTableJobDouble { re = re, im = im, srcRe = srcRe, srcIm = srcIm, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double(ws)", n, stat);
        }

        // ---- table FFT WITH build included (one-shot, build clocked in Burst) ----
        static string FftTableBuiltDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.doubleVec(n);
            var im    = arena.doubleVec(n);
            var srcRe = arena.doubleVec(n);
            var srcIm = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextDouble(-1f, 1f);
                srcIm[i] = rng.NextDouble(-1f, 1f);
            }

            var job = new FftBuildRunJobDouble { re = re, im = im, srcRe = srcRe, srcIm = srcIm, n = n };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double(ws+build)", n, stat);
        }

        // ---- rfft helpers ----
        static string RfftDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.doubleVec(n);
            var re    = arena.doubleVec(n / 2 + 1);
            var im    = arena.doubleVec(n / 2 + 1);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextDouble(-1f, 1f);

            var job = new RfftJobDouble { real = real, re = re, im = im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- table rfft helpers ----
        static string RfftTableDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.doubleVec(n);
            var re    = arena.doubleVec(n / 2 + 1);
            var im    = arena.doubleVec(n / 2 + 1);
            var ws    = arena.doubleFFTCache(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextDouble(-1f, 1f);

            var job = new RfftTableJobDouble { real = real, re = re, im = im, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double(ws)", n, stat);
        }

        // ---- DFT helpers ----
        static string DftDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var inRe  = arena.doubleVec(n);
            var inIm  = arena.doubleVec(n);
            var outRe = arena.doubleVec(n);
            var outIm = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                inRe[i] = rng.NextDouble(-1f, 1f);
                inIm[i] = rng.NextDouble(-1f, 1f);
            }

            var job = new DftJobDouble { inRe = inRe, inIm = inIm, outRe = outRe, outIm = outIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
