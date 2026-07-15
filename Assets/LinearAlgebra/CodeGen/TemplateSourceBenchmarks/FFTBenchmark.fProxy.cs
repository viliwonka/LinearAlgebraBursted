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

    // ---- table-indexed FFT ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftTableJobFProxy : IJob
    {
        public fProxyN re;
        public fProxyN im;
        public fProxyN srcRe;
        public fProxyN srcIm;
        public fProxyFFTCache ws;

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            FFT.fft(ref re, ref im, in ws);
        }
    }

    // ---- table-indexed complex inverse FFT ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IfftTableJobFProxy : IJob
    {
        public fProxyN re;
        public fProxyN im;
        public fProxyN srcRe;
        public fProxyN srcIm;
        public fProxyFFTCache ws;

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            FFT.ifft(ref re, ref im, in ws);
        }
    }

    // ---- table FFT WITH in-job table build (one-shot cost, build clocked in Burst) ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftBuildRunJobFProxy : IJob
    {
        public fProxyN re;
        public fProxyN im;
        public fProxyN srcRe;
        public fProxyN srcIm;
        public int n;

        // Builds the workspace from scratch (the cos/sin table build, Burst-compiled) then runs one
        // transform — the true one-shot cost of the table path, vs the reuse rows that build once.
        // The in-job Persistent arena is deliberate: its alloc/free is part of the one-shot cost
        // this row measures.
        public void Execute()
        {
            var a = new Arena(Allocator.Persistent);
            var ws = a.fProxyFFTCache(n);
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            FFT.fft(ref re, ref im, in ws);
            a.Dispose();
        }
    }

    // ---- table-indexed real-input rfft ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftTableJobFProxy : IJob
    {
        public fProxyN real;
        public fProxyN re;
        public fProxyN im;
        public fProxyFFTCache ws;   // built once, outside the timed loop

        public void Execute() => FFT.rfft(in real, ref re, ref im, in ws);
    }

    // ---- table-indexed real inverse irfft ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct IrfftTableJobFProxy : IJob
    {
        public fProxyN re;     // half-spectrum input (N/2+1), NOT modified
        public fProxyN im;     // half-spectrum input (N/2+1), NOT modified
        public fProxyN real;   // output length N, overwritten each Execute
        public fProxyFFTCache ws;   // built once, outside the timed loop

        public void Execute() => FFT.irfft(in re, in im, ref real, in ws);
    }

    // ---- direct DFT ----
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DftJobFProxy : IJob
    {
        public fProxyN inRe;     // NOT modified
        public fProxyN inIm;     // NOT modified
        public fProxyN outRe;
        public fProxyN outIm;

        public void Execute() => FFT.dft(in inRe, in inIm, ref outRe, ref outIm);
    }

    public static partial class FFTBenchmark
    {
        // ---- table FFT helpers ----
        static string FftTableFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.fProxyVec(n);
            var im    = arena.fProxyVec(n);
            var srcRe = arena.fProxyVec(n);
            var srcIm = arena.fProxyVec(n);
            var ws    = arena.fProxyFFTCache(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextFProxy(-1f, 1f);
                srcIm[i] = rng.NextFProxy(-1f, 1f);
            }

            var job = new FftTableJobFProxy { re = re, im = im, srcRe = srcRe, srcIm = srcIm, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy(ws)", n, stat);
        }

        // ---- table complex inverse FFT helpers ----
        static string IfftTableFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.fProxyVec(n);
            var im    = arena.fProxyVec(n);
            var srcRe = arena.fProxyVec(n);
            var srcIm = arena.fProxyVec(n);
            var ws    = arena.fProxyFFTCache(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextFProxy(-1f, 1f);
                srcIm[i] = rng.NextFProxy(-1f, 1f);
            }

            var job = new IfftTableJobFProxy { re = re, im = im, srcRe = srcRe, srcIm = srcIm, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy(ws)", n, stat);
        }

        // ---- table FFT WITH build included (one-shot, build clocked in Burst) ----
        static string FftTableBuiltFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.fProxyVec(n);
            var im    = arena.fProxyVec(n);
            var srcRe = arena.fProxyVec(n);
            var srcIm = arena.fProxyVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextFProxy(-1f, 1f);
                srcIm[i] = rng.NextFProxy(-1f, 1f);
            }

            var job = new FftBuildRunJobFProxy { re = re, im = im, srcRe = srcRe, srcIm = srcIm, n = n };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy(ws+build)", n, stat);
        }

        // ---- table rfft helpers ----
        static string RfftTableFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.fProxyVec(n);
            var re    = arena.fProxyVec(n / 2 + 1);
            var im    = arena.fProxyVec(n / 2 + 1);
            var ws    = arena.fProxyFFTCache(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextFProxy(-1f, 1f);

            var job = new RfftTableJobFProxy { real = real, re = re, im = im, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy(ws)", n, stat);
        }

        // ---- table real inverse irfft helpers ----
        static string IrfftTableFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            int h     = n / 2 + 1;
            var re    = arena.fProxyVec(h);
            var im    = arena.fProxyVec(h);
            var real  = arena.fProxyVec(n);
            var ws    = arena.fProxyFFTCache(n);   // built ONCE outside the timed loop

            // Arbitrary half-spectrum input (exact Hermitian validity is irrelevant to timing).
            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xB16B00B5u);
            for (int i = 0; i < h; i++)
            {
                re[i] = rng.NextFProxy(-1f, 1f);
                im[i] = rng.NextFProxy(-1f, 1f);
            }

            var job = new IrfftTableJobFProxy { re = re, im = im, real = real, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy(ws)", n, stat);
        }

        // ---- DFT helpers ----
        static string DftFProxy(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var inRe  = arena.fProxyVec(n);
            var inIm  = arena.fProxyVec(n);
            var outRe = arena.fProxyVec(n);
            var outIm = arena.fProxyVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                inRe[i] = rng.NextFProxy(-1f, 1f);
                inIm[i] = rng.NextFProxy(-1f, 1f);
            }

            var job = new DftJobFProxy { inRe = inRe, inIm = inIm, outRe = outRe, outIm = outIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("fProxy", n, stat);
        }
    }
}
