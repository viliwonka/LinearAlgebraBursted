using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // 1D FFT and DFT benchmarks. Own size arrays (not Bench.Sizes — which is for square matrices):
    //   FFT: radix-2 in-place, lengths must be power of two, O(N log N).
    //   DFT: direct O(N²) for arbitrary N (smaller sizes since it's quadratic).
    //
    // FFT is IN-PLACE so re/im are destroyed each call; the job copies pristine srcRe/srcIm
    // into re/im at the start of each Execute so every timed sample does identical work.
    // DFT inputs are `in` (not modified), outputs are separate; no copy needed each call.
    // float-only for FFT and DFT; doubleFFT_OP mirrors the API.

    // ---- radix-2 in-place FFT (float) ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftJobFloat : IJob
    {
        public floatN re;       // working buffer — overwritten each Execute
        public floatN im;       // working buffer — overwritten each Execute
        public floatN srcRe;    // pristine copy of input re
        public floatN srcIm;    // pristine copy of input im

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            floatFFT_OP.fft(ref re, ref im);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftJobDouble : IJob
    {
        public doubleN re;
        public doubleN im;
        public doubleN srcRe;
        public doubleN srcIm;

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            doubleFFT_OP.fft(ref re, ref im);
        }
    }

    // ---- real-input half-spectrum rfft (float) ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftJobFloat : IJob
    {
        public floatN real;   // NOT modified (rfft takes `in`)
        public floatN re;     // output — overwritten each Execute
        public floatN im;     // output — overwritten each Execute

        public void Execute() => floatFFT_OP.rfft(in real, ref re, ref im);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftJobDouble : IJob
    {
        public doubleN real;
        public doubleN re;
        public doubleN im;

        public void Execute() => doubleFFT_OP.rfft(in real, ref re, ref im);
    }

    // ---- table-indexed FFT (float) ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftTableJobFloat : IJob
    {
        public floatN re;
        public floatN im;
        public floatN srcRe;
        public floatN srcIm;
        public floatFft_WS ws;

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            floatFFT_OP.fft(ref re, ref im, in ws);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftTableJobDouble : IJob
    {
        public doubleN re;
        public doubleN im;
        public doubleN srcRe;
        public doubleN srcIm;
        public doubleFft_WS ws;

        public void Execute()
        {
            int n = srcRe.N;
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            doubleFFT_OP.fft(ref re, ref im, in ws);
        }
    }

    // ---- table FFT WITH in-job table build (one-shot cost, build clocked in Burst) ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftBuildRunJobFloat : IJob
    {
        public floatN re;
        public floatN im;
        public floatN srcRe;
        public floatN srcIm;
        public int n;

        // Builds the workspace from scratch (the cos/sin table build, Burst-compiled) then runs one
        // transform — the true one-shot cost of the table path, vs the reuse rows that build once.
        public void Execute()
        {
            var a = new Arena(Allocator.Persistent);
            var ws = a.floatFft_WS(n);
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            floatFFT_OP.fft(ref re, ref im, in ws);
            a.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FftBuildRunJobDouble : IJob
    {
        public doubleN re;
        public doubleN im;
        public doubleN srcRe;
        public doubleN srcIm;
        public int n;

        public void Execute()
        {
            var a = new Arena(Allocator.Persistent);
            var ws = a.doubleFft_WS(n);
            for (int i = 0; i < n; i++) { re[i] = srcRe[i]; im[i] = srcIm[i]; }
            doubleFFT_OP.fft(ref re, ref im, in ws);
            a.Dispose();
        }
    }

    // ---- table-indexed real-input rfft (float) ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftTableJobFloat : IJob
    {
        public floatN real;
        public floatN re;
        public floatN im;
        public floatFft_WS ws;   // built once, outside the timed loop

        public void Execute() => floatFFT_OP.rfft(in real, ref re, ref im, in ws);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct RfftTableJobDouble : IJob
    {
        public doubleN real;
        public doubleN re;
        public doubleN im;
        public doubleFft_WS ws;

        public void Execute() => doubleFFT_OP.rfft(in real, ref re, ref im, in ws);
    }

    // ---- direct DFT (float) ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DftJobFloat : IJob
    {
        public floatN inRe;     // NOT modified
        public floatN inIm;     // NOT modified
        public floatN outRe;
        public floatN outIm;

        public void Execute() => floatFFT_OP.dft(in inRe, in inIm, ref outRe, ref outIm);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct DftJobDouble : IJob
    {
        public doubleN inRe;
        public doubleN inIm;
        public doubleN outRe;
        public doubleN outIm;

        public void Execute() => doubleFFT_OP.dft(in inRe, in inIm, ref outRe, ref outIm);
    }

    public static class FFTBenchmark
    {
        // Power-of-two sizes for the O(N log N) FFT.
        static readonly int[] FftSizes = { 1024, 4096, 16384, 65536, 262144, 1048576 };

        // Smaller sizes for the O(N²) direct DFT (quadratic cost limits feasible N).
        static readonly int[] DftSizes = { 256, 512, 1024, 2048 };

        public static void Run() => Bench.WriteReport("benchmark-fft.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== No-workspace FFT in-place (floatFFT_OP.fft / doubleFFT_OP.fft; O(N log N); ms) ===");
            sb.AppendLine("    Auto-dispatch: power-of-4 → radix-4 recurrence (table-free, zero-alloc); else → radix-2 recurrence.");
            sb.AppendLine("    Input destroyed each call; job copies srcRe/srcIm -> re/im before each run.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(FftFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(FftDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== FFT, twiddle-table workspace — auto radix-4/mixed/radix-2 (floatFFT_OP.fft(ws) / doubleFFT_OP.fft(ws); ms) ===");
            sb.AppendLine("    Workspace built once; dispatches: IsPowerOf4 → radix-4 (log4N passes), 2·4^k → mixed, else → radix-2 table.");
            sb.AppendLine("    No per-element cos/sin; full-circle twiddle table built ONCE outside the timed loop.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(FftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(FftTableDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== FFT, table workspace WITH BUILD INCLUDED (one-shot: build + single transform; build in Burst; ms) ===");
            sb.AppendLine("    Burst job builds the workspace from scratch then runs one fft(ws) — the one-shot cost the reuse rows hide.");
            sb.AppendLine("    Crosses over the no-workspace path after ~1-3 transforms at N>=1024 (table build amortizes fast).");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(FftTableBuiltFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(FftTableBuiltDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Real-input half-spectrum FFT (floatFFT_OP.rfft / doubleFFT_OP.rfft; two-for-one; ms) ===");
            sb.AppendLine("    real input `in` — not modified; re/im output length N/2+1 overwritten each call.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(RfftFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(RfftDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Real-input FFT, twiddle-table workspace (floatFFT_OP.rfft(ws) / doubleFFT_OP.rfft(ws); ms) ===");
            sb.AppendLine("    Workspace built ONCE (arena persistent); no cos/sin in the unpack or butterfly.");
            sb.AppendLine("    Expectation: float no longer slower than double at large N (trig anomaly eliminated).");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(RfftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(RfftTableDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Direct DFT (floatFFT_OP.dft / doubleFFT_OP.dft; O(N^2); ms) ===");
            sb.AppendLine("    Inputs `in` — not modified; output written to separate outRe/outIm.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in DftSizes) sb.AppendLine(DftFloat(n));
            foreach (var n in DftSizes) sb.AppendLine(DftDouble(n));
            sb.AppendLine();
        }

        // ---- FFT helpers ----

        static string FftFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.floatVec(n);
            var im    = arena.floatVec(n);
            var srcRe = arena.floatVec(n);
            var srcIm = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextFloat(-1f, 1f);
                srcIm[i] = rng.NextFloat(-1f, 1f);
            }

            var job = new FftJobFloat { re = re, im = im, srcRe = srcRe, srcIm = srcIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

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
                srcRe[i] = rng.NextDouble(-1.0, 1.0);
                srcIm[i] = rng.NextDouble(-1.0, 1.0);
            }

            var job = new FftJobDouble { re = re, im = im, srcRe = srcRe, srcIm = srcIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- table FFT helpers ----

        static string FftTableFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.floatVec(n);
            var im    = arena.floatVec(n);
            var srcRe = arena.floatVec(n);
            var srcIm = arena.floatVec(n);
            var ws    = arena.floatFft_WS(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextFloat(-1f, 1f);
                srcIm[i] = rng.NextFloat(-1f, 1f);
            }

            var job = new FftTableJobFloat { re = re, im = im, srcRe = srcRe, srcIm = srcIm, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float(ws)", n, stat);
        }

        static string FftTableDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.doubleVec(n);
            var im    = arena.doubleVec(n);
            var srcRe = arena.doubleVec(n);
            var srcIm = arena.doubleVec(n);
            var ws    = arena.doubleFft_WS(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextDouble(-1.0, 1.0);
                srcIm[i] = rng.NextDouble(-1.0, 1.0);
            }

            var job = new FftTableJobDouble { re = re, im = im, srcRe = srcRe, srcIm = srcIm, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double(ws)", n, stat);
        }

        // ---- table FFT WITH build included (one-shot, build clocked in Burst) ----

        static string FftTableBuiltFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var re    = arena.floatVec(n);
            var im    = arena.floatVec(n);
            var srcRe = arena.floatVec(n);
            var srcIm = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                srcRe[i] = rng.NextFloat(-1f, 1f);
                srcIm[i] = rng.NextFloat(-1f, 1f);
            }

            var job = new FftBuildRunJobFloat { re = re, im = im, srcRe = srcRe, srcIm = srcIm, n = n };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float(ws+build)", n, stat);
        }

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
                srcRe[i] = rng.NextDouble(-1.0, 1.0);
                srcIm[i] = rng.NextDouble(-1.0, 1.0);
            }

            var job = new FftBuildRunJobDouble { re = re, im = im, srcRe = srcRe, srcIm = srcIm, n = n };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double(ws+build)", n, stat);
        }

        // ---- rfft helpers ----

        static string RfftFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.floatVec(n);
            var re    = arena.floatVec(n / 2 + 1);
            var im    = arena.floatVec(n / 2 + 1);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextFloat(-1f, 1f);

            var job = new RfftJobFloat { real = real, re = re, im = im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

        static string RfftDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.doubleVec(n);
            var re    = arena.doubleVec(n / 2 + 1);
            var im    = arena.doubleVec(n / 2 + 1);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextDouble(-1.0, 1.0);

            var job = new RfftJobDouble { real = real, re = re, im = im };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }

        // ---- table rfft helpers ----

        static string RfftTableFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.floatVec(n);
            var re    = arena.floatVec(n / 2 + 1);
            var im    = arena.floatVec(n / 2 + 1);
            var ws    = arena.floatFft_WS(n);   // built ONCE outside the timed loop

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextFloat(-1f, 1f);

            var job = new RfftTableJobFloat { real = real, re = re, im = im, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float(ws)", n, stat);
        }

        static string RfftTableDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var real  = arena.doubleVec(n);
            var re    = arena.doubleVec(n / 2 + 1);
            var im    = arena.doubleVec(n / 2 + 1);
            var ws    = arena.doubleFft_WS(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n ^ 0xDEADBEEFu);
            for (int i = 0; i < n; i++)
                real[i] = rng.NextDouble(-1.0, 1.0);

            var job = new RfftTableJobDouble { real = real, re = re, im = im, ws = ws };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double(ws)", n, stat);
        }

        // ---- DFT helpers ----

        static string DftFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var inRe  = arena.floatVec(n);
            var inIm  = arena.floatVec(n);
            var outRe = arena.floatVec(n);
            var outIm = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
            {
                inRe[i] = rng.NextFloat(-1f, 1f);
                inIm[i] = rng.NextFloat(-1f, 1f);
            }

            var job = new DftJobFloat { inRe = inRe, inIm = inIm, outRe = outRe, outIm = outIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("float", n, stat);
        }

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
                inRe[i] = rng.NextDouble(-1.0, 1.0);
                inIm[i] = rng.NextDouble(-1.0, 1.0);
            }

            var job = new DftJobDouble { inRe = inRe, inIm = inIm, outRe = outRe, outIm = outIm };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.RowTime("double", n, stat);
        }
    }
}
