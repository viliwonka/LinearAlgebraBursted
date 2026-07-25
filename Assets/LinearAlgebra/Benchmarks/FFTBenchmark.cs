using System.Text;

namespace BULA.Benchmarks
{
    // 1D FFT and DFT benchmarks. Own size arrays (not Bench.Sizes — which is for square matrices):
    //   FFT: twiddle-table workspace, lengths must be power of two, O(N log N).
    //   DFT: direct O(N²) for arbitrary N (smaller sizes since it's quadratic).
    //
    // FFT is IN-PLACE so re/im are destroyed each call; the job copies pristine srcRe/srcIm
    // into re/im at the start of each Execute so every timed sample does identical work.
    // DFT inputs are `in` (not modified), outputs are separate; no copy needed each call.
    //
    // Hand-written harness half. The timed IJobs (FftTable/IfftTable/FftBuildRun/RfftTable/IrfftTable/Dft
    // Job {Float,Double}) and build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/FFTBenchmark.fProxy.cs.
    public static partial class FFTBenchmark
    {
        // Every power of two in range: 4^k sizes hit the pure radix-4 path, 2·4^k the mixed-radix
        // path — so both cores (and the mixed de-interleave) are exercised, no path left unmeasured.
        static readonly int[] FftSizes =
            { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576 };

        // Smaller sizes for the O(N²) direct DFT (quadratic cost limits feasible N).
        static readonly int[] DftSizes = { 256, 512, 1024, 2048 };

        public static void Run() => Bench.WriteReport("benchmark-fft.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== FFT, twiddle-table workspace — auto radix-4/mixed/radix-2 (FFT.fft(ws) / FFT.fft(ws); ms) ===");
            sb.AppendLine("    Workspace built once; dispatches: IsPowerOf4 → radix-4 (log4N passes), 2·4^k → mixed, else → radix-2 table.");
            sb.AppendLine("    No per-element cos/sin; full-circle twiddle table built ONCE outside the timed loop.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(FftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(FftTableDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Complex inverse FFT, twiddle-table workspace (FFT.ifft(ws) / FFT.ifft(ws); ms) ===");
            sb.AppendLine("    Same dispatch as fft(ws); conjugate → forward → conjugate + 1/N scale. Regression guard for the shared cores.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(IfftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(IfftTableDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== FFT, table workspace WITH BUILD INCLUDED (one-shot: build + single transform; build in Burst; ms) ===");
            sb.AppendLine("    Burst job builds the workspace from scratch then runs one fft(ws) — the one-shot cost the reuse rows hide.");
            sb.AppendLine("    Crosses over the no-workspace path after ~1-3 transforms at N>=1024 (table build amortizes fast).");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(FftTableBuiltFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(FftTableBuiltDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Real-input FFT, twiddle-table workspace (FFT.rfft(ws) / FFT.rfft(ws); ms) ===");
            sb.AppendLine("    Workspace built ONCE (arena persistent); no cos/sin in the unpack or butterfly.");
            sb.AppendLine("    Expectation: float no longer slower than double at large N (trig anomaly eliminated).");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(RfftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(RfftTableDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Real inverse FFT, twiddle-table workspace (FFT.irfft(ws) / FFT.irfft(ws); ms) ===");
            sb.AppendLine("    Half-spectrum re/im (N/2+1) input → length-N real output; workspace built ONCE.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(IrfftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(IrfftTableDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Direct DFT (FFT.dft / FFT.dft; O(N^2); ms) ===");
            sb.AppendLine("    Inputs `in` — not modified; output written to separate outRe/outIm.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in DftSizes) sb.AppendLine(DftFloat(n));
            foreach (var n in DftSizes) sb.AppendLine(DftDouble(n));
            sb.AppendLine();
        }
    }
}
