using System.Text;

namespace LinearAlgebra.Benchmarks
{
    // 1D FFT and DFT benchmarks. Own size arrays (not Bench.Sizes — which is for square matrices):
    //   FFT: radix-2 in-place, lengths must be power of two, O(N log N).
    //   DFT: direct O(N²) for arbitrary N (smaller sizes since it's quadratic).
    //
    // FFT is IN-PLACE so re/im are destroyed each call; the job copies pristine srcRe/srcIm
    // into re/im at the start of each Execute so every timed sample does identical work.
    // DFT inputs are `in` (not modified), outputs are separate; no copy needed each call.
    //
    // Hand-written harness half. The timed IJobs (Fft/Rfft/FftTable/FftBuildRun/RfftTable/Dft Job
    // {Float,Double}) and build+measure methods are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/FFTBenchmark.fProxy.cs.
    public static partial class FFTBenchmark
    {
        // Power-of-two sizes for the O(N log N) FFT.
        static readonly int[] FftSizes = { 1024, 4096, 16384, 65536, 262144, 1048576 };

        // Smaller sizes for the O(N²) direct DFT (quadratic cost limits feasible N).
        static readonly int[] DftSizes = { 256, 512, 1024, 2048 };

        public static void Run() => Bench.WriteReport("benchmark-fft.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== No-workspace FFT in-place (FFT.fft / FFT.fft; O(N log N); ms) ===");
            sb.AppendLine("    Auto-dispatch: power-of-4 → radix-4 recurrence (table-free, zero-alloc); else → radix-2 recurrence.");
            sb.AppendLine("    Input destroyed each call; job copies srcRe/srcIm -> re/im before each run.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(FftFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(FftDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== FFT, twiddle-table workspace — auto radix-4/mixed/radix-2 (FFT.fft(ws) / FFT.fft(ws); ms) ===");
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

            sb.AppendLine("=== Real-input half-spectrum FFT (FFT.rfft / FFT.rfft; two-for-one; ms) ===");
            sb.AppendLine("    real input `in` — not modified; re/im output length N/2+1 overwritten each call.");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(RfftFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(RfftDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== Real-input FFT, twiddle-table workspace (FFT.rfft(ws) / FFT.rfft(ws); ms) ===");
            sb.AppendLine("    Workspace built ONCE (arena persistent); no cos/sin in the unpack or butterfly.");
            sb.AppendLine("    Expectation: float no longer slower than double at large N (trig anomaly eliminated).");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in FftSizes) sb.AppendLine(RfftTableFloat(n));
            foreach (var n in FftSizes) sb.AppendLine(RfftTableDouble(n));
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
