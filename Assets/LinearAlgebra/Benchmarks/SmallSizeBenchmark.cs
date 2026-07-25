using System.Globalization;
using System.Text;

namespace BULA.Benchmarks
{
    // Small-size + non-square regression coverage. The blocked (level-3) kernels gate BELOW their
    // crossover onto the ORIGINAL unblocked path: QR blocks at N_Cols >= 64 (QR_BLOCK=32, gate =
    // 2*QR_BLOCK), LQ blocks at M_Rows >= 512 (LQ_BLOCK_MIN_M), Cholesky/LU block at N >= 256
    // (CHOL_BLOCK_MIN_N / LU_BLOCK_MIN_N = 8*32). Below those gates every kernel here runs the exact
    // pre-blocking rank-1 sweep, so small matrices should show NO regression from the blocking work —
    // this section puts that claim on record instead of just asserting it.
    //
    // Square sizes straddle the QR gate (64) and stay well below the LQ/Chol/LU gates. The two
    // non-square subsections (tall QR, wide LQ) cover shapes the square-only sections never exercise;
    // Cholesky/LU stay square-only (SPD / partial-pivot square, per the library's contract).
    //
    // TIME columns only (Bench.HeaderTime/RowTime): at N in [16..128] the work is small enough that a
    // GFLOP/s figure is dominated by fixed overhead and run-to-run noise.
    //
    // Hand-written harness half. The timed IJobs (SmallQR/SmallLQ/SmallChol/SmallLU Job {Float,Double})
    // and build+measure methods (QR/LQ/Chol/LU {Float,Double}) are code-generated per dtype from
    // Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/SmallSizeBenchmark.fProxy.cs.
    public static partial class SmallSizeBenchmark
    {
        // Square sizes straddling the QR blocking gate (64); well below LQ (512) / Cholesky / LU (256).
        static readonly int[] SquareSizes = { 16, 32, 48, 64, 96, 128 };

        // Tall QR shapes (m x n, m > n) - the overdetermined regime the square QR subsection never
        // exercises.
        static readonly int[] TallM = { 64, 128, 128 };
        static readonly int[] TallN = { 32, 32, 64 };

        // Wide LQ shapes (m x n, n > m) - the underdetermined regime the square LQ subsection never
        // exercises.
        static readonly int[] WideM = { 32, 32, 64 };
        static readonly int[] WideN = { 64, 128, 128 };

        public static void Run() => Bench.WriteReport("benchmark-small.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Small square QR (QR.decompInPlace, time = copy-in + factor; blocked path kicks in only at N>=64) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("float", n, QRFloat(n, n)));
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("double", n, QRDouble(n, n)));
            sb.AppendLine();

            sb.AppendLine("=== Tall QR (QR.decompInPlace, m x n, m > n; forms thin Q; spans the N_Cols>=64 gate - n=32 unblocked, n=64 already blocked) ===");
            sb.AppendLine(HeaderTimeShape());
            for (int i = 0; i < TallM.Length; i++) sb.AppendLine(RowTimeShape("float", TallM[i], TallN[i], QRFloat(TallM[i], TallN[i])));
            for (int i = 0; i < TallM.Length; i++) sb.AppendLine(RowTimeShape("double", TallM[i], TallN[i], QRDouble(TallM[i], TallN[i])));
            sb.AppendLine();

            sb.AppendLine("=== Small square LQ (LQ.decomp; blocked path kicks in only at M_Rows>=512) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("float", n, LQFloat(n, n)));
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("double", n, LQDouble(n, n)));
            sb.AppendLine();

            sb.AppendLine("=== Wide LQ (LQ.decomp, m x n, n > m; all far below the M_Rows>=512 gate) ===");
            sb.AppendLine(HeaderTimeShape());
            for (int i = 0; i < WideM.Length; i++) sb.AppendLine(RowTimeShape("float", WideM[i], WideN[i], LQFloat(WideM[i], WideN[i])));
            for (int i = 0; i < WideM.Length; i++) sb.AppendLine(RowTimeShape("double", WideM[i], WideN[i], LQDouble(WideM[i], WideN[i])));
            sb.AppendLine();

            sb.AppendLine("=== Small square Cholesky (CHO.decomp, SPD input; blocked path kicks in only at N>=256) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("float", n, CholFloat(n)));
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("double", n, CholDouble(n)));
            sb.AppendLine();

            sb.AppendLine("=== Small square LU (LU.decomp, partial pivoting; blocked path kicks in only at N>=256) ===");
            sb.AppendLine(Bench.HeaderTime());
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("float", n, LUFloat(n)));
            foreach (var n in SquareSizes) sb.AppendLine(Bench.RowTime("double", n, LUDouble(n)));
            sb.AppendLine();
        }

        // Local variant of Bench.HeaderTime/RowTime with an "m x n" shape column instead of a bare N,
        // since the tall/wide shapes below aren't a fixed-ratio function of one dimension (so a single
        // int column would collide, e.g. both 64x32 and 128x32 share N_Cols=32).
        static string HeaderTimeShape()
        {
            return string.Format("{0,-7} {1,-9} {2,11} {3,11} {4,11} {5,11}",
                "dtype", "shape", "min(ms)", "med(ms)", "mean(ms)", "max(ms)");
        }

        static string RowTimeShape(string dtype, int m, int n, Bench.Stat st)
        {
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-7} {1,-9} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4}",
                dtype, m + "x" + n, st.Min, st.Median, st.Mean, st.Max);
        }
    }
}
