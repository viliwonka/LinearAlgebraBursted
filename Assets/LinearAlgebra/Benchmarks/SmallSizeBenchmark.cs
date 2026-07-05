using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
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
    // GFLOP/s figure is dominated by fixed overhead and run-to-run noise, so it would be a misleading
    // comparator here (unlike the large-N sections, where GFLOP/s tracks cache behaviour).
    //
    // Deliberately a SEPARATE size list from Bench.Sizes so this section doesn't bloat every other
    // report; per project convention each benchmark file owns its own job structs (see QRBenchmark.cs
    // / QRVariantsBenchmark.cs both defining their own QR-shaped jobs) rather than reaching into
    // another file's types.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallQRJobFloat : IJob
    {
        public floatMxN Q;     // m x n (m >= n); receives Src, overwritten with the orthonormal factor
        public floatMxN R;     // n x n
        public floatMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            QR.decompInPlace(ref Q, ref R);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallQRJobDouble : IJob
    {
        public doubleMxN Q;
        public doubleMxN R;
        public doubleMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            QR.decompInPlace(ref Q, ref R);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallLQJobFloat : IJob
    {
        public floatMxN A;     // m x n (m <= n); not modified by LQ.decomp
        public floatMxN L;     // m x m
        public floatMxN Q;     // m x n

        public void Execute() => LQ.decomp(ref A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallLQJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;
        public doubleMxN Q;

        public void Execute() => LQ.decomp(ref A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallCholJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN L;

        public void Execute() => CHO.decomp(in A, ref L);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallCholJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;

        public void Execute() => CHO.decomp(in A, ref L);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallLUJobFloat : IJob
    {
        public floatMxN U;
        public floatMxN L;
        public floatMxN Src;

        public void Execute()
        {
            int rows = Src.M_Rows;
            var P = new Pivot(rows, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SmallLUJobDouble : IJob
    {
        public doubleMxN U;
        public doubleMxN L;
        public doubleMxN Src;

        public void Execute()
        {
            int rows = Src.M_Rows;
            var P = new Pivot(rows, Allocator.Temp);
            LU.decomp(in Src, ref L, ref U, ref P);
            P.Dispose();
        }
    }

    public static class SmallSizeBenchmark
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

        // ---- QR (square + tall share one path; sized by m x n, m >= n) ----

        static Bench.Stat QRFloat(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.floatMat(m, n);
            var R = arena.floatMat(n, n);
            var Src = arena.floatMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 1000003 + n));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;             // diagonal dominance => full rank, no zero-column early-out

            var job = new SmallQRJobFloat { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        static Bench.Stat QRDouble(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.doubleMat(m, n);
            var R = arena.doubleMat(n, n);
            var Src = arena.doubleMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 1000003 + n));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;

            var job = new SmallQRJobDouble { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        // ---- LQ (square + wide share one path; sized by m x n, m <= n) ----

        static Bench.Stat LQFloat(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var L = arena.floatMat(m, m);
            var Q = arena.floatMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 1000003 + n));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new SmallLQJobFloat { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        static Bench.Stat LQDouble(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 1000003 + n));
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new SmallLQJobDouble { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        // ---- Cholesky (square SPD only) ----

        static Bench.Stat CholFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var L = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    float v = rng.NextFloat(-1f, 1f);
                    A[i, j] = v;
                    A[j, i] = v;                 // symmetric
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;                    // diagonal dominance => SPD

            var job = new SmallCholJobFloat { A = A, L = L };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        static Bench.Stat CholDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var L = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                {
                    double v = rng.NextDouble(-1.0, 1.0);
                    A[i, j] = v;
                    A[j, i] = v;
                }
            for (int d = 0; d < n; d++)
                A[d, d] += n;

            var job = new SmallCholJobDouble { A = A, L = L };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        // ---- LU (square, partial pivoting) ----

        static Bench.Stat LUFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.floatMat(n, n);
            var L = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;                  // diagonal dominance => well-conditioned, full rank

            var job = new SmallLUJobFloat { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }

        static Bench.Stat LUDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var U = arena.doubleMat(n, n);
            var L = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new SmallLUJobDouble { U = U, L = L, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return stat;
        }
    }
}
