using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // The other Householder paths that share QR's reflector-apply hot loop: column-pivoted QR (QRCP,
    // rank-revealing) and the direct least-squares solve. Each Execute copies a pristine source into
    // the working matrix so every timed sample does identical work.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPJobFloat : IJob
    {
        public floatMxN Q;
        public floatMxN R;
        public floatMxN Src;

        public void Execute()
        {
            int rows = Q.M_Rows, cols = Q.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    Q[r, c] = Src[r, c];

            var P = new Pivot(Q.N_Cols, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPJobDouble : IJob
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

            var P = new Pivot(Q.N_Cols, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRSolveJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN b;
        public floatN bSrc;
        public floatN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;
        public doubleN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    // QRCP.solveInPlace: QRCP-based rank-safe LS solve using the zero-alloc, no-copy primitive — the
    // fused destructive fast path (applies Qᵀ to b during factorization, never reconstructs Q). It
    // DESTROYS both A_to_Q (reflectors + R) and b (overwritten with Qᵀb). Since Bench.Time re-runs
    // Execute() many times, A_to_Q is re-copied from a pristine Src and b from bSrc each Execute
    // (matching the QRSolveJob pattern) so every timed sample factors identical data.
    // R, u are pre-allocated arena scratch. Pivot is per-Execute Temp alloc.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPSolveJobFloat : IJob
    {
        public floatMxN A;     // n x n scratch; receives Src, destroyed by solveInPlace
        public floatMxN Src;   // pristine source, re-copied into A each Execute
        public floatN b;       // m, destroyed (becomes Qᵀb); reset from bSrc each Execute
        public floatN bSrc;    // pristine RHS
        public floatN x;       // n, solution output
        public floatMxN R;     // n x n scratch
        public floatN u;       // m scratch

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            var P = new Pivot(A.N_Cols, Allocator.Temp);
            QRCP.solveInPlace(ref A, ref b, ref x, ref R, ref P, ref u);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;
        public doubleN x;
        public doubleMxN R;
        public doubleN u;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            var P = new Pivot(A.N_Cols, Allocator.Temp);
            QRCP.solveInPlace(ref A, ref b, ref x, ref R, ref P, ref u);
            P.Dispose();
        }
    }

    // QRCP.minNormSolveInPlace: the complete-orthogonal-decomposition (COD / xGELSY) min-norm solve.
    // Structurally identical to QRCPSolveJob (same scratch, same destroys-A-and-b contract), so timing
    // one against the other on the SAME rank-deficient input isolates the COD overhead: when rank r < n
    // it runs a SECOND orthogonal sweep (an LQ-compress of the r x n top block) that basic solveInPlace
    // skips. At full rank the two coincide (COD short-circuits to the basic finish) — hence the
    // rank-deficient input in the builder below.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPMinNormJobFloat : IJob
    {
        public floatMxN A;
        public floatMxN Src;
        public floatN b;
        public floatN bSrc;
        public floatN x;
        public floatMxN R;
        public floatN u;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            var P = new Pivot(A.N_Cols, Allocator.Temp);
            QRCP.minNormSolveInPlace(ref A, ref b, ref x, ref R, ref P, ref u);
            P.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QRCPMinNormJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN bSrc;
        public doubleN x;
        public doubleMxN R;
        public doubleN u;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            for (int i = 0; i < rows; i++)
                b[i] = bSrc[i];

            var P = new Pivot(A.N_Cols, Allocator.Temp);
            QRCP.minNormSolveInPlace(ref A, ref b, ref x, ref R, ref P, ref u);
            P.Dispose();
        }
    }

    public static class QRVariantsBenchmark
    {
        // (4/3) N^3 leading term (approximate). QRCP adds an O(N^3) exact pivot-norm recompute on top,
        // and QR.solveInPlace skips the Q reconstruction, so GFLOP/s here is only a rough comparator —
        // the time columns and the A/B speedup are the honest signal.
        static double Flops(int n) => (4.0 / 3.0) * n * (double)n * n;

        public static void Run() => Bench.WriteReport("benchmark-qrvariants.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== QRCP (column-pivoted, rank-revealing QR; forms Q) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== QR.solveInPlace (Householder least-squares solve; no Q reconstruction) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(SolveFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(SolveDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== QRCP.solveInPlace (QRCP rank-safe LS solve; zero-alloc primitive) ===");
            sb.AppendLine(Bench.Header());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPSolveFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPSolveDouble(n));
            sb.AppendLine();

            // COD overhead: each size emits the basic and the COD row adjacently on the SAME rank-deficient
            // matrix (rank = 3n/4), so the extra second-sweep cost reads straight off the pair. GFLOP/s~
            // uses the plain (4/3)n^3 for both, so COD's throughput reads low (its compress work isn't in
            // the count) — compare the ms columns, not GFLOP/s.
            sb.AppendLine("=== QRCP rank-deficient (n x n, rank = 3n/4): basic solveInPlace vs COD minNormSolveInPlace ===");
            sb.AppendLine(HeaderKernel());
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPRankDefFloat(n));
            foreach (var n in Bench.Sizes) sb.AppendLine(QRCPRankDefDouble(n));
            sb.AppendLine();

            sb.AppendLine("=== TALL overdetermined least squares (m x n, m > n): QR.solveInPlace vs QRCP.solveInPlace ===");
            sb.AppendLine(HeaderTall());
            foreach (var s in TallSizes) sb.AppendLine(SolveTallFloat(s[0], s[1]));
            foreach (var s in TallSizes) sb.AppendLine(SolveTallDouble(s[0], s[1]));
            foreach (var s in TallSizes) sb.AppendLine(QRCPSolveTallFloat(s[0], s[1]));
            foreach (var s in TallSizes) sb.AppendLine(QRCPSolveTallDouble(s[0], s[1]));
            sb.AppendLine();
        }

        // Tall least-squares shapes. The reflector sweep's leading term is 2 n^2 (m - n/3).
        static readonly int[][] TallSizes = { new[] { 2048, 512 }, new[] { 2048, 1024 } };
        static double TallFlops(int m, int n) => 2.0 * n * (double)n * (m - n / 3.0);

        static string HeaderTall()
        {
            return string.Format("{0,-7} {1,-24} {2,11} {3,11} {4,11} {5,11} {6,12}",
                "dtype", "kernel m x n", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~");
        }

        static string RowTall(string dtype, string kernel, int m, int n, Bench.Stat st, double flops)
        {
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-24} {2,11:F4} {3,11:F4} {4,11:F4} {5,11:F4} {6,12:F2}",
                dtype, kernel + " " + m + "x" + n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }

        // Labeled row/header for the rank-deficient basic-vs-COD comparison (adds a kernel column so the
        // two rows per size are distinguishable; N still varies down the block).
        static string HeaderKernel()
        {
            return string.Format("{0,-7} {1,-24} {2,-6} {3,11} {4,11} {5,11} {6,11} {7,12}",
                "dtype", "kernel", "N", "min(ms)", "med(ms)", "mean(ms)", "max(ms)", "GFLOP/s~");
        }

        static string RowKernel(string dtype, string kernel, int n, Bench.Stat st, double flops)
        {
            double gflops = flops / (st.Median / 1000.0) / 1e9;
            return string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-24} {2,-6} {3,11:F4} {4,11:F4} {5,11:F4} {6,11:F4} {7,12:F2}",
                dtype, kernel, n, st.Min, st.Median, st.Mean, st.Max, gflops);
        }

        // Rank-deficient n x n input of exact rank r: fill the first r columns at random, then set each
        // trailing column j>=r to a copy of column j-r. Duplicate columns are a clean, exactly-rank-r
        // structure the rank detector resolves to r < n, which is what makes COD run its second sweep.
        static string QRCPRankDefFloat(int n)
        {
            int rank = (3 * n) / 4;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var b = arena.floatVec(n);
            var bSrc = arena.floatVec(n);
            var x = arena.floatVec(n);
            var R = arena.floatMat(n, n);
            var u = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < rank; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int r = 0; r < n; r++)
                for (int c = rank; c < n; c++)
                    Src[r, c] = Src[r, c - rank];

            var basic = new QRCPSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var sB = Bench.Time(() => basic.Run());
            var cod = new QRCPMinNormJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var sC = Bench.Time(() => cod.Run());

            arena.Dispose();
            return RowKernel("float", "basic solveInPlace", n, sB, Flops(n))
                 + "\n" + RowKernel("float", "COD minNormSolveInPlace", n, sC, Flops(n));
        }

        static string QRCPRankDefDouble(int n)
        {
            int rank = (3 * n) / 4;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var bSrc = arena.doubleVec(n);
            var x = arena.doubleVec(n);
            var R = arena.doubleMat(n, n);
            var u = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < rank; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int r = 0; r < n; r++)
                for (int c = rank; c < n; c++)
                    Src[r, c] = Src[r, c - rank];

            var basic = new QRCPSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var sB = Bench.Time(() => basic.Run());
            var cod = new QRCPMinNormJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var sC = Bench.Time(() => cod.Run());

            arena.Dispose();
            return RowKernel("double", "basic solveInPlace", n, sB, Flops(n))
                 + "\n" + RowKernel("double", "COD minNormSolveInPlace", n, sC, Flops(n));
        }

        static string QRCPFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.floatMat(n, n);
            var R = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPJobFloat { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string QRCPDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.doubleMat(n, n);
            var R = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPJobDouble { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }

        static string SolveFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var b = arena.floatVec(n);
            var bSrc = arena.floatVec(n);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string SolveDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var bSrc = arena.doubleVec(n);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }

        static string QRCPSolveFloat(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(n, n);
            var Src = arena.floatMat(n, n);
            var b = arena.floatVec(n);
            var bSrc = arena.floatVec(n);
            var x = arena.floatVec(n);
            var R = arena.floatMat(n, n);
            var u = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", n, stat, Flops(n));
        }

        static string SolveTallFloat(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var bSrc = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 31 + n));
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return RowTall("float", "QR.solveInPlace", m, n, stat, TallFlops(m, n));
        }

        static string SolveTallDouble(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var Src = arena.doubleMat(m, n);
            var b = arena.doubleVec(m);
            var bSrc = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 31 + n));
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return RowTall("double", "QR.solveInPlace", m, n, stat, TallFlops(m, n));
        }

        static string QRCPSolveTallFloat(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var bSrc = arena.floatVec(m);
            var x = arena.floatVec(n);
            var R = arena.floatMat(n, n);
            var u = arena.floatVec(m);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 31 + n));
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPSolveJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return RowTall("float", "QRCP.solveInPlace", m, n, stat, TallFlops(m, n));
        }

        static string QRCPSolveTallDouble(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var Src = arena.doubleMat(m, n);
            var b = arena.doubleVec(m);
            var bSrc = arena.doubleVec(m);
            var x = arena.doubleVec(n);
            var R = arena.doubleMat(n, n);
            var u = arena.doubleVec(m);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)(m * 31 + n));
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return RowTall("double", "QRCP.solveInPlace", m, n, stat, TallFlops(m, n));
        }

        static string QRCPSolveDouble(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(n, n);
            var Src = arena.doubleMat(n, n);
            var b = arena.doubleVec(n);
            var bSrc = arena.doubleVec(n);
            var x = arena.doubleVec(n);
            var R = arena.doubleMat(n, n);
            var u = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)n);
            for (int r = 0; r < n; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += n;

            var job = new QRCPSolveJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x, R = R, u = u };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", n, stat, Flops(n));
        }
    }
}
