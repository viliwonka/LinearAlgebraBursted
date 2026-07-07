using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Non-square solve paths: the rectangular problems the square LU/Cholesky benchmarks never touch.
    //
    //   TALL  (m = 2n, more equations than unknowns): the OVERDETERMINED least-squares problem
    //         min ||A x - b||.  Householder QR — decompInPlace (forms the thin Q) and the
    //         no-Q-reconstruction direct solve solveInPlace.
    //   WIDE  (n = 2m, more unknowns than equations): the UNDERDETERMINED minimum-norm problem
    //         min ||x|| s.t. A x = b.  LQ — decomp (A = L Q) and minNormSolve
    //         (x = Qᵀ L⁻¹ b).
    //
    // Sized by the SMALLER dimension k (the N column), with a fixed 2:1 aspect: tall is 2k x k,
    // wide is k x 2k. Every Execute copies a pristine source into the working buffers when the kernel
    // destroys its input (QR does; LQ does not), so every timed sample does identical work.
    //
    // All four share the same leading-term flop count — the QR/LQ reflector sweep over a 2k x k panel,
    // QrFlops(2k, k) = (10/3) k^3 — so the GFLOP/s column is directly comparable across them and
    // against the square QR benchmark. Forming Q (QR.decompInPlace / LQ.decomp) does extra work
    // on top, so their GFLOP/s is a lower bound; the solves skip it.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TallQRJobFloat : IJob
    {
        public floatMxN Q;     // m x n; receives A, overwritten with the orthonormal factor
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
    public struct TallQRJobDouble : IJob
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
    public struct TallLSJobFloat : IJob
    {
        public floatMxN A;     // m x n, destroyed (becomes R)
        public floatMxN Src;
        public floatN b;       // length m, destroyed (becomes Qᵀb)
        public floatN bSrc;
        public floatN x;       // length n, solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
            {
                b[r] = bSrc[r];
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            }

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TallLSJobDouble : IJob
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
            {
                b[r] = bSrc[r];
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];
            }

            QR.solveInPlace(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQJobFloat : IJob
    {
        public floatMxN A;     // m x n (n >= m); not modified by LQ.decomp
        public floatMxN L;     // m x m
        public floatMxN Q;     // m x n

        public void Execute() => LQ.decomp(in A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;
        public doubleMxN Q;

        public void Execute() => LQ.decomp(in A, ref L, ref Q);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideMinNormJobFloat : IJob
    {
        public floatMxN A;     // m x n (n >= m); not modified
        public floatN b;       // length m; not modified (copied internally)
        public floatN x;       // length n, min-norm solution

        public void Execute() => LQ.minNormSolve(ref A, ref b, ref x);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideMinNormJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b;
        public doubleN x;

        public void Execute() => LQ.minNormSolve(ref A, ref b, ref x);
    }

    // LQRP (row-pivoted, rank-revealing LQ) at the same wide shapes as the LQ jobs above, so the two
    // sit side by side in the report. LQRP.decomp does the same reflector sweep as LQ.decomp PLUS row
    // pivoting with downdated partial norms; it runs the UNBLOCKED core at every size, while LQ.decomp
    // switches to the blocked (level-3) core above Consts.fProxyLqBlockMinM. So the LQRP-vs-LQ gap
    // below that gate is the pure pivot+downdate overhead, and the widening gap above it is the
    // headroom a future blocked LQRP core would recover.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPJobFloat : IJob
    {
        public floatMxN A;     // m x n (n >= m); not modified by LQRP.decomp
        public floatMxN L;     // m x m
        public floatMxN Q;     // m x n
        public Pivot P;        // size m

        public void Execute() => LQRP.decomp(in A, ref L, ref Q, ref P);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN L;
        public doubleMxN Q;
        public Pivot P;

        public void Execute() => LQRP.decomp(in A, ref L, ref Q, ref P);
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPSolveJobFloat : IJob
    {
        public floatMxN A;     // m x n; DESTROYED by solveInPlace (restored from Src each sample)
        public floatMxN Src;
        public floatN b;       // length m; NOT modified by LQRP.solveInPlace (read-only)
        public floatN x;       // length n, basic solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQRP.solveInPlace(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPSolveJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQRP.solveInPlace(ref A, ref b, ref x);
        }
    }

    // LQRP.minNormSolveInPlace: the complete-orthogonal-decomposition (COD) min-norm solve, sharing
    // WideLQRPSolveJob's shape and destroys-A / b-read-only contract. On a rank-deficient wide system
    // (r < m) it does strictly more than the basic solve: after the row-pivoted factor it runs a second
    // orthogonal stage (a least-squares solve against the first-r columns of L, the K-block) so that the
    // returned x is the true minimum-norm least-squares solution rather than the basic one. At full row
    // rank the basic solve already IS min-norm, so the builder below uses a rank-deficient input.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPMinNormJobFloat : IJob
    {
        public floatMxN A;     // m x n; DESTROYED by minNormSolveInPlace (restored from Src each sample)
        public floatMxN Src;
        public floatN b;       // length m; NOT modified (b is preserved)
        public floatN x;       // length n, min-norm solution

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQRP.minNormSolveInPlace(ref A, ref b, ref x);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WideLQRPMinNormJobDouble : IJob
    {
        public doubleMxN A;
        public doubleMxN Src;
        public doubleN b;
        public doubleN x;

        public void Execute()
        {
            int rows = A.M_Rows, cols = A.N_Cols;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    A[r, c] = Src[r, c];

            LQRP.minNormSolveInPlace(ref A, ref b, ref x);
        }
    }

    public static class TallWideSolveBenchmark
    {
        // Householder QR reflector-sweep leading term for a rows x cols panel (rows >= cols):
        // 2 cols^2 (rows - cols/3). At rows = cols this is the (4/3)n^3 of the square QR benchmark.
        // For every section here the controlling panel is 2k x k, so QrFlops(2k, k) = (10/3) k^3.
        static double QrFlops(int rows, int cols) => 2.0 * cols * (double)cols * (rows - cols / 3.0);

        public static void Run() => Bench.WriteReport("benchmark-tallwide.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Tall QR factorization (decompInPlace, A is 2k x k; forms thin Q); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(TallQRFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(TallQRDouble(k));
            sb.AppendLine();

            sb.AppendLine("=== Overdetermined least squares (solveInPlace, A is 2k x k; no Q reconstruction); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(TallLSFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(TallLSDouble(k));
            sb.AppendLine();

            sb.AppendLine("=== Wide LQ factorization (decomp, A is k x 2k); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQDouble(k));
            sb.AppendLine();

            sb.AppendLine("=== Underdetermined minimum-norm (minNormSolve, A is k x 2k); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideMinNormFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideMinNormDouble(k));
            sb.AppendLine();

            sb.AppendLine("=== Wide LQRP row-pivoted factorization (decomp, A is k x 2k; forms L,Q,P; UNBLOCKED); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPDecompFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPDecompDouble(k));
            sb.AppendLine();

            sb.AppendLine("=== Underdetermined rank-safe basic solve (LQRP.solveInPlace, A is k x 2k; no Q); N column = k ===");
            sb.AppendLine(Bench.Header());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPSolveFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPSolveDouble(k));
            sb.AppendLine();

            // COD overhead on a rank-deficient wide system: each size emits the basic and the COD row
            // adjacently on the SAME rank-deficient matrix (rank = 3k/4 < k = m), so the extra second-stage
            // cost reads straight off the pair. Both call the allocating convenience overload, so the only
            // measured delta is algorithm. GFLOP/s~ uses the plain factor flop count for both, so COD reads
            // low (its second-stage work isn't in the count) — compare the ms columns, not GFLOP/s.
            sb.AppendLine("=== LQRP rank-deficient (k x 2k, rank = 3k/4): basic solveInPlace vs COD minNormSolveInPlace; N column = k ===");
            sb.AppendLine(HeaderKernel());
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPRankDefFloat(k));
            foreach (var k in Bench.Sizes) sb.AppendLine(WideLQRPRankDefDouble(k));
            sb.AppendLine();
        }

        // Labeled row/header for the rank-deficient basic-vs-COD comparison (kernel column added so the
        // two rows per size are distinguishable; N = k still varies down the block).
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

        // Rank-deficient k x 2k input of exact rank r: fill the first r rows at random, then set each
        // trailing row i>=r to a copy of row i-r. Duplicate rows are a clean, exactly-rank-r structure the
        // row-pivoted rank detector resolves to r < m, which is what makes COD run its second stage.
        static string WideLQRPRankDefFloat(int k)
        {
            int m = k, n = 2 * k, rank = (3 * k) / 4;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < rank; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int r = rank; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = Src[r - rank, c];
            for (int r = 0; r < m; r++)
                b[r] = rng.NextFloat(-1f, 1f);

            var basic = new WideLQRPSolveJobFloat { A = A, Src = Src, b = b, x = x };
            var sB = Bench.Time(() => basic.Run());
            var cod = new WideLQRPMinNormJobFloat { A = A, Src = Src, b = b, x = x };
            var sC = Bench.Time(() => cod.Run());

            arena.Dispose();
            return RowKernel("float", "basic solveInPlace", k, sB, QrFlops(n, m))
                 + "\n" + RowKernel("float", "COD minNormSolveInPlace", k, sC, QrFlops(n, m));
        }

        static string WideLQRPRankDefDouble(int k)
        {
            int m = k, n = 2 * k, rank = (3 * k) / 4;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var Src = arena.doubleMat(m, n);
            var b = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < rank; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int r = rank; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = Src[r - rank, c];
            for (int r = 0; r < m; r++)
                b[r] = rng.NextDouble(-1.0, 1.0);

            var basic = new WideLQRPSolveJobDouble { A = A, Src = Src, b = b, x = x };
            var sB = Bench.Time(() => basic.Run());
            var cod = new WideLQRPMinNormJobDouble { A = A, Src = Src, b = b, x = x };
            var sC = Bench.Time(() => cod.Run());

            arena.Dispose();
            return RowKernel("double", "basic solveInPlace", k, sB, QrFlops(n, m))
                 + "\n" + RowKernel("double", "COD minNormSolveInPlace", k, sC, QrFlops(n, m));
        }

        // ---- Tall QR factorization (overdetermined: 2k x k) ----

        static string TallQRFloat(int k)
        {
            int m = 2 * k, n = k;
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.floatMat(m, n);
            var R = arena.floatMat(n, n);
            var Src = arena.floatMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;             // full column rank, no zero-column early-out

            var job = new TallQRJobFloat { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, QrFlops(m, n));
        }

        static string TallQRDouble(int k)
        {
            int m = 2 * k, n = k;
            var arena = new Arena(Allocator.Persistent);
            var Q = arena.doubleMat(m, n);
            var R = arena.doubleMat(n, n);
            var Src = arena.doubleMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;

            var job = new TallQRJobDouble { Q = Q, R = R, Src = Src };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", k, stat, QrFlops(m, n));
        }

        // ---- Overdetermined least squares (2k x k) ----

        static string TallLSFloat(int k)
        {
            int m = 2 * k, n = k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var bSrc = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;

            var job = new TallLSJobFloat { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, QrFlops(m, n));
        }

        static string TallLSDouble(int k)
        {
            int m = 2 * k, n = k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var Src = arena.doubleMat(m, n);
            var b = arena.doubleVec(m);
            var bSrc = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                bSrc[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < n; d++)
                Src[d, d] += m + n;

            var job = new TallLSJobDouble { A = A, Src = Src, b = b, bSrc = bSrc, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", k, stat, QrFlops(m, n));
        }

        // ---- Wide LQ factorization (underdetermined: k x 2k) ----

        static string WideLQFloat(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var L = arena.floatMat(m, m);
            var Q = arena.floatMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new WideLQJobFloat { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, QrFlops(n, m));
        }

        static string WideLQDouble(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new WideLQJobDouble { A = A, L = L, Q = Q };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", k, stat, QrFlops(n, m));
        }

        // ---- Underdetermined minimum-norm (k x 2k) ----

        static string WideMinNormFloat(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new WideMinNormJobFloat { A = A, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, QrFlops(n, m));
        }

        static string WideMinNormDouble(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var b = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new WideMinNormJobDouble { A = A, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", k, stat, QrFlops(n, m));
        }

        // ---- Wide LQRP row-pivoted factorization (k x 2k) ----

        static string WideLQRPDecompFloat(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var L = arena.floatMat(m, m);
            var Q = arena.floatMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;               // full row rank

            var job = new WideLQRPJobFloat { A = A, L = L, Q = Q, P = P };
            var stat = Bench.Time(() => job.Run());

            P.Dispose();
            arena.Dispose();
            return Bench.Row("float", k, stat, QrFlops(n, m));
        }

        static string WideLQRPDecompDouble(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = rng.NextDouble(-1.0, 1.0);
            for (int d = 0; d < m; d++)
                A[d, d] += m + n;

            var job = new WideLQRPJobDouble { A = A, L = L, Q = Q, P = P };
            var stat = Bench.Time(() => job.Run());

            P.Dispose();
            arena.Dispose();
            return Bench.Row("double", k, stat, QrFlops(n, m));
        }

        // ---- Underdetermined rank-safe basic solve (k x 2k) ----

        static string WideLQRPSolveFloat(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(m, n);
            var Src = arena.floatMat(m, n);
            var b = arena.floatVec(m);
            var x = arena.floatVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextFloat(-1f, 1f);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextFloat(-1f, 1f);
            }
            for (int d = 0; d < m; d++)
                Src[d, d] += m + n;

            var job = new WideLQRPSolveJobFloat { A = A, Src = Src, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("float", k, stat, QrFlops(n, m));
        }

        static string WideLQRPSolveDouble(int k)
        {
            int m = k, n = 2 * k;
            var arena = new Arena(Allocator.Persistent);
            var A = arena.doubleMat(m, n);
            var Src = arena.doubleMat(m, n);
            var b = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)k);
            for (int r = 0; r < m; r++)
            {
                b[r] = rng.NextDouble(-1.0, 1.0);
                for (int c = 0; c < n; c++)
                    Src[r, c] = rng.NextDouble(-1.0, 1.0);
            }
            for (int d = 0; d < m; d++)
                Src[d, d] += m + n;

            var job = new WideLQRPSolveJobDouble { A = A, Src = Src, b = b, x = x };
            var stat = Bench.Time(() => job.Run());

            arena.Dispose();
            return Bench.Row("double", k, stat, QrFlops(n, m));
        }
    }
}
