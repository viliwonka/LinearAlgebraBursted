using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Multi-RHS solve, END TO END: given A (n x n) and B (n x k), produce X. The whole operation is
    // timed, INCLUDING the O(n^3) factorization that dominates the O(n^2 k) solve. Two competent
    // approaches are compared (both factor ONCE):
    //   loop  = factor once, then k single-RHS solves.
    //   block = factor once, then one matrix-RHS solve over all k columns.
    // The `factor` column is the factorization alone, so you can see how much of each total is just the
    // shared factorization. speedup = loop / block = the real whole-operation speedup: ~1x when the
    // factorization dominates (small k), growing as k makes the solve a bigger slice. (A naive
    // "re-factor per RHS" loop, not shown, would instead cost ~k * factor.) float, N=512.

    // ---------------------------------------------------------------- LU
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuE2EFactor : IJob {
        public floatMxN A; public floatMxN Src; public Pivot P;
        public void Execute() { A.Data.CopyFrom(Src.Data); LU.decompInPlace(ref A, ref P); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuE2ELoop : IJob {
        public floatMxN A; public floatMxN Src; public Pivot P; public floatMxN BXsrc; public floatN col; public int K;
        public void Execute() {
            A.Data.CopyFrom(Src.Data);
            LU.decompInPlace(ref A, ref P);
            int n = A.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < n; i++) col[i] = BXsrc[i, c];
                LU.decompSolve(ref A, in P, ref col);
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuE2EBlock : IJob {
        public floatMxN A; public floatMxN Src; public Pivot P; public floatMxN BX; public floatMxN BXsrc;
        public void Execute() {
            A.Data.CopyFrom(Src.Data);
            LU.decompInPlace(ref A, ref P);
            BX.Data.CopyFrom(BXsrc.Data);
            LU.decompSolve(ref A, in P, ref BX);
        }
    }

    // ---------------------------------------------------------------- Cholesky
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoE2EFactor : IJob {
        public floatMxN A; public floatMxN L;
        public void Execute() { CHO.decomp(in A, ref L); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoE2ELoop : IJob {
        public floatMxN A; public floatMxN L; public floatMxN BXsrc; public floatN col; public int K;
        public void Execute() {
            CHO.decomp(in A, ref L);
            int n = L.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < n; i++) col[i] = BXsrc[i, c];
                CHO.decompSolve(ref L, ref col);
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoE2EBlock : IJob {
        public floatMxN A; public floatMxN L; public floatMxN BX; public floatMxN BXsrc;
        public void Execute() {
            CHO.decomp(in A, ref L);
            BX.Data.CopyFrom(BXsrc.Data);
            CHO.decompSolve(ref L, ref BX);
        }
    }

    // ---------------------------------------------------------------- QR
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrE2EFactor : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R;
        public void Execute() { Q.Data.CopyFrom(Src.Data); QR.decompInPlace(ref Q, ref R); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrE2ELoop : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R; public floatMxN Bsrc; public floatN bcol; public floatN xcol; public int K;
        public void Execute() {
            Q.Data.CopyFrom(Src.Data);
            QR.decompInPlace(ref Q, ref R);
            int m = Q.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < m; i++) bcol[i] = Bsrc[i, c];
                QR.decompSolve(ref Q, ref R, ref bcol, ref xcol);
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrE2EBlock : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R; public floatMxN B; public floatMxN X;
        public void Execute() {
            Q.Data.CopyFrom(Src.Data);
            QR.decompInPlace(ref Q, ref R);
            QR.decompSolve(ref Q, ref R, ref B, ref X);
        }
    }

    // ---------------------------------------------------------------- QRCP
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpE2EFactor : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R; public Pivot P;
        public void Execute() { Q.Data.CopyFrom(Src.Data); QRCP.decompInPlace(ref Q, ref R, ref P); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpE2ELoop : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R; public Pivot P; public floatMxN Bsrc; public floatMxN b1; public floatMxN x1; public int K;
        public void Execute() {
            Q.Data.CopyFrom(Src.Data);
            QRCP.decompInPlace(ref Q, ref R, ref P);
            int m = Q.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < m; i++) b1[i, 0] = Bsrc[i, c];
                QRCP.decompSolve(ref Q, ref R, in P, ref b1, ref x1, (float)(-1));
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpE2EBlock : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R; public Pivot P; public floatMxN B; public floatMxN X;
        public void Execute() {
            Q.Data.CopyFrom(Src.Data);
            QRCP.decompInPlace(ref Q, ref R, ref P);
            QRCP.decompSolve(ref Q, ref R, in P, ref B, ref X, (float)(-1));
        }
    }

    public static class MultiRhsSolveBenchmark
    {
        const int N = 512;
        static readonly int[] Ks = { 16, 64, 256 };

        public static void Run() => Bench.WriteReport("benchmark-multirhs.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Multi-RHS solve AX=B, END TO END (factor + solve), square N=" + N + ", float ===");
            sb.AppendLine("loop = factor once + k single-RHS solves; block = factor once + one k-column solve.");
            sb.AppendLine("factor = the shared factorization alone (dominates for small k). speedup = loop/block.");
            sb.AppendLine(string.Format("{0,-8} {1,-5} {2,10} {3,10} {4,10} {5,9}",
                "solver", "k", "factor(ms)", "loop(ms)", "block(ms)", "speedup"));

            foreach (int k in Ks) sb.AppendLine(RunLU(k));
            foreach (int k in Ks) sb.AppendLine(RunCHO(k));
            foreach (int k in Ks) sb.AppendLine(RunQR(k));
            foreach (int k in Ks) sb.AppendLine(RunQRCP(k));
            sb.AppendLine();
        }

        static string Row(string solver, int k, Bench.Stat factor, Bench.Stat loop, Bench.Stat block)
        {
            double sp = loop.Median / block.Median;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-8} {1,-5} {2,10:F4} {3,10:F4} {4,10:F4} {5,8:F2}x",
                solver, k, factor.Median, loop.Median, block.Median, sp);
        }

        static void FillSpd(floatMxN A, int n, uint seed) {
            var rng = new Random(seed);
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) { float v = rng.NextFloat(-1f, 1f); A[i, j] = v; A[j, i] = v; }
            for (int d = 0; d < n; d++) A[d, d] += n;
        }
        static void FillGen(floatMxN A, int n, uint seed) {
            var rng = new Random(seed);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++) A[r, c] = rng.NextFloat(-1f, 1f);
            for (int d = 0; d < n; d++) A[d, d] += n;
        }
        static void FillRhs(floatMxN B, int n, int k, uint seed) {
            var rng = new Random(seed);
            for (int i = 0; i < n; i++) for (int c = 0; c < k; c++) B[i, c] = rng.NextFloat(-1f, 1f);
        }

        static string RunLU(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(N, N); FillGen(Src, N, 2654435761u ^ (uint)N);
            var A = arena.floatMat(N, N);
            var P = new Pivot(N, Allocator.Persistent);
            var BXsrc = arena.floatMat(N, k); FillRhs(BXsrc, N, k, 40503u ^ (uint)k);
            var BX = arena.floatMat(N, k);
            var col = arena.floatVec(N);

            var factor = Bench.Time(() => new LuE2EFactor { A = A, Src = Src, P = P }.Run());
            var loop = Bench.Time(() => new LuE2ELoop { A = A, Src = Src, P = P, BXsrc = BXsrc, col = col, K = k }.Run());
            var block = Bench.Time(() => new LuE2EBlock { A = A, Src = Src, P = P, BX = BX, BXsrc = BXsrc }.Run());

            P.Dispose(); arena.Dispose();
            return Row("LU", k, factor, loop, block);
        }

        static string RunCHO(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(N, N); FillSpd(A, N, 2654435761u ^ (uint)N);
            var L = arena.floatMat(N, N);
            var BXsrc = arena.floatMat(N, k); FillRhs(BXsrc, N, k, 40503u ^ (uint)k);
            var BX = arena.floatMat(N, k);
            var col = arena.floatVec(N);

            var factor = Bench.Time(() => new ChoE2EFactor { A = A, L = L }.Run());
            var loop = Bench.Time(() => new ChoE2ELoop { A = A, L = L, BXsrc = BXsrc, col = col, K = k }.Run());
            var block = Bench.Time(() => new ChoE2EBlock { A = A, L = L, BX = BX, BXsrc = BXsrc }.Run());

            arena.Dispose();
            return Row("Cholesky", k, factor, loop, block);
        }

        static string RunQR(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(N, N); FillGen(Src, N, 2654435761u ^ (uint)N);
            var Q = arena.floatMat(N, N);
            var R = arena.floatMat(N, N);
            var Bsrc = arena.floatMat(N, k); FillRhs(Bsrc, N, k, 40503u ^ (uint)k);
            var X = arena.floatMat(N, k);
            var bcol = arena.floatVec(N);
            var xcol = arena.floatVec(N);

            var factor = Bench.Time(() => new QrE2EFactor { Q = Q, Src = Src, R = R }.Run());
            var loop = Bench.Time(() => new QrE2ELoop { Q = Q, Src = Src, R = R, Bsrc = Bsrc, bcol = bcol, xcol = xcol, K = k }.Run());
            var block = Bench.Time(() => new QrE2EBlock { Q = Q, Src = Src, R = R, B = Bsrc, X = X }.Run());

            arena.Dispose();
            return Row("QR", k, factor, loop, block);
        }

        static string RunQRCP(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(N, N); FillGen(Src, N, 2654435761u ^ (uint)N);
            var Q = arena.floatMat(N, N);
            var R = arena.floatMat(N, N);
            var P = new Pivot(N, Allocator.Persistent);
            var Bsrc = arena.floatMat(N, k); FillRhs(Bsrc, N, k, 40503u ^ (uint)k);
            var X = arena.floatMat(N, k);
            var b1 = arena.floatMat(N, 1);
            var x1 = arena.floatMat(N, 1);

            var factor = Bench.Time(() => new QrcpE2EFactor { Q = Q, Src = Src, R = R, P = P }.Run());
            var loop = Bench.Time(() => new QrcpE2ELoop { Q = Q, Src = Src, R = R, P = P, Bsrc = Bsrc, b1 = b1, x1 = x1, K = k }.Run());
            var block = Bench.Time(() => new QrcpE2EBlock { Q = Q, Src = Src, R = R, P = P, B = Bsrc, X = X }.Run());

            P.Dispose(); arena.Dispose();
            return Row("QRCP", k, factor, loop, block);
        }
    }
}
