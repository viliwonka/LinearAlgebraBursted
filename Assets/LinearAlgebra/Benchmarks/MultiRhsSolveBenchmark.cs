using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;

namespace BULA.Benchmarks
{
    // Multi-RHS solveInPlace (the fused, in-place one-call solvers), END TO END. Given A (n x n) and B
    // (n x k), produce X. Compares the two ways to use the SAME solveInPlace API:
    //   block = solveInPlace(A, B, X)          -- one call: factor ONCE, then solve all k columns.
    //   loop  = for each column: solveInPlace(A, b_i, x_i)  -- one call per RHS: RE-FACTORS every time.
    // The fused solveInPlace fuses factor+solve into one destructive call (for QR/QRCP it also skips
    // reconstructing Q), so there is no "factor once then loop" with this API -- calling it per RHS
    // necessarily re-factors. That k-fold refactorization is exactly what the block overload avoids, so
    // the speedup is dominated by amortizing the O(n^3) factorization and approaches k for the
    // factor-heavy solvers. float, N=512.

    // ---------------------------------------------------------------- LU
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuIpLoop : IJob {
        public floatMxN A; public floatMxN Src; public Pivot P; public floatMxN BXsrc; public floatN col; public int K;
        public void Execute() {
            int n = Src.M_Rows;
            for (int c = 0; c < K; c++) {
                A.Data.CopyFrom(Src.Data);
                for (int i = 0; i < n; i++) col[i] = BXsrc[i, c];
                LU.solveInPlace(ref A, ref P, ref col);   // factor + solve (re-factors each RHS)
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuIpBlock : IJob {
        public floatMxN A; public floatMxN Src; public Pivot P; public floatMxN BX; public floatMxN BXsrc;
        public void Execute() {
            A.Data.CopyFrom(Src.Data);
            BX.Data.CopyFrom(BXsrc.Data);
            LU.solveInPlace(ref A, ref P, ref BX);   // factor once + block solve
        }
    }

    // ---------------------------------------------------------------- Cholesky
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoIpLoop : IJob {
        public floatMxN A; public floatMxN Src; public floatN col; public floatMxN BXsrc; public int K;
        public void Execute() {
            int n = Src.M_Rows;
            for (int c = 0; c < K; c++) {
                A.Data.CopyFrom(Src.Data);
                for (int i = 0; i < n; i++) col[i] = BXsrc[i, c];
                CHO.solveInPlace(ref A, ref col);
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoIpBlock : IJob {
        public floatMxN A; public floatMxN Src; public floatMxN BX; public floatMxN BXsrc;
        public void Execute() {
            A.Data.CopyFrom(Src.Data);
            BX.Data.CopyFrom(BXsrc.Data);
            CHO.solveInPlace(ref A, ref BX);
        }
    }

    // ---------------------------------------------------------------- QR (fused, no Q reconstruction)
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrIpLoop : IJob {
        public floatMxN A; public floatMxN Src; public floatMxN Bsrc; public floatN bcol; public floatN xcol; public int K;
        public void Execute() {
            int m = Src.M_Rows;
            for (int c = 0; c < K; c++) {
                A.Data.CopyFrom(Src.Data);
                for (int i = 0; i < m; i++) bcol[i] = Bsrc[i, c];
                QR.solveInPlace(ref A, ref bcol, ref xcol);
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrIpBlock : IJob {
        public floatMxN A; public floatMxN Src; public floatMxN B; public floatMxN Bsrc; public floatMxN X;
        public void Execute() {
            A.Data.CopyFrom(Src.Data);
            B.Data.CopyFrom(Bsrc.Data);
            QR.solveInPlace(ref A, ref B, ref X);
        }
    }

    // ---------------------------------------------------------------- QRCP (fused, no Q reconstruction)
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpIpLoop : IJob {
        public floatMxN A; public floatMxN Src; public floatMxN R; public Pivot P; public floatN u;
        public floatMxN Bsrc; public floatN bcol; public floatN xcol; public int K;
        public void Execute() {
            int m = Src.M_Rows;
            for (int c = 0; c < K; c++) {
                A.Data.CopyFrom(Src.Data);
                for (int i = 0; i < m; i++) bcol[i] = Bsrc[i, c];
                QRCP.solveInPlace(ref A, ref bcol, ref xcol, ref R, ref P, ref u);
            }
        }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpIpBlock : IJob {
        public floatMxN A; public floatMxN Src; public floatMxN R; public Pivot P; public floatN u;
        public floatMxN B; public floatMxN Bsrc; public floatMxN X;
        public void Execute() {
            A.Data.CopyFrom(Src.Data);
            B.Data.CopyFrom(Bsrc.Data);
            QRCP.solveInPlace(ref A, ref B, ref X, ref R, ref P, ref u, (float)(-1));
        }
    }

    public static class MultiRhsSolveBenchmark
    {
        const int N = 512;
        static readonly int[] Ks = { 16, 64, 256 };

        public static void Run() => Bench.WriteReport("benchmark-multirhs.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Multi-RHS solveInPlace AX=B, END TO END, square N=" + N + ", float ===");
            sb.AppendLine("block = solveInPlace(A,B,X): factor once + solve k columns (QR/QRCP: fused, no Q).");
            sb.AppendLine("loop  = k x solveInPlace(A,b_i,x_i): the same one-call API per RHS -> re-factors each.");
            sb.AppendLine(string.Format("{0,-8} {1,-5} {2,10} {3,10} {4,9}", "solver", "k", "loop(ms)", "block(ms)", "speedup"));

            foreach (int k in Ks) sb.AppendLine(RunLU(k));
            foreach (int k in Ks) sb.AppendLine(RunCHO(k));
            foreach (int k in Ks) sb.AppendLine(RunQR(k));
            foreach (int k in Ks) sb.AppendLine(RunQRCP(k));
            sb.AppendLine();
        }

        static string Row(string solver, int k, Bench.Stat loop, Bench.Stat block)
        {
            double sp = loop.Median / block.Median;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-8} {1,-5} {2,10:F4} {3,10:F4} {4,8:F2}x", solver, k, loop.Median, block.Median, sp);
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
            var Src = new floatMxN(N, N, Allocator.Persistent); FillGen(Src, N, 2654435761u ^ (uint)N);
            var A = new floatMxN(N, N, Allocator.Persistent);
            var P = new Pivot(N, Allocator.Persistent);
            var BXsrc = new floatMxN(N, k, Allocator.Persistent); FillRhs(BXsrc, N, k, 40503u ^ (uint)k);
            var BX = new floatMxN(N, k, Allocator.Persistent);
            var col = new floatN(N, Allocator.Persistent);

            var loop = Bench.Time(() => new LuIpLoop { A = A, Src = Src, P = P, BXsrc = BXsrc, col = col, K = k }.Run());
            var block = Bench.Time(() => new LuIpBlock { A = A, Src = Src, P = P, BX = BX, BXsrc = BXsrc }.Run());

            P.Dispose(); Src.Dispose(); A.Dispose(); BXsrc.Dispose(); BX.Dispose(); col.Dispose();
            return Row("LU", k, loop, block);
        }

        static string RunCHO(int k)
        {
            var Src = new floatMxN(N, N, Allocator.Persistent); FillSpd(Src, N, 2654435761u ^ (uint)N);
            var A = new floatMxN(N, N, Allocator.Persistent);
            var BXsrc = new floatMxN(N, k, Allocator.Persistent); FillRhs(BXsrc, N, k, 40503u ^ (uint)k);
            var BX = new floatMxN(N, k, Allocator.Persistent);
            var col = new floatN(N, Allocator.Persistent);

            var loop = Bench.Time(() => new ChoIpLoop { A = A, Src = Src, col = col, BXsrc = BXsrc, K = k }.Run());
            var block = Bench.Time(() => new ChoIpBlock { A = A, Src = Src, BX = BX, BXsrc = BXsrc }.Run());

            Src.Dispose(); A.Dispose(); BXsrc.Dispose(); BX.Dispose(); col.Dispose();
            return Row("Cholesky", k, loop, block);
        }

        static string RunQR(int k)
        {
            var Src = new floatMxN(N, N, Allocator.Persistent); FillGen(Src, N, 2654435761u ^ (uint)N);
            var A = new floatMxN(N, N, Allocator.Persistent);
            var Bsrc = new floatMxN(N, k, Allocator.Persistent); FillRhs(Bsrc, N, k, 40503u ^ (uint)k);
            var B = new floatMxN(N, k, Allocator.Persistent);
            var X = new floatMxN(N, k, Allocator.Persistent);
            var bcol = new floatN(N, Allocator.Persistent);
            var xcol = new floatN(N, Allocator.Persistent);

            var loop = Bench.Time(() => new QrIpLoop { A = A, Src = Src, Bsrc = Bsrc, bcol = bcol, xcol = xcol, K = k }.Run());
            var block = Bench.Time(() => new QrIpBlock { A = A, Src = Src, B = B, Bsrc = Bsrc, X = X }.Run());

            Src.Dispose(); A.Dispose(); Bsrc.Dispose(); B.Dispose(); X.Dispose(); bcol.Dispose(); xcol.Dispose();
            return Row("QR", k, loop, block);
        }

        static string RunQRCP(int k)
        {
            var Src = new floatMxN(N, N, Allocator.Persistent); FillGen(Src, N, 2654435761u ^ (uint)N);
            var A = new floatMxN(N, N, Allocator.Persistent);
            var R = new floatMxN(N, N, Allocator.Persistent);
            var P = new Pivot(N, Allocator.Persistent);
            var u = new floatN(N, Allocator.Persistent);
            var Bsrc = new floatMxN(N, k, Allocator.Persistent); FillRhs(Bsrc, N, k, 40503u ^ (uint)k);
            var B = new floatMxN(N, k, Allocator.Persistent);
            var X = new floatMxN(N, k, Allocator.Persistent);
            var bcol = new floatN(N, Allocator.Persistent);
            var xcol = new floatN(N, Allocator.Persistent);

            var loop = Bench.Time(() => new QrcpIpLoop { A = A, Src = Src, R = R, P = P, u = u, Bsrc = Bsrc, bcol = bcol, xcol = xcol, K = k }.Run());
            var block = Bench.Time(() => new QrcpIpBlock { A = A, Src = Src, R = R, P = P, u = u, B = B, Bsrc = Bsrc, X = X }.Run());

            P.Dispose(); Src.Dispose(); A.Dispose(); R.Dispose(); u.Dispose(); Bsrc.Dispose(); B.Dispose(); X.Dispose(); bcol.Dispose(); xcol.Dispose();
            return Row("QRCP", k, loop, block);
        }
    }
}
