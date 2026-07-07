using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;

namespace LinearAlgebra.Benchmarks
{
    // Multi-RHS (matrix B/X) block solve vs looping the single-RHS solver. ONE factorization is shared
    // by both (computed once, UNTIMED, in setup) and the timed region is the SOLVE ONLY — so this
    // isolates the level-2 (per-vector substitution / GEMV) -> level-3 (block TRSM / GEMM) jump the
    // matrix overloads buy, independent of the O(n^3) factorization cost. Each right-hand side is a
    // column of an n x k block; the speedup grows with k as the solve moves from bandwidth-bound
    // per-vector work to a cache-reused block. float, square N; double is analogous.

    // ---------------------------------------------------------------- LU
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuMrhsFactor : IJob {
        public floatMxN LU; public floatMxN Src; public Pivot P;
        public void Execute() { LU.Data.CopyFrom(Src.Data); LinearAlgebra.LU.decompInPlace(ref LU, ref P); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuMrhsBlock : IJob {
        public floatMxN LU; public Pivot P; public floatMxN BX; public floatMxN BXsrc;
        public void Execute() { BX.Data.CopyFrom(BXsrc.Data); LinearAlgebra.LU.decompSolve(ref LU, in P, ref BX); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LuMrhsLoop : IJob {
        public floatMxN LU; public Pivot P; public floatMxN BXsrc; public floatN col; public int K;
        public void Execute() {
            int n = LU.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < n; i++) col[i] = BXsrc[i, c];
                LinearAlgebra.LU.decompSolve(ref LU, in P, ref col);
            }
        }
    }

    // ---------------------------------------------------------------- Cholesky
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoMrhsFactor : IJob {
        public floatMxN A; public floatMxN L;
        public void Execute() { CHO.decomp(in A, ref L); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoMrhsBlock : IJob {
        public floatMxN L; public floatMxN BX; public floatMxN BXsrc;
        public void Execute() { BX.Data.CopyFrom(BXsrc.Data); CHO.decompSolve(ref L, ref BX); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct ChoMrhsLoop : IJob {
        public floatMxN L; public floatMxN BXsrc; public floatN col; public int K;
        public void Execute() {
            int n = L.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < n; i++) col[i] = BXsrc[i, c];
                CHO.decompSolve(ref L, ref col);
            }
        }
    }

    // ---------------------------------------------------------------- QR
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrMrhsFactor : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R;
        public void Execute() { Q.Data.CopyFrom(Src.Data); QR.decompInPlace(ref Q, ref R); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrMrhsBlock : IJob {
        public floatMxN Q; public floatMxN R; public floatMxN B; public floatMxN X;
        public void Execute() { QR.decompSolve(ref Q, ref R, ref B, ref X); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrMrhsLoop : IJob {
        public floatMxN Q; public floatMxN R; public floatMxN Bsrc; public floatN bcol; public floatN xcol; public int K;
        public void Execute() {
            int m = Q.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < m; i++) bcol[i] = Bsrc[i, c];
                QR.decompSolve(ref Q, ref R, ref bcol, ref xcol);
            }
        }
    }

    // ---------------------------------------------------------------- QRCP
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpMrhsFactor : IJob {
        public floatMxN Q; public floatMxN Src; public floatMxN R; public Pivot P;
        public void Execute() { Q.Data.CopyFrom(Src.Data); QRCP.decompInPlace(ref Q, ref R, ref P); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpMrhsBlock : IJob {
        public floatMxN Q; public floatMxN R; public Pivot P; public floatMxN B; public floatMxN X;
        public void Execute() { QRCP.decompSolve(ref Q, ref R, in P, ref B, ref X, (float)(-1)); }
    }
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct QrcpMrhsLoop : IJob {
        public floatMxN Q; public floatMxN R; public Pivot P; public floatMxN Bsrc; public floatMxN b1; public floatMxN x1; public int K;
        public void Execute() {
            int m = Q.M_Rows;
            for (int c = 0; c < K; c++) {
                for (int i = 0; i < m; i++) b1[i, 0] = Bsrc[i, c];
                QRCP.decompSolve(ref Q, ref R, in P, ref b1, ref x1, (float)(-1));
            }
        }
    }

    public static class MultiRhsSolveBenchmark
    {
        const int N = 512;
        static readonly int[] Ks = { 16, 64, 256 };

        public static void Run() => Bench.WriteReport("benchmark-multirhs.txt", Section);

        public static void Section(StringBuilder sb)
        {
            sb.AppendLine("=== Multi-RHS solve, square N=" + N + " (SOLVE ONLY; factorization shared/untimed), float ===");
            sb.AppendLine("Block = one matrix-RHS decompSolve over k columns; Loop = k single-RHS decompSolve calls.");
            sb.AppendLine(string.Format("{0,-8} {1,-5} {2,12} {3,12} {4,10}", "solver", "k", "block(ms)", "loop(ms)", "speedup"));

            foreach (int k in Ks) sb.AppendLine(RunLU(k));
            foreach (int k in Ks) sb.AppendLine(RunCHO(k));
            foreach (int k in Ks) sb.AppendLine(RunQR(k));
            foreach (int k in Ks) sb.AppendLine(RunQRCP(k));
            sb.AppendLine();
        }

        static string Row(string solver, int k, Bench.Stat block, Bench.Stat loop)
        {
            double sp = loop.Median / block.Median;
            return string.Format(CultureInfo.InvariantCulture,
                "{0,-8} {1,-5} {2,12:F4} {3,12:F4} {4,9:F2}x", solver, k, block.Median, loop.Median, sp);
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
            var LU = arena.floatMat(N, N);
            var P = new Pivot(N, Allocator.Persistent);
            new LuMrhsFactor { LU = LU, Src = Src, P = P }.Run();   // factor once, untimed

            var BXsrc = arena.floatMat(N, k); FillRhs(BXsrc, N, k, 40503u ^ (uint)k);
            var BX = arena.floatMat(N, k);
            var col = arena.floatVec(N);
            var block = Bench.Time(() => new LuMrhsBlock { LU = LU, P = P, BX = BX, BXsrc = BXsrc }.Run());
            var loop = Bench.Time(() => new LuMrhsLoop { LU = LU, P = P, BXsrc = BXsrc, col = col, K = k }.Run());

            P.Dispose(); arena.Dispose();
            return Row("LU", k, block, loop);
        }

        static string RunCHO(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatMat(N, N); FillSpd(A, N, 2654435761u ^ (uint)N);
            var L = arena.floatMat(N, N);
            new ChoMrhsFactor { A = A, L = L }.Run();

            var BXsrc = arena.floatMat(N, k); FillRhs(BXsrc, N, k, 40503u ^ (uint)k);
            var BX = arena.floatMat(N, k);
            var col = arena.floatVec(N);
            var block = Bench.Time(() => new ChoMrhsBlock { L = L, BX = BX, BXsrc = BXsrc }.Run());
            var loop = Bench.Time(() => new ChoMrhsLoop { L = L, BXsrc = BXsrc, col = col, K = k }.Run());

            arena.Dispose();
            return Row("Cholesky", k, block, loop);
        }

        static string RunQR(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(N, N); FillGen(Src, N, 2654435761u ^ (uint)N);
            var Q = arena.floatMat(N, N);
            var R = arena.floatMat(N, N);
            new QrMrhsFactor { Q = Q, Src = Src, R = R }.Run();

            var Bsrc = arena.floatMat(N, k); FillRhs(Bsrc, N, k, 40503u ^ (uint)k);
            var X = arena.floatMat(N, k);
            var bcol = arena.floatVec(N);
            var xcol = arena.floatVec(N);
            var block = Bench.Time(() => new QrMrhsBlock { Q = Q, R = R, B = Bsrc, X = X }.Run());
            var loop = Bench.Time(() => new QrMrhsLoop { Q = Q, R = R, Bsrc = Bsrc, bcol = bcol, xcol = xcol, K = k }.Run());

            arena.Dispose();
            return Row("QR", k, block, loop);
        }

        static string RunQRCP(int k)
        {
            var arena = new Arena(Allocator.Persistent);
            var Src = arena.floatMat(N, N); FillGen(Src, N, 2654435761u ^ (uint)N);
            var Q = arena.floatMat(N, N);
            var R = arena.floatMat(N, N);
            var P = new Pivot(N, Allocator.Persistent);
            new QrcpMrhsFactor { Q = Q, Src = Src, R = R, P = P }.Run();

            var Bsrc = arena.floatMat(N, k); FillRhs(Bsrc, N, k, 40503u ^ (uint)k);
            var X = arena.floatMat(N, k);
            var b1 = arena.floatMat(N, 1);
            var x1 = arena.floatMat(N, 1);
            var block = Bench.Time(() => new QrcpMrhsBlock { Q = Q, R = R, P = P, B = Bsrc, X = X }.Run());
            var loop = Bench.Time(() => new QrcpMrhsLoop { Q = Q, R = R, P = P, Bsrc = Bsrc, b1 = b1, x1 = x1, K = k }.Run());

            P.Dispose(); arena.Dispose();
            return Row("QRCP", k, block, loop);
        }
    }
}
