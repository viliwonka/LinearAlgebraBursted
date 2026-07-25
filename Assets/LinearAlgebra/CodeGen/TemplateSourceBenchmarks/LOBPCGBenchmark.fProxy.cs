using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;
using BULA.Gallery;
using BULA.Sparse;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of LOBPCGBenchmark: the timed IJob plus the build+measure method.
    // The dtype-agnostic harness (constants, Run, Section) lives in the hand-written partial in
    // Assets/LinearAlgebra/Benchmarks/LOBPCGBenchmark.cs. See that file for what this benchmark measures.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgJobFProxy : IJob
    {
        public fProxyMxN A;      // SPD, not modified (dense forwarder copies internally as needed)
        public fProxyLOBPCGCache ws;
        public int k, maxIter;
        public fProxy tol;
        public NativeArray<LOBPCGInfo> infoOut; // length 1

        public void Execute()
        {
            // Cold start every timed run (same as the BSR jobs below): an all-zero X makes lobpcg
            // re-seed deterministically instead of warm-starting from the previous sample's
            // converged block and timing a no-op.
            for (int i = 0; i < ws.X.M_Rows; i++)
                for (int c = 0; c < ws.X.N_Cols; c++)
                    ws.X[i, c] = (fProxy)0;
            infoOut[0] = Eigen.lobpcg(in A, ref ws, k, tol, maxIter);
        }
    }


    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgBsrNoneJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyLOBPCGCache ws;
        public int k, maxIter;
        public fProxy tol;
        public NativeArray<LOBPCGInfo> infoOut;
        public void Execute()
        {
            // Cold start every timed run: an all-zero X makes lobpcg re-seed deterministically
            // (otherwise the reused workspace warm-starts already-converged and times a no-op).
            for (int i = 0; i < ws.X.M_Rows; i++)
                for (int c = 0; c < ws.X.N_Cols; c++)
                    ws.X[i, c] = (fProxy)0;
            infoOut[0] = Eigen.lobpcg(new fProxyBSROperator(in A), ref ws, k, tol, maxIter);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgBsrJacobiJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyBlockJacobi M;
        public fProxyLOBPCGCache ws;
        public int k, maxIter;
        public fProxy tol;
        public NativeArray<LOBPCGInfo> infoOut;
        public void Execute()
        {
            // Cold start every timed run: an all-zero X makes lobpcg re-seed deterministically
            // (otherwise the reused workspace warm-starts already-converged and times a no-op).
            for (int i = 0; i < ws.X.M_Rows; i++)
                for (int c = 0; c < ws.X.N_Cols; c++)
                    ws.X[i, c] = (fProxy)0;
            infoOut[0] = Eigen.lobpcg(new fProxyBSROperator(in A), in M, ref ws, k, tol, maxIter);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgBsrSsorJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxySSOR M;
        public fProxyLOBPCGCache ws;
        public int k, maxIter;
        public fProxy tol;
        public NativeArray<LOBPCGInfo> infoOut;
        public void Execute()
        {
            // Cold start every timed run: an all-zero X makes lobpcg re-seed deterministically
            // (otherwise the reused workspace warm-starts already-converged and times a no-op).
            for (int i = 0; i < ws.X.M_Rows; i++)
                for (int c = 0; c < ws.X.N_Cols; c++)
                    ws.X[i, c] = (fProxy)0;
            infoOut[0] = Eigen.lobpcg(new fProxyBSROperator(in A), in M, ref ws, k, tol, maxIter);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LobpcgBsrIc0JobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyIC0 M;
        public fProxyLOBPCGCache ws;
        public int k, maxIter;
        public fProxy tol;
        public NativeArray<LOBPCGInfo> infoOut;
        public void Execute()
        {
            // Cold start every timed run: an all-zero X makes lobpcg re-seed deterministically
            // (otherwise the reused workspace warm-starts already-converged and times a no-op).
            for (int i = 0; i < ws.X.M_Rows; i++)
                for (int c = 0; c < ws.X.N_Cols; c++)
                    ws.X[i, c] = (fProxy)0;
            infoOut[0] = Eigen.lobpcg(new fProxyBSROperator(in A), in M, ref ws, k, tol, maxIter);
        }
    }

    public static partial class LOBPCGBenchmark
    {
        // Preconditioner face-off for the k smallest eigenpairs over sparse BSR systems:
        // solve-to-tolerance wall-clock + iteration count, none/Jacobi/SSOR/IC0.
        static string BenchSparsePrecondFProxy(bool laplacian, int p1, int p2, float density, uint seed, int k)
        {
            const string fmt = "{0,-7} {1,-6} {2,-12} {3,11:F4} {4,11:F4} {5,7} {6,10} {7,14:E3}";
            var A = laplacian ? fProxyGallery.fProxyLaplacian2D(p1, p2, Allocator.Persistent)
                              : fProxyGallery.fProxyRandomSparseSPD(p1, p2, (fProxy)density, seed, Allocator.Persistent);
            int n = A.M_Rows;
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 500;
            var ws = new fProxyLOBPCGCache(n, k, Allocator.Persistent);
            var infoOut = new NativeArray<LOBPCGInfo>(1, Allocator.Persistent);
            var sb = new StringBuilder();

            var jN = new LobpcgBsrNoneJobFProxy { A = A, ws = ws, k = k, maxIter = maxIter, tol = tol, infoOut = infoOut };
            var sN = Bench.Time(() => jN.Run());
            var iN = infoOut[0];
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "none", sN.Median, sN.Min, iN.iterations, iN.converged, iN.maxResidual));

            var mJ = new fProxyBlockJacobi(in A, Allocator.Persistent);
            var jJ = new LobpcgBsrJacobiJobFProxy { A = A, M = mJ, ws = ws, k = k, maxIter = maxIter, tol = tol, infoOut = infoOut };
            var sJ = Bench.Time(() => jJ.Run());
            var iJ = infoOut[0];
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "Jacobi", sJ.Median, sJ.Min, iJ.iterations, iJ.converged, iJ.maxResidual));

            var mS = new fProxySSOR(in A, Allocator.Persistent);
            var jS = new LobpcgBsrSsorJobFProxy { A = A, M = mS, ws = ws, k = k, maxIter = maxIter, tol = tol, infoOut = infoOut };
            var sS = Bench.Time(() => jS.Run());
            var iS = infoOut[0];
            sb.AppendLine(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "SSOR", sS.Median, sS.Min, iS.iterations, iS.converged, iS.maxResidual));

            var mI = new fProxyIC0(in A, Allocator.Persistent);
            var jI = new LobpcgBsrIc0JobFProxy { A = A, M = mI, ws = ws, k = k, maxIter = maxIter, tol = tol, infoOut = infoOut };
            var sI = Bench.Time(() => jI.Run());
            var iI = infoOut[0];
            sb.Append(string.Format(System.Globalization.CultureInfo.InvariantCulture, fmt,
                "fProxy", n, "IC0", sI.Median, sI.Min, iI.iterations, iI.converged, iI.maxResidual));

            infoOut.Dispose();
            A.Dispose(); ws.Dispose(); mJ.Dispose(); mS.Dispose(); mI.Dispose();
            return sb.ToString();
        }

        static string BenchFProxy(int N, int K, int maxIter)
        {
            var M = new fProxyMxN(N, N, Allocator.Persistent);
            var A = new fProxyMxN(N, N, Allocator.Persistent);

            var rng = new Unity.Mathematics.Random(2654435761u ^ (uint)N);
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    M[r, c] = rng.NextFProxy(-1f, 1f);
            Blas.dot(in M, in M, ref A, true);
            for (int d = 0; d < N; d++) A[d, d] += (fProxy)1;

            var ws = new fProxyLOBPCGCache(N, K, Allocator.Persistent);
            var infoOut = new NativeArray<LOBPCGInfo>(1, Allocator.Persistent);
            var job = new LobpcgJobFProxy { A = A, ws = ws, k = K, maxIter = maxIter, tol = (fProxy)1e-20, infoOut = infoOut };
            var stat = Bench.Time(() => job.Run());

            var info = infoOut[0];
            string row = string.Format(System.Globalization.CultureInfo.InvariantCulture,
                "{0,-7} {1,-6} {2,11:F4} {3,11:F4} {4,10} {5,10} {6,14:E3}",
                "fProxy", N, stat.Min, stat.Median, info.iterations, info.converged, info.maxResidual);

            infoOut.Dispose();
            M.Dispose(); A.Dispose(); ws.Dispose();
            return row;
        }
    }
}
