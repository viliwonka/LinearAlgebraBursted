using System;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        static void RequireSvdTruncatedWorkspace(in floatSvdTruncatedWorkspace ws, int m, int n, int p, string who)
        {
            bool ok =
                ws.UL.M_Rows == m && ws.UL.N_Cols == p &&
                ws.VL.M_Rows == n && ws.VL.N_Cols == p + 1 &&
                ws.B.M_Rows == p && ws.B.N_Cols == p &&
                ws.BsvdWs.U.M_Rows == p && ws.BsvdWs.U.N_Cols == p &&
                ws.BsvdWs.S.N == p &&
                ws.BsvdWs.V.M_Rows == p && ws.BsvdWs.V.N_Cols == p &&
                ws.uBuf.N == m &&
                ws.vBuf.N == n &&
                ws.alpha.N == p &&
                ws.beta.N == p;
            if (!ok)
                throw new ArgumentException(
                    who + ": workspace must be sized for this (m, n, k, oversample) — use " +
                    "Arena.floatSvdTruncatedWorkspace(m, n, k, oversample) with the SAME k and oversample");
        }
    }

    /// <summary>
    /// Reusable scratch storage for svdTruncated (Golub-Kahan-Lanczos). Allocate ONCE via
    /// Arena.floatSvdTruncatedWorkspace(m, n, k, oversample) and reuse across same-shape calls.
    ///
    /// Layout (p = min(k+oversample, n)): UL (m x p) holds the left Lanczos basis u1..up;
    /// VL (n x (p+1)) holds v1..v_{p+1} (the extra column absorbs the last step's v_{p+1} without
    /// overflow); B (p x p) holds the upper-bidiagonal reduction; BsvdWs (p x p U/S/V) is the
    /// inner svdThin scratch; uBuf/vBuf are m/n matvec temporaries; alpha/beta (length p each)
    /// hold the Lanczos diagonal and superdiagonal (beta[p-1] used only for early-stop check).
    ///
    /// NOTE: this workspace avoids the large O(m·p + n·p) Lanczos-basis allocations on each call,
    /// but the inner bidiagonal SVD (svdThin called on the p×p B matrix) still allocates a small
    /// O(p²) Allocator.Temp workspace per svdTruncated call. The workspace is therefore NOT fully
    /// zero-alloc on reuse; only the dominant Lanczos-basis memory is persistent.
    /// </summary>
    public struct floatSvdTruncatedWorkspace
    {
        public floatMxN UL;
        public floatMxN VL;
        public floatMxN B;
        public floatSvdFullWorkspace BsvdWs;
        public floatN uBuf;
        public floatN vBuf;
        public floatN alpha;
        public floatN beta;
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates a GKL-truncated-SVD workspace for an m x n (m >= n) matrix with target rank k
        /// and oversampling p_extra, sized by p = min(k + oversample, n). Pass the SAME k and
        /// oversample to svdTruncated's ref-workspace overload. The buffers are persistent in this
        /// arena (disposed with it), so create the workspace once outside a hot loop.
        /// </summary>
        public floatSvdTruncatedWorkspace floatSvdTruncatedWorkspace(int m, int n, int k, int oversample)
        {
            int p = math.min(k + oversample, n);
            return new floatSvdTruncatedWorkspace
            {
                UL     = floatMat(m, p),
                VL     = floatMat(n, p + 1),
                B      = floatMat(p, p),
                BsvdWs = new floatSvdFullWorkspace
                {
                    U = floatMat(p, p),
                    S = floatVec(p),
                    V = floatMat(p, p)
                },
                uBuf  = floatVec(m),
                vBuf  = floatVec(n),
                alpha = floatVec(p),
                beta  = floatVec(p)
            };
        }

        /// <summary>
        /// Allocates a GKL-truncated-SVD workspace with the generous default Krylov width
        /// p = min(n, max(2*k, k+12)) — matches the svdTruncated convenience overloads that do
        /// not take an explicit oversample. For k in [1,12], p >= k+12; for k > 12, p >= 2*k.
        /// </summary>
        public floatSvdTruncatedWorkspace floatSvdTruncatedWorkspace(int m, int n, int k)
        {
            int p = math.min(n, math.max(2 * k, k + 12));
            return new floatSvdTruncatedWorkspace
            {
                UL     = floatMat(m, p),
                VL     = floatMat(n, p + 1),
                B      = floatMat(p, p),
                BsvdWs = new floatSvdFullWorkspace
                {
                    U = floatMat(p, p),
                    S = floatVec(p),
                    V = floatMat(p, p)
                },
                uBuf  = floatVec(m),
                vBuf  = floatVec(n),
                alpha = floatVec(p),
                beta  = floatVec(p)
            };
        }
    }
}
