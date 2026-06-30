using System;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        static void RequireSvdTruncatedWorkspace(in doubleSvdTruncatedWorkspace ws, int m, int n, int p, string who)
        {
            bool ok =
                ws.UL.M_Rows == p && ws.UL.N_Cols == m &&
                ws.VL.M_Rows == p + 1 && ws.VL.N_Cols == n &&
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
                    "Arena.doubleSvdTruncatedWorkspace(m, n, k, oversample) with the SAME k and oversample");
        }
    }

    /// <summary>
    /// Reusable scratch storage for svdTruncated (Golub-Kahan-Lanczos). Allocate ONCE via
    /// Arena.doubleSvdTruncatedWorkspace(m, n, k, oversample) and reuse across same-shape calls.
    ///
    /// Layout (p = min(k+oversample, n)): UL (p x m) holds the left Lanczos basis u_1..u_p as
    /// ROWS (each u_j is a contiguous row of length m, enabling cache-coherent GEMV); VL ((p+1) x n)
    /// holds v_1..v_{p+1} as ROWS (each v_j is a contiguous row of length n; the extra row absorbs
    /// the last step's v_{p+1} without overflow); B (p x p) holds the upper-bidiagonal reduction;
    /// BsvdWs (p x p U/S/V) is the inner svdThin scratch; uBuf/vBuf are m/n matvec temporaries
    /// (also reused as coefficient buffers in DGKS reorthogonalization); alpha/beta (length p each)
    /// hold the Lanczos diagonal and superdiagonal.
    ///
    /// NOTE: this workspace avoids the large O(m·p + n·p) Lanczos-basis allocations on each call,
    /// but the inner bidiagonal SVD (svdThin called on the p×p B matrix) still allocates a small
    /// O(p²) Allocator.Temp workspace per svdTruncated call. The workspace is therefore NOT fully
    /// zero-alloc on reuse; only the dominant Lanczos-basis memory is persistent.
    /// </summary>
    public struct doubleSvdTruncatedWorkspace
    {
        public doubleMxN UL;
        public doubleMxN VL;
        public doubleMxN B;
        public doubleSvdFullWorkspace BsvdWs;
        public doubleN uBuf;
        public doubleN vBuf;
        public doubleN alpha;
        public doubleN beta;
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates a GKL-truncated-SVD workspace for an m x n (m >= n) matrix with target rank k
        /// and oversampling p_extra, sized by p = min(k + oversample, n). Pass the SAME k and
        /// oversample to svdTruncated's ref-workspace overload. The buffers are persistent in this
        /// arena (disposed with it), so create the workspace once outside a hot loop.
        /// </summary>
        public doubleSvdTruncatedWorkspace doubleSvdTruncatedWorkspace(int m, int n, int k, int oversample)
        {
            int p = math.min(k + oversample, n);
            return new doubleSvdTruncatedWorkspace
            {
                UL     = doubleMat(p, m),
                VL     = doubleMat(p + 1, n),
                B      = doubleMat(p, p),
                BsvdWs = new doubleSvdFullWorkspace
                {
                    U = doubleMat(p, p),
                    S = doubleVec(p),
                    V = doubleMat(p, p)
                },
                uBuf  = doubleVec(m),
                vBuf  = doubleVec(n),
                alpha = doubleVec(p),
                beta  = doubleVec(p)
            };
        }

        /// <summary>
        /// Allocates a GKL-truncated-SVD workspace with the generous default Krylov width
        /// p = min(n, max(2*k, k+12)) — matches the svdTruncated convenience overloads that do
        /// not take an explicit oversample. For k in [1,12], p >= k+12; for k > 12, p >= 2*k.
        /// </summary>
        public doubleSvdTruncatedWorkspace doubleSvdTruncatedWorkspace(int m, int n, int k)
        {
            int p = math.min(n, math.max(2 * k, k + 12));
            return new doubleSvdTruncatedWorkspace
            {
                UL     = doubleMat(p, m),
                VL     = doubleMat(p + 1, n),
                B      = doubleMat(p, p),
                BsvdWs = new doubleSvdFullWorkspace
                {
                    U = doubleMat(p, p),
                    S = doubleVec(p),
                    V = doubleMat(p, p)
                },
                uBuf  = doubleVec(m),
                vBuf  = doubleVec(n),
                alpha = doubleVec(p),
                beta  = doubleVec(p)
            };
        }
    }
}
