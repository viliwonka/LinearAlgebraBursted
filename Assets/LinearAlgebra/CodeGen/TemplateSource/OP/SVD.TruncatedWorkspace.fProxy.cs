using System;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        static void RequireSvdTruncatedWorkspace(in fProxySvdTruncated_WS ws, int m, int n, int p, string who)
        {
            bool ok =
                ws.UL.M_Rows == p && ws.UL.N_Cols == m &&
                ws.VL.M_Rows == p + 1 && ws.VL.N_Cols == n &&
                ws.dB.N == p && ws.eB.N == p &&
                ws.UtB.M_Rows == p && ws.UtB.N_Cols == p &&
                ws.VtB.M_Rows == p && ws.VtB.N_Cols == p &&
                ws.BsvdWs.U.M_Rows == p && ws.BsvdWs.U.N_Cols == p &&
                ws.BsvdWs.S.N == p &&
                ws.BsvdWs.V.M_Rows == p && ws.BsvdWs.V.N_Cols == p &&
                ws.uBuf.N == m &&
                ws.vBuf.N == n &&
                ws.alpha.N == p &&
                ws.beta.N == p &&
                ws.mu.N == p + 1 &&
                ws.nu.N == p + 1;
            if (!ok)
                throw new ArgumentException(
                    who + ": workspace must be sized for this (m, n, k, oversample) — use " +
                    "Arena.fProxySvdTruncated_WS(m, n, k, oversample) with the SAME k and oversample");
        }
    }

    /// <summary>
    /// Reusable scratch storage for svdTruncated (Golub-Kahan-Lanczos). Allocate ONCE via
    /// Arena.fProxySvdTruncated_WS(m, n, k, oversample) and reuse across same-shape calls.
    ///
    /// Layout (p = min(k+oversample, n)): UL (p x m) holds the left Lanczos basis u_1..u_p as
    /// ROWS (each u_j is a contiguous row of length m, enabling cache-coherent GEMV); VL ((p+1) x n)
    /// holds v_1..v_{p+1} as ROWS (each v_j is a contiguous row of length n; the extra row absorbs
    /// the last step's v_{p+1} without overflow); dB/eB (length p each) hold the diagonal and
    /// superdiagonal of the Lanczos bidiagonal; UtB/VtB (p x p each) are the transposed accumulator
    /// scratch for bidiagonalSvdFromDE; BsvdWs (p x p U/S/V) receives the output singular triplets
    /// of the inner bidiagonal SVD; uBuf/vBuf are m/n matvec temporaries (also reused as coefficient
    /// buffers in DGKS reorthogonalization); alpha/beta (length p each) hold the Lanczos diagonal
    /// and superdiagonal; mu/nu (length p+1 each) hold the ω-recurrence estimates of orthogonality
    /// loss among the left/right Lanczos bases (used by the partial-reorth path; inert otherwise).
    ///
    /// svdTruncated is FULLY zero-alloc on workspace reuse: the inner bidiagonal SVD runs entirely
    /// in dB/eB/UtB/VtB + BsvdWs (all persistent arena memory), with no Allocator.Temp usage.
    /// </summary>
    public struct fProxySvdTruncated_WS
    {
        public fProxyMxN UL;
        public fProxyMxN VL;
        public fProxyN dB;
        public fProxyN eB;
        public fProxyMxN UtB;
        public fProxyMxN VtB;
        public fProxySvdFull_WS BsvdWs;
        public fProxyN uBuf;
        public fProxyN vBuf;
        public fProxyN alpha;
        public fProxyN beta;
        public fProxyN mu;   // length p+1: μ estimates ⟨û_j, û_i⟩ for partial reorth ω-recurrence
        public fProxyN nu;   // length p+1: ν estimates ⟨v̂_j, v̂_i⟩ for partial reorth ω-recurrence
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates a GKL-truncated-SVD workspace for an m x n (m >= n) matrix with target rank k
        /// and oversampling p_extra, sized by p = min(k + oversample, n). Pass the SAME k and
        /// oversample to svdTruncated's ref-workspace overload. The buffers are persistent in this
        /// arena (disposed with it), so create the workspace once outside a hot loop.
        /// </summary>
        public fProxySvdTruncated_WS fProxySvdTruncated_WS(int m, int n, int k, int oversample)
        {
            int p = math.min(k + oversample, n);
            return new fProxySvdTruncated_WS
            {
                UL     = fProxyMat(p, m),
                VL     = fProxyMat(p + 1, n),
                dB     = fProxyVec(p),
                eB     = fProxyVec(p),
                UtB    = fProxyMat(p, p),
                VtB    = fProxyMat(p, p),
                BsvdWs = new fProxySvdFull_WS
                {
                    U = fProxyMat(p, p),
                    S = fProxyVec(p),
                    V = fProxyMat(p, p)
                },
                uBuf  = fProxyVec(m),
                vBuf  = fProxyVec(n),
                alpha = fProxyVec(p),
                beta  = fProxyVec(p),
                mu    = fProxyVec(p + 1),
                nu    = fProxyVec(p + 1)
            };
        }

        /// <summary>
        /// Allocates a GKL-truncated-SVD workspace with the generous default Krylov width
        /// p = min(n, max(2*k, k+12)) — matches the svdTruncated convenience overloads that do
        /// not take an explicit oversample. For k in [1,12], p >= k+12; for k > 12, p >= 2*k.
        /// </summary>
        public fProxySvdTruncated_WS fProxySvdTruncated_WS(int m, int n, int k)
        {
            int p = math.min(n, math.max(2 * k, k + 12));
            return new fProxySvdTruncated_WS
            {
                UL     = fProxyMat(p, m),
                VL     = fProxyMat(p + 1, n),
                dB     = fProxyVec(p),
                eB     = fProxyVec(p),
                UtB    = fProxyMat(p, p),
                VtB    = fProxyMat(p, p),
                BsvdWs = new fProxySvdFull_WS
                {
                    U = fProxyMat(p, p),
                    S = fProxyVec(p),
                    V = fProxyMat(p, p)
                },
                uBuf  = fProxyVec(m),
                vBuf  = fProxyVec(n),
                alpha = fProxyVec(p),
                beta  = fProxyVec(p),
                mu    = fProxyVec(p + 1),
                nu    = fProxyVec(p + 1)
            };
        }
    }
}
