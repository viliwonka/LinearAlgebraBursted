using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n thin SVD (B/Vt n x n, dVec/eVec
        /// length n, Ut n x m) — the layout produced by Arena.fProxySVDThin_WS(m, n). Also validates
        /// the nested Bidiag workspace via Bidiag's own requirement (through bidiagonalize itself).
        /// </summary>
        static void RequireSvdThinWorkspace(in fProxySVDThin_WS ws, int m, int n)
        {
            bool ok =
                ws.B.M_Rows == n && ws.B.N_Cols == n &&
                ws.dVec.N == n && ws.eVec.N == n &&
                ws.Ut.M_Rows == n && ws.Ut.N_Cols == m &&
                ws.Vt.M_Rows == n && ws.Vt.N_Cols == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.fProxySVDThin_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.svdThin (Golub-Kahan bidiagonalization + implicit-shift bidiagonal QR).
    /// Allocate ONCE (sized for the matrix shape) via Arena.fProxySVDThin_WS(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations svdThin's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.bidiagonalize needs (see fProxyBidiag_WS); B (n x n) is
    /// the bidiagonal factor; dVec/eVec (length n) are the diagonal/superdiagonal the bidiagonal QR
    /// diagonalizes; Ut (n x m) / Vt (n x n) are the transposed accumulators the QR sweep rotates
    /// (unit-stride rows, same trick as eigenSymmetric/svdDecomposition).
    /// </summary>
    public struct fProxySVDThin_WS
    {
        public fProxyBidiag_WS BidiagWs;
        public fProxyMxN B;
        public fProxyN dVec;
        public fProxyN eVec;
        public fProxyMxN Ut;
        public fProxyMxN Vt;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an svdThin workspace sized for an m x n (m >= n) system: nested Bidiag workspace,
        /// B/Vt n x n, dVec/eVec length n, Ut n x m. The buffers are persistent in this arena (disposed
        /// with it), so create the workspace once outside a hot loop and pass it to the ref-workspace
        /// overload of svdThin.
        /// </summary>
        public static fProxySVDThin_WS fProxySVDThin_WS(this ref Arena arena, int m, int n)
        {
            return new fProxySVDThin_WS
            {
                BidiagWs = arena.fProxyBidiag_WS(m, n),
                B = arena.fProxyMat(n, n),
                dVec = arena.fProxyVec(n),
                eVec = arena.fProxyVec(n),
                Ut = arena.fProxyMat(n, m),
                Vt = arena.fProxyMat(n, n)
            };
        }
    }
}
