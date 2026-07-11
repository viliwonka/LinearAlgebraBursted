using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.fProxySVDThinCache(m, n) sizing (BidiagWs is validated separately, by Bidiag.decomp itself).</summary>
        static void RequireSvdThinWorkspace(in fProxySVDThinCache ws, int m, int n)
        {
            bool ok =
                ws.B.M_Rows == n && ws.B.N_Cols == n &&
                ws.dVec.N == n && ws.eVec.N == n &&
                ws.Ut.M_Rows == n && ws.Ut.N_Cols == m &&
                ws.Vt.M_Rows == n && ws.Vt.N_Cols == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.fProxySVDThinCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.thin (Golub-Kahan bidiagonalization + implicit-shift bidiagonal QR).
    /// Allocate ONCE (sized for the matrix shape) via Arena.fProxySVDThinCache(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations thin's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.decomp needs (see fProxyBidiagCache); B (n x n) is
    /// the bidiagonal factor; dVec/eVec (length n) are the diagonal/superdiagonal the bidiagonal QR
    /// diagonalizes; Ut (n x m) / Vt (n x n) are the transposed accumulators the QR sweep rotates
    /// (unit-stride rows, same trick as Eigen.symmetricInPlace).
    /// </summary>
    public struct fProxySVDThinCache
    {
        public fProxyBidiagCache BidiagWs;
        public fProxyMxN B;
        public fProxyN dVec;
        public fProxyN eVec;
        public fProxyMxN Ut;
        public fProxyMxN Vt;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an thin workspace for an m x n (m >= n) system — see
        /// <see cref="fProxySVDThinCache"/> for layout. Persistent in this arena; create once outside a
        /// hot loop and pass to thin's ref-workspace overload.
        /// </summary>
        public static fProxySVDThinCache fProxySVDThinCache(this ref Arena arena, int m, int n)
        {
            return new fProxySVDThinCache
            {
                BidiagWs = arena.fProxyBidiagCache(m, n),
                B = arena.fProxyMat(n, n),
                dVec = arena.fProxyVec(n),
                eVec = arena.fProxyVec(n),
                Ut = arena.fProxyMat(n, m),
                Vt = arena.fProxyMat(n, n)
            };
        }
    }
}
