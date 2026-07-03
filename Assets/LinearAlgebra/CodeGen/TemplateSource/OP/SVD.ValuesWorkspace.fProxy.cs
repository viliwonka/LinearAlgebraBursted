using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.fProxySVDValues_WS(m, n) sizing.</summary>
        static void RequireSvdValuesWorkspace(in fProxySVDValues_WS ws, int n)
        {
            bool ok = ws.dVec.N == n && ws.eVec.N == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.fProxySVDValues_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.svdValues (Golub-Kahan bidiagonalization, values-only + implicit-shift
    /// bidiagonal QR, values-only). Allocate ONCE via Arena.fProxySVDValues_WS(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations svdValues's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.bidiagonalizeValues needs (see fProxyBidiag_WS); dVec/eVec
    /// (length n) are the diagonal/superdiagonal the values-only bidiagonal QR diagonalizes in place.
    /// </summary>
    public struct fProxySVDValues_WS
    {
        public fProxyBidiag_WS BidiagWs;
        public fProxyN dVec;
        public fProxyN eVec;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an svdValues workspace for an m x n (m >= n) system — see
        /// <see cref="fProxySVDValues_WS"/> for layout. Persistent in this arena; create once outside a
        /// hot loop and pass to svdValues's ref-workspace overload.
        /// </summary>
        public static fProxySVDValues_WS fProxySVDValues_WS(this ref Arena arena, int m, int n)
        {
            return new fProxySVDValues_WS
            {
                BidiagWs = arena.fProxyBidiag_WS(m, n),
                dVec = arena.fProxyVec(n),
                eVec = arena.fProxyVec(n)
            };
        }
    }
}
