using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n values-only SVD (dVec/eVec length
        /// n) — the layout produced by Arena.floatSVDValues_WS(m, n).
        /// </summary>
        static void RequireSvdValuesWorkspace(in floatSVDValues_WS ws, int n)
        {
            bool ok = ws.dVec.N == n && ws.eVec.N == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.floatSVDValues_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.svdValues (Golub-Kahan bidiagonalization, values-only + implicit-shift
    /// bidiagonal QR, values-only). Allocate ONCE via Arena.floatSVDValues_WS(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations svdValues's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.bidiagonalizeValues needs (see floatBidiag_WS); dVec/eVec
    /// (length n) are the diagonal/superdiagonal the values-only bidiagonal QR diagonalizes in place.
    /// </summary>
    public struct floatSVDValues_WS
    {
        public floatBidiag_WS BidiagWs;
        public floatN dVec;
        public floatN eVec;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an svdValues workspace sized for an m x n (m >= n) system: nested Bidiag workspace,
        /// dVec/eVec length n. The buffers are persistent in this arena (disposed with it), so create
        /// the workspace once outside a hot loop and pass it to the ref-workspace overload of svdValues.
        /// </summary>
        public static floatSVDValues_WS floatSVDValues_WS(this ref Arena arena, int m, int n)
        {
            return new floatSVDValues_WS
            {
                BidiagWs = arena.floatBidiag_WS(m, n),
                dVec = arena.floatVec(n),
                eVec = arena.floatVec(n)
            };
        }
    }
}
