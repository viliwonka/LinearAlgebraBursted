using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.floatSVDValuesCache(m, n) sizing.</summary>
        static void RequireSvdValuesWorkspace(in floatSVDValuesCache ws, int n)
        {
            bool ok = ws.dVec.N == n && ws.eVec.N == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.floatSVDValuesCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.values (Golub-Kahan bidiagonalization, values-only + implicit-shift
    /// bidiagonal QR, values-only). Allocate ONCE via Arena.floatSVDValuesCache(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations values's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.values needs (see floatBidiagCache); dVec/eVec
    /// (length n) are the diagonal/superdiagonal the values-only bidiagonal QR diagonalizes in place.
    /// </summary>
    public struct floatSVDValuesCache
    {
        public floatBidiagCache BidiagWs;
        public floatN dVec;
        public floatN eVec;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an values workspace for an m x n (m >= n) system — see
        /// <see cref="floatSVDValuesCache"/> for layout. Persistent in this arena; create once outside a
        /// hot loop and pass to values's ref-workspace overload.
        /// </summary>
        public static floatSVDValuesCache floatSVDValuesCache(this ref Arena arena, int m, int n)
        {
            return new floatSVDValuesCache
            {
                BidiagWs = arena.floatBidiagCache(m, n),
                dVec = arena.floatVec(n),
                eVec = arena.floatVec(n)
            };
        }
    }
}
