using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.fProxySVDValuesCache(m, n) sizing.</summary>
        static void RequireSvdValuesWorkspace(in fProxySVDValuesCache ws, int n)
        {
            bool ok = ws.dVec.N == n && ws.eVec.N == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.fProxySVDValuesCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.values (Golub-Kahan bidiagonalization, values-only + implicit-shift
    /// bidiagonal QR, values-only). Allocate ONCE via Arena.fProxySVDValuesCache(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations values's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.values needs (see fProxyBidiagCache); dVec/eVec
    /// (length n) are the diagonal/superdiagonal the values-only bidiagonal QR diagonalizes in place.
    /// </summary>
    public struct fProxySVDValuesCache : IDisposable
    {
        public fProxyBidiagCache BidiagWs;
        public fProxyN dVec;
        public fProxyN eVec;

        /// <summary>Standalone allocation sized identically to <c>Arena.fProxySVDValuesCache(m, n)</c>. Pair with <see cref="Dispose"/>.</summary>
        public fProxySVDValuesCache(int m, int n, Allocator allocator)
        {
            BidiagWs = new fProxyBidiagCache(m, n, allocator);
            dVec = new fProxyN(n, allocator);
            eVec = new fProxyN(n, allocator);
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            BidiagWs.Dispose();
            dVec.Dispose();
            eVec.Dispose();
        }
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an values workspace for an m x n (m >= n) system — see
        /// <see cref="fProxySVDValuesCache"/> for layout. Persistent in this arena; create once outside a
        /// hot loop and pass to values's ref-workspace overload.
        /// </summary>
        public static fProxySVDValuesCache fProxySVDValuesCache(this ref Arena arena, int m, int n)
        {
            return new fProxySVDValuesCache
            {
                BidiagWs = arena.fProxyBidiagCache(m, n),
                dVec = arena.fProxyVec(n),
                eVec = arena.fProxyVec(n)
            };
        }
    }
}
