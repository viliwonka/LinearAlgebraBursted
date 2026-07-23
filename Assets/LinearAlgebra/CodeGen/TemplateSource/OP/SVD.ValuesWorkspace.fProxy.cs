using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches the fProxySVDValuesCache(m, n, allocator) constructor sizing.</summary>
        static void RequireSvdValuesWorkspace(in fProxySVDValuesCache ws, int n)
        {
            bool ok = ws.dVec.N == n && ws.eVec.N == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use new fProxySVDValuesCache(m, n, allocator))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.values (Golub-Kahan bidiagonalization, values-only + implicit-shift
    /// bidiagonal QR, values-only). Allocate ONCE via the Allocator ctor and reuse it across
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

        /// <summary>Allocates a values workspace for an m x n (m >= n) system. Pair with <see cref="Dispose"/>.</summary>
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
}
