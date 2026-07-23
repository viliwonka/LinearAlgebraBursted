using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class QRCP
    {
        /// <summary>
        /// Throws if <paramref name="cache"/> is not sized for an n-column QRCP problem. vn1/vn2 are
        /// the ONLY scratch this cache carries — see <see cref="fProxyQRCPCache"/> for why u and the
        /// reflector-apply accumulator w are NOT folded in. Matches the fProxyQRCPCache(n, allocator)
        /// constructor.
        /// </summary>
        static void RequireQRCPWorkspace(in fProxyQRCPCache cache, int n)
        {
            if (cache.vn1.N != n || cache.vn2.N != n)
                throw new ArgumentException("QRCP: cache must be sized for n columns (use new fProxyQRCPCache(n, allocator))");
        }
    }

    /// <summary>
    /// Reusable scratch for QRCP's cache overloads (decomp / decompInPlace / solveInPlace): the
    /// guarded norm-downdating state. vn1 (length n) is the current tracked partial column norm; vn2
    /// (length n) is the norm at the last EXACT computation — the guard compares decay since vn2, not
    /// just the current step's own ratio, so gradual decay across many steps still gets caught (see
    /// decompInPlaceCore).
    ///
    /// Holds the two n-length downdating vectors. Allocate once via the Allocator ctor and reuse
    /// across same-shape calls to avoid the per-call Allocator.Temp allocations the non-cache overloads
    /// make.
    /// </summary>
    public struct fProxyQRCPCache : IDisposable
    {
        public fProxyN vn1;
        public fProxyN vn2;

        /// <summary>Allocates a QRCP downdating workspace for an n-column system. Pair with <see cref="Dispose"/>.</summary>
        public fProxyQRCPCache(int n, Allocator allocator)
        {
            vn1 = new fProxyN(n, allocator);
            vn2 = new fProxyN(n, allocator);
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            vn1.Dispose();
            vn2.Dispose();
        }
    }
}
