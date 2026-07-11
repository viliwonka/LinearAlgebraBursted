using System;

namespace LinearAlgebra
{
    public static partial class QRCP
    {
        /// <summary>
        /// Throws if <paramref name="cache"/> is not sized for an n-column QRCP problem. vn1/vn2 are
        /// the ONLY scratch this cache carries — see <see cref="fProxyQRCPCache"/> for why u and the
        /// reflector-apply accumulator w are NOT folded in. Matches Arena.fProxyQRCPCache(n).
        /// </summary>
        static void RequireQRCPWorkspace(in fProxyQRCPCache cache, int n)
        {
            if (cache.vn1.N != n || cache.vn2.N != n)
                throw new ArgumentException("QRCP: cache must be sized for n columns (use Arena.fProxyQRCPCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for QRCP's cache overloads (decomp / decompInPlace / solveInPlace): the
    /// guarded norm-downdating state. vn1 (length n) is the current tracked partial column norm; vn2
    /// (length n) is the norm at the last EXACT computation — the guard compares decay since vn2, not
    /// just the current step's own ratio, so gradual decay across many steps still gets caught (see
    /// decompInPlaceCore).
    ///
    /// Holds the two n-length downdating vectors. Allocate once via Arena.fProxyQRCPCache(n) and reuse
    /// across same-shape calls to avoid the per-call Allocator.Temp allocations the non-cache overloads
    /// make.
    /// </summary>
    public struct fProxyQRCPCache
    {
        public fProxyN vn1;
        public fProxyN vn2;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a QRCP downdating workspace for an n-column system. See
        /// <see cref="fProxyQRCPCache"/> for reuse guidance and per-field purpose.
        /// </summary>
        public static fProxyQRCPCache fProxyQRCPCache(this ref Arena arena, int n)
        {
            return new fProxyQRCPCache
            {
                vn1 = arena.fProxyVec(n),
                vn2 = arena.fProxyVec(n)
            };
        }
    }
}
