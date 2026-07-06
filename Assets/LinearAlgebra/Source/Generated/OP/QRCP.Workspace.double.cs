using System;

namespace LinearAlgebra
{
    public static partial class QRCP
    {
        /// <summary>
        /// Throws if <paramref name="cache"/> is not sized for an n-column QRCP problem. vn1/vn2 are
        /// the ONLY scratch this cache carries — see <see cref="doubleQRCPCache"/> for why u and the
        /// reflector-apply accumulator w are NOT folded in. Matches Arena.doubleQRCPCache(n).
        /// </summary>
        static void RequireQRCPWorkspace(in doubleQRCPCache cache, int n)
        {
            if (cache.vn1.N != n || cache.vn2.N != n)
                throw new ArgumentException("QRCP: cache must be sized for n columns (use Arena.doubleQRCPCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for QRCP's cache overloads (decomp / decompInPlace / solveInPlace): the
    /// guarded norm-downdating state from docs/spec-qrcp-downdate.md. vn1 (length n) is the current
    /// tracked partial column norm; vn2 (length n) is the norm at the last EXACT computation — the
    /// guard compares decay since vn2, not just the current step's own ratio, so gradual decay across
    /// many steps still gets caught (see decompInPlaceCore).
    ///
    /// Deliberately does NOT include u (the Householder scratch, length m) or w (the reflector-apply
    /// accumulator, length n): QRCP's pivot at each step depends on norms known only after the
    /// previous step's reflector is applied, so this kernel is inherently level-2 and will never gain
    /// QR's level-3 blocked buffers — this cache exists solely to give the downdating state a home
    /// (revisiting OQ-7 of docs/spec-solver-api-rework.md: QRCP now earns a cache, with no dead
    /// fields). Allocate ONCE via Arena.doubleQRCPCache(n) and reuse across same-shape calls to avoid
    /// the per-call Allocator.Temp allocations the non-cache overloads make internally.
    /// </summary>
    public struct doubleQRCPCache
    {
        public doubleN vn1;
        public doubleN vn2;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a QRCP downdating workspace for an n-column system. See
        /// <see cref="doubleQRCPCache"/> for reuse guidance and per-field purpose.
        /// </summary>
        public static doubleQRCPCache doubleQRCPCache(this ref Arena arena, int n)
        {
            return new doubleQRCPCache
            {
                vn1 = arena.doubleVec(n),
                vn2 = arena.doubleVec(n)
            };
        }
    }
}
