using System;

namespace LinearAlgebra
{
    public static partial class LQRP
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQRP decomposition (W m x n,
        /// v length n) — the layout produced by Arena.floatLQRPCache(m, n).
        /// </summary>
        static void RequireLQRPWorkspace(in floatLQRPCache ws, int m, int n)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.v.N == n;

            if (!ok)
                throw new ArgumentException("LQRP: workspace must be sized for m x n (use Arena.floatLQRPCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for LQRP.decomp. Allocate ONCE (sized for the matrix shape) via
    /// Arena.floatLQRPCache(m, n) and reuse it across many same-shape calls to avoid the per-call
    /// Allocator.Temp allocations decomp's allocating overload makes internally.
    ///
    /// W (m x n) holds the working copy of A, reduced to [L | reflectors] in place during the forward
    /// sweep (the discarded upper part of each pivoted row is reused to stash that row's Householder
    /// reflector for the backward Q-reconstruction pass); v (length n) is the reflector scratch vector
    /// shared by both passes. The row Pivot P is caller-owned (its size carries the row count), not
    /// folded in here — mirroring how QRCP keeps its column Pivot separate from floatQRCPCache.
    /// </summary>
    public struct floatLQRPCache
    {
        public floatMxN W;
        public floatN v;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQRP-decomposition workspace sized for an m x n (m &lt;= n) system. See
        /// <see cref="floatLQRPCache"/> for reuse guidance.
        /// </summary>
        public static floatLQRPCache floatLQRPCache(this ref Arena arena, int m, int n)
        {
            return new floatLQRPCache
            {
                W = arena.floatMat(m, n),
                v = arena.floatVec(n)
            };
        }
    }
}
