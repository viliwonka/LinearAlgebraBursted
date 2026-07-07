using System;

namespace LinearAlgebra
{
    public static partial class LQRP
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQRP decomposition (W m x n,
        /// v length n) — the layout produced by Arena.fProxyLQRPCache(m, n).
        /// </summary>
        static void RequireLQRPWorkspace(in fProxyLQRPCache ws, int m, int n)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.v.N == n;

            if (!ok)
                throw new ArgumentException("LQRP: workspace must be sized for m x n (use Arena.fProxyLQRPCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for LQRP.decomp. Allocate ONCE (sized for the matrix shape) via
    /// Arena.fProxyLQRPCache(m, n) and reuse it across many same-shape calls to avoid the per-call
    /// Allocator.Temp allocations decomp's allocating overload makes internally.
    ///
    /// W (m x n) holds the working copy of A, reduced to [L | reflectors] in place during the forward
    /// sweep (the discarded upper part of each pivoted row is reused to stash that row's Householder
    /// reflector for the backward Q-reconstruction pass); v (length n) is the reflector scratch vector
    /// shared by both passes. The row Pivot P is caller-owned (its size carries the row count), not
    /// folded in here — mirroring how QRCP keeps its column Pivot separate from fProxyQRCPCache.
    /// </summary>
    public struct fProxyLQRPCache
    {
        public fProxyMxN W;
        public fProxyN v;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQRP-decomposition workspace sized for an m x n (m &lt;= n) system. See
        /// <see cref="fProxyLQRPCache"/> for reuse guidance.
        /// </summary>
        public static fProxyLQRPCache fProxyLQRPCache(this ref Arena arena, int m, int n)
        {
            return new fProxyLQRPCache
            {
                W = arena.fProxyMat(m, n),
                v = arena.fProxyVec(n)
            };
        }
    }
}
