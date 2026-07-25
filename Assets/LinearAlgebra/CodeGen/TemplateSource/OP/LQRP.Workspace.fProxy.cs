using System;
using Unity.Collections;

namespace BULA
{
    public static partial class LQRP
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQRP decomposition (W m x n,
        /// v length n) — the layout produced by the fProxyLQRPCache(m, n, allocator) constructor.
        /// </summary>
        static void RequireLQRPWorkspace(in fProxyLQRPCache ws, int m, int n)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.v.N == n;

            if (!ok)
                throw new ArgumentException("LQRP: workspace must be sized for m x n (use new fProxyLQRPCache(m, n, allocator))");
        }
    }

    /// <summary>
    /// Reusable scratch for LQRP.decomp. Allocate ONCE (sized for the matrix shape) via
    /// the Allocator ctor and reuse it across many same-shape calls to avoid the per-call
    /// Allocator.Temp allocations decomp's allocating overload makes internally.
    ///
    /// W (m x n) holds the working copy of A, reduced to [L | reflectors] in place during the forward
    /// sweep (the discarded upper part of each pivoted row is reused to stash that row's Householder
    /// reflector for the backward Q-reconstruction pass); v (length n) is the reflector scratch vector
    /// shared by both passes. The row Pivot P is caller-owned (its size carries the row count), not
    /// folded in here — mirroring how QRCP keeps its column Pivot separate from fProxyQRCPCache.
    /// </summary>
    public struct fProxyLQRPCache : IDisposable
    {
        public fProxyMxN W;
        public fProxyN v;

        /// <summary>Allocates an LQRP-decomposition workspace sized for an m x n (m &lt;= n) system. Pair with <see cref="Dispose"/>.</summary>
        public fProxyLQRPCache(int m, int n, Allocator allocator)
        {
            W = new fProxyMxN(m, n, allocator);
            v = new fProxyN(n, allocator);
        }

        /// <summary>Dispose when done.</summary>
        public void Dispose()
        {
            W.Dispose();
            v.Dispose();
        }
    }
}
