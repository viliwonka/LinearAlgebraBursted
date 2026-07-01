using System;

namespace LinearAlgebra
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ decomposition (W m x n,
        /// v length n) — the layout produced by Arena.fProxyLQ_WS(m, n).
        /// </summary>
        static void RequireLQWorkspace(in fProxyLQ_WS ws, int m, int n)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.v.N == n;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use Arena.fProxyLQ_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.lqDecomposition. Allocate ONCE (sized for the matrix shape) via
    /// Arena.fProxyLQ_WS(m, n) and reuse it across many same-shape calls to avoid the per-call
    /// Allocator.Temp allocations lqDecomposition's allocating overload makes internally.
    ///
    /// W (m x n) holds the working copy of A, reduced to [L | 0] in place during the forward sweep
    /// (the discarded upper part of each row is then reused to stash that row's Householder reflector
    /// for the backward Q-reconstruction pass); v (length n) is the reflector scratch vector shared
    /// by both passes.
    /// </summary>
    public struct fProxyLQ_WS
    {
        public fProxyMxN W;
        public fProxyN v;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQ-decomposition workspace sized for an m x n (m &lt;= n) system: W m x n,
        /// v length n. The buffers are persistent in this arena (disposed with it), so create the
        /// workspace once outside a hot loop and pass it to the ref-workspace overload of
        /// lqDecomposition.
        /// </summary>
        public static fProxyLQ_WS fProxyLQ_WS(this ref Arena arena, int m, int n)
        {
            return new fProxyLQ_WS
            {
                W = arena.fProxyMat(m, n),
                v = arena.fProxyVec(n)
            };
        }
    }
}
