using System;

namespace LinearAlgebra
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ decomposition (W m x n,
        /// v length n) — the layout produced by Arena.doubleLQCache(m, n).
        /// </summary>
        static void RequireLQWorkspace(in doubleLQCache ws, int m, int n)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.v.N == n;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use Arena.doubleLQCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.decomp. Allocate ONCE (sized for the matrix shape) via
    /// Arena.doubleLQCache(m, n) and reuse it across many same-shape calls to avoid the per-call
    /// Allocator.Temp allocations decomp's allocating overload makes internally.
    ///
    /// W (m x n) holds the working copy of A, reduced to [L | 0] in place during the forward sweep
    /// (the discarded upper part of each row is then reused to stash that row's Householder reflector
    /// for the backward Q-reconstruction pass); v (length n) is the reflector scratch vector shared
    /// by both passes.
    /// </summary>
    public struct doubleLQCache
    {
        public doubleMxN W;
        public doubleN v;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQ-decomposition workspace sized for an m x n (m &lt;= n) system. See
        /// <see cref="doubleLQCache"/> for reuse guidance.
        /// </summary>
        public static doubleLQCache doubleLQCache(this ref Arena arena, int m, int n)
        {
            return new doubleLQCache
            {
                W = arena.doubleMat(m, n),
                v = arena.doubleVec(n)
            };
        }
    }
}
