using System;

namespace LinearAlgebra
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ min-norm solve (L m x m,
        /// Q m x n, y length m, plus the nested LQ-decomposition workspace) — the layout produced
        /// by Arena.fProxyLQMinNormSolveCache(m, n).
        /// </summary>
        static void RequireLQMinNormSolveWorkspace(in fProxyLQMinNormSolveCache ws, int m, int n)
        {
            bool ok =
                ws.L.M_Rows == m && ws.L.N_Cols == m &&
                ws.Q.M_Rows == m && ws.Q.N_Cols == n &&
                ws.y.N == m;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use Arena.fProxyLQMinNormSolveCache(m, n))");
            RequireLQWorkspace(in ws.LQWs, m, n);
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.lqMinNormSolve. Allocate ONCE (sized for the matrix shape) via
    /// Arena.fProxyLQMinNormSolveCache(m, n) and reuse it across many same-shape calls to avoid the
    /// per-call Allocator.Temp allocations lqMinNormSolve's allocating overload makes internally.
    ///
    /// LQWs is the nested workspace lqDecomposition needs (see fProxyLQCache); L (m x m) / Q (m x n)
    /// receive the LQ factors; y (length m) is the forward-solve scratch (starts as a copy of b).
    /// </summary>
    public struct fProxyLQMinNormSolveCache
    {
        public fProxyLQCache LQWs;
        public fProxyMxN L;
        public fProxyMxN Q;
        public fProxyN y;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQ-min-norm-solve workspace sized for an m x n (m &lt;= n) system. See
        /// <see cref="fProxyLQMinNormSolveCache"/> for reuse guidance.
        /// </summary>
        public static fProxyLQMinNormSolveCache fProxyLQMinNormSolveCache(this ref Arena arena, int m, int n)
        {
            return new fProxyLQMinNormSolveCache
            {
                LQWs = arena.fProxyLQCache(m, n),
                L = arena.fProxyMat(m, m),
                Q = arena.fProxyMat(m, n),
                y = arena.fProxyVec(m)
            };
        }
    }
}
