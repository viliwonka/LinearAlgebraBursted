using System;

namespace LinearAlgebra
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ min-norm solve (L m x m,
        /// Q m x n, y length m, plus the nested LQ-decomposition workspace) — the layout produced
        /// by Arena.floatLQMinNormSolveCache(m, n).
        /// </summary>
        static void RequireLQMinNormSolveWorkspace(in floatLQMinNormSolveCache ws, int m, int n)
        {
            bool ok =
                ws.L.M_Rows == m && ws.L.N_Cols == m &&
                ws.Q.M_Rows == m && ws.Q.N_Cols == n &&
                ws.y.N == m;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use Arena.floatLQMinNormSolveCache(m, n))");
            RequireLQWorkspace(in ws.LQWs, m, n);
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.lqMinNormSolve. Allocate ONCE (sized for the matrix shape) via
    /// Arena.floatLQMinNormSolveCache(m, n) and reuse it across many same-shape calls to avoid the
    /// per-call Allocator.Temp allocations lqMinNormSolve's allocating overload makes internally.
    ///
    /// LQWs is the nested workspace lqDecomposition needs (see floatLQCache); L (m x m) / Q (m x n)
    /// receive the LQ factors; y (length m) is the forward-solve scratch (starts as a copy of b).
    /// </summary>
    public struct floatLQMinNormSolveCache
    {
        public floatLQCache LQWs;
        public floatMxN L;
        public floatMxN Q;
        public floatN y;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQ-min-norm-solve workspace sized for an m x n (m &lt;= n) system. See
        /// <see cref="floatLQMinNormSolveCache"/> for reuse guidance.
        /// </summary>
        public static floatLQMinNormSolveCache floatLQMinNormSolveCache(this ref Arena arena, int m, int n)
        {
            return new floatLQMinNormSolveCache
            {
                LQWs = arena.floatLQCache(m, n),
                L = arena.floatMat(m, m),
                Q = arena.floatMat(m, n),
                y = arena.floatVec(m)
            };
        }
    }
}
