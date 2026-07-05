using System;

namespace LinearAlgebra
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ min-norm solve (L m x m,
        /// Q m x n, y length m, plus the nested LQ-decomposition workspace) — the layout produced
        /// by Arena.floatLQMinNormCache(m, n).
        /// </summary>
        static void RequireLQMinNormSolveWorkspace(in floatLQMinNormCache ws, int m, int n)
        {
            bool ok =
                ws.L.M_Rows == m && ws.L.N_Cols == m &&
                ws.Q.M_Rows == m && ws.Q.N_Cols == n &&
                ws.y.N == m;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use Arena.floatLQMinNormCache(m, n))");
            RequireLQWorkspace(in ws.LQWs, m, n);
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.minNormSolve. Allocate ONCE (sized for the matrix shape) via
    /// Arena.floatLQMinNormCache(m, n) and reuse it across many same-shape calls to avoid the
    /// per-call Allocator.Temp allocations minNormSolve's allocating overload makes internally.
    ///
    /// LQWs is the nested workspace decomp needs (see floatLQCache); L (m x m) / Q (m x n)
    /// receive the LQ factors; y (length m) is the forward-solve scratch (starts as a copy of b).
    /// </summary>
    public struct floatLQMinNormCache
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
        /// <see cref="floatLQMinNormCache"/> for reuse guidance.
        /// </summary>
        public static floatLQMinNormCache floatLQMinNormCache(this ref Arena arena, int m, int n)
        {
            return new floatLQMinNormCache
            {
                LQWs = arena.floatLQCache(m, n),
                L = arena.floatMat(m, m),
                Q = arena.floatMat(m, n),
                y = arena.floatVec(m)
            };
        }
    }
}
