using System;
using Unity.Collections;

namespace BULA
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ min-norm solve (L m x m,
        /// y length m, plus the nested LQ workspace whose W doubles as the factor-only working
        /// buffer) — the layout produced by the fProxyLQMinNormCache(m, n, allocator) constructor.
        /// </summary>
        static void RequireLQMinNormSolveWorkspace(in fProxyLQMinNormCache ws, int m, int n)
        {
            bool ok =
                ws.L.M_Rows == m && ws.L.N_Cols == m &&
                ws.y.N == m;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use new fProxyLQMinNormCache(m, n, allocator))");
            RequireLQWorkspace(in ws.LQWs, m, n);
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.minNormSolve. Allocate ONCE (sized for the matrix shape) via
    /// the Allocator ctor and reuse it across many same-shape calls to avoid the
    /// per-call Allocator.Temp allocations minNormSolve's allocating overload makes internally.
    ///
    /// LQWs is the nested LQ workspace (see fProxyLQCache): its W (m x n) is the factor-only working
    /// buffer that receives a copy of A and the stored row-reflectors, and its v is the reflector
    /// scratch. L (m x m) receives the lower-triangular factor; y (length m) is the forward-solve
    /// scratch (starts as a copy of b). No dense-Q buffer is carried — the fused solve applies Qᵀ
    /// straight from W's reflectors (see LQ.applyQtFromReflectors).
    /// </summary>
    public struct fProxyLQMinNormCache : IDisposable
    {
        public fProxyLQCache LQWs;
        public fProxyMxN L;
        public fProxyN y;

        /// <summary>Allocates an LQ-min-norm-solve workspace sized for an m x n (m &lt;= n) system. Pair with <see cref="Dispose"/>.</summary>
        public fProxyLQMinNormCache(int m, int n, Allocator allocator)
        {
            LQWs = new fProxyLQCache(m, n, allocator);
            L = new fProxyMxN(m, m, allocator);
            y = new fProxyN(m, allocator);
        }

        /// <summary>Dispose when done.</summary>
        public void Dispose()
        {
            LQWs.Dispose();
            L.Dispose();
            y.Dispose();
        }
    }
}
