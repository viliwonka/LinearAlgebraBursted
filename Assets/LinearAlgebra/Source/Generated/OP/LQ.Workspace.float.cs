using System;

namespace LinearAlgebra
{
    public static partial class LQ
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n LQ decomposition (T n x m,
        /// Rqr m x m, qrU length n, qrW length m) — the layout produced by Arena.floatLQ_WS(m, n).
        /// </summary>
        static void RequireLQWorkspace(in floatLQ_WS ws, int m, int n)
        {
            bool ok =
                ws.T.M_Rows == n && ws.T.N_Cols == m &&
                ws.Rqr.M_Rows == m && ws.Rqr.N_Cols == m &&
                ws.qrU.N == n && ws.qrW.N == m;

            if (!ok)
                throw new ArgumentException("LQ: workspace must be sized for m x n (use Arena.floatLQ_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for LQ.lqDecomposition. Allocate ONCE (sized for the matrix shape) via
    /// Arena.floatLQ_WS(m, n) and reuse it across many same-shape calls to avoid the per-call
    /// Allocator.Temp allocations lqDecomposition's allocating overload makes internally.
    ///
    /// T (n x m) holds Aᵀ, consumed in place by QR.qrDecomposition (destroys its input); Rqr (m x m)
    /// receives the QR factor R; qrU (length n) / qrW (length m) are QR.qrDecomposition's own
    /// zero-alloc scratch vectors (u.N == T.M_Rows == n, w.N == T.N_Cols == m).
    /// </summary>
    public struct floatLQ_WS
    {
        public floatMxN T;
        public floatMxN Rqr;
        public floatN qrU;
        public floatN qrW;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an LQ-decomposition workspace sized for an m x n (m &lt;= n) system: T n x m,
        /// Rqr m x m, qrU length n, qrW length m. The buffers are persistent in this arena (disposed
        /// with it), so create the workspace once outside a hot loop and pass it to the ref-workspace
        /// overload of lqDecomposition.
        /// </summary>
        public static floatLQ_WS floatLQ_WS(this ref Arena arena, int m, int n)
        {
            return new floatLQ_WS
            {
                T = arena.floatMat(n, m),
                Rqr = arena.floatMat(m, m),
                qrU = arena.floatVec(n),
                qrW = arena.floatVec(m)
            };
        }
    }
}
