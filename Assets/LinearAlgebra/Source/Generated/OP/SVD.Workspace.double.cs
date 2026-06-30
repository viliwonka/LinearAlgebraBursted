namespace LinearAlgebra
{
    /// <summary>
    /// Reusable scratch storage for the zero-alloc SVD solvers (SVD.pinvSolve / SVD.pseudoInverse).
    /// Allocate ONCE (sized for the matrix shape) and reuse it across many same-shape solves to
    /// avoid per-call allocations — create via Arena.doubleSvd_WS(m, n).
    ///
    /// Layout: with k = min(m, n), S is length k, M is k x k (the singular-vector matrix — V for a
    /// tall/square system, W for a wide one), U is the left-factor scratch max(m,n) x k (receives the
    /// Golub-Kahan U of the decomposed orientation), and At is the A^T scratch (n x m) used ONLY when A
    /// is wide (m &lt; n); for m &gt;= n At is left as default (unused). This is exactly the (S, M, U, At)
    /// tuple the scratch-primitive overloads expect, bundled so callers don't size them by hand.
    /// </summary>
    public struct doubleSvd_WS
    {
        public doubleN S;
        public doubleMxN M;
        public doubleMxN U;
        public doubleMxN At;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an SVD-solver workspace sized for an m x n system. With k = min(m, n):
        /// S is length k, M is k x k, U is max(m, n) x k, and At is n x m only when the system is wide
        /// (m &lt; n) — otherwise At is left default (the tall/square path never reads it). The buffers
        /// are persistent in this arena (disposed with it), so create the workspace once outside a
        /// hot loop and pass it to the workspace overloads of pinvSolve / pseudoInverse.
        /// </summary>
        public static doubleSvd_WS doubleSvd_WS(this ref Arena arena, int m, int n)
        {
            int k   = m < n ? m : n;
            int big = m < n ? n : m;
            return new doubleSvd_WS
            {
                S  = arena.doubleVec(k),
                M  = arena.doubleMat(k, k),
                U  = arena.doubleMat(big, k),
                At = (m < n) ? arena.doubleMat(n, m) : default
            };
        }
    }
}
