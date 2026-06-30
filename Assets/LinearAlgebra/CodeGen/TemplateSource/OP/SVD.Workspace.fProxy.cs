namespace LinearAlgebra
{
    /// <summary>
    /// Reusable scratch storage for the zero-alloc SVD solvers (SVD.pinvSolve / SVD.pseudoInverse).
    /// Allocate ONCE (sized for the matrix shape) and reuse it across many same-shape solves to
    /// avoid per-call allocations — create via Arena.fProxySvd_WS(m, n).
    ///
    /// Layout: with k = min(m, n), S is length k, M is k x k (the singular-vector matrix — V for a
    /// tall/square system, W for a wide one), U is the left-factor scratch max(m,n) x k (receives the
    /// Golub-Kahan U of the decomposed orientation), and At is the A^T scratch (n x m) used ONLY when A
    /// is wide (m &lt; n); for m &gt;= n At is left as default (unused). This is exactly the (S, M, U, At)
    /// tuple the scratch-primitive overloads expect, bundled so callers don't size them by hand.
    /// </summary>
    public struct fProxySvd_WS
    {
        public fProxyN S;
        public fProxyMxN M;
        public fProxyMxN U;
        public fProxyMxN At;
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates an SVD-solver workspace sized for an m x n system. With k = min(m, n):
        /// S is length k, M is k x k, U is max(m, n) x k, and At is n x m only when the system is wide
        /// (m &lt; n) — otherwise At is left default (the tall/square path never reads it). The buffers
        /// are persistent in this arena (disposed with it), so create the workspace once outside a
        /// hot loop and pass it to the workspace overloads of pinvSolve / pseudoInverse.
        /// </summary>
        public fProxySvd_WS fProxySvd_WS(int m, int n)
        {
            int k   = m < n ? m : n;
            int big = m < n ? n : m;
            return new fProxySvd_WS
            {
                S  = fProxyVec(k),
                M  = fProxyMat(k, k),
                U  = fProxyMat(big, k),
                At = (m < n) ? fProxyMat(n, m) : default
            };
        }
    }
}
