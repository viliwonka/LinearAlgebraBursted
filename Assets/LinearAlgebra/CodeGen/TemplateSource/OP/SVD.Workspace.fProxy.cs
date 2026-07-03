namespace LinearAlgebra
{
    /// <summary>
    /// Reusable scratch storage for the zero-alloc SVD solvers (SVD.pinvSolve / SVD.pseudoInverse).
    /// Allocate ONCE (sized for the matrix shape) and reuse it across many same-shape solves to
    /// avoid per-call allocations — create via Arena.fProxySVDCache(m, n).
    ///
    /// Layout: with k = min(m, n), S is length k, M is k x k (the singular-vector matrix — V for a
    /// tall/square system, W for a wide one), U is the left-factor scratch max(m,n) x k (receives the
    /// Golub-Kahan U of the decomposed orientation), and At is the A^T scratch (n x m) used ONLY when A
    /// is wide (m &lt; n); for m &gt;= n At is left as default (unused). This is exactly the (S, M, U, At)
    /// tuple the scratch-primitive overloads expect, bundled so callers don't size them by hand.
    /// </summary>
    public struct fProxySVDCache
    {
        public fProxyN S;
        public fProxyMxN M;
        public fProxyMxN U;
        public fProxyMxN At;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>Allocates an SVD-solver workspace sized for an m x n system — see <see cref="fProxySVDCache"/> for layout. Persistent in this arena; create once outside a hot loop.</summary>
        public static fProxySVDCache fProxySVDCache(this ref Arena arena, int m, int n)
        {
            int k   = m < n ? m : n;
            int big = m < n ? n : m;
            return new fProxySVDCache
            {
                S  = arena.fProxyVec(k),
                M  = arena.fProxyMat(k, k),
                U  = arena.fProxyMat(big, k),
                At = (m < n) ? arena.fProxyMat(n, m) : default
            };
        }
    }
}
