namespace LinearAlgebra
{
    /// <summary>
    /// Reusable scratch storage for the zero-alloc SVD solvers (SVD.pinvSolve / SVD.pseudoInverse).
    /// Allocate ONCE (sized for the matrix shape) and reuse it across many same-shape solves to
    /// avoid per-call allocations — create via Arena.floatSVD_WS(m, n).
    ///
    /// Layout: with k = min(m, n), S is length k, M is k x k (the singular-vector matrix — V for a
    /// tall/square system, W for a wide one), U is the left-factor scratch max(m,n) x k (receives the
    /// Golub-Kahan U of the decomposed orientation), and At is the A^T scratch (n x m) used ONLY when A
    /// is wide (m &lt; n); for m &gt;= n At is left as default (unused). This is exactly the (S, M, U, At)
    /// tuple the scratch-primitive overloads expect, bundled so callers don't size them by hand.
    /// </summary>
    public struct floatSVD_WS
    {
        public floatN S;
        public floatMxN M;
        public floatMxN U;
        public floatMxN At;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>Allocates an SVD-solver workspace sized for an m x n system — see <see cref="floatSVD_WS"/> for layout. Persistent in this arena; create once outside a hot loop.</summary>
        public static floatSVD_WS floatSVD_WS(this ref Arena arena, int m, int n)
        {
            int k   = m < n ? m : n;
            int big = m < n ? n : m;
            return new floatSVD_WS
            {
                S  = arena.floatVec(k),
                M  = arena.floatMat(k, k),
                U  = arena.floatMat(big, k),
                At = (m < n) ? arena.floatMat(n, m) : default
            };
        }
    }
}
