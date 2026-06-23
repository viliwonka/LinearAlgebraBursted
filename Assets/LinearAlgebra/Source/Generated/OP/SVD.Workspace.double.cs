namespace LinearAlgebra
{
    /// <summary>
    /// Reusable scratch storage for the zero-alloc SVD solvers (SVD.pinvSolve / SVD.pseudoInverse).
    /// Allocate ONCE (sized for the matrix shape) and reuse it across many same-shape solves to
    /// avoid per-call allocations — create via Arena.doubleSvdWorkspace(m, n).
    ///
    /// Layout: with k = min(m, n), S is length k, M is k x k (the singular-vector matrix — V for a
    /// tall/square system, W for a wide one), and At is the A^T scratch (n x m) used ONLY when A is
    /// wide (m &lt; n); for m &gt;= n At is left as default (unused). This is exactly the (S, M, At)
    /// triple the scratch-primitive overloads expect, bundled so callers don't size them by hand.
    /// </summary>
    public struct doubleSvdWorkspace
    {
        public doubleN S;
        public doubleMxN M;
        public doubleMxN At;
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates an SVD-solver workspace sized for an m x n system. With k = min(m, n):
        /// S is length k, M is k x k, and At is n x m only when the system is wide (m &lt; n) —
        /// otherwise At is left default (the tall/square path never reads it). The buffers are
        /// persistent in this arena (disposed with it), so create the workspace once outside a
        /// hot loop and pass it to the workspace overloads of pinvSolve / pseudoInverse.
        /// </summary>
        public doubleSvdWorkspace doubleSvdWorkspace(int m, int n)
        {
            int k = m < n ? m : n;
            return new doubleSvdWorkspace
            {
                S  = doubleVec(k),
                M  = doubleMat(k, k),
                At = (m < n) ? doubleMat(n, m) : default
            };
        }
    }
}
