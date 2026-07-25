using System;
using Unity.Collections;

namespace BULA
{
    /// <summary>
    /// Reusable scratch storage for the zero-alloc SVD solvers (SVD.pinvSolve / SVD.pseudoInverse).
    /// Allocate ONCE (sized for the matrix shape) and reuse it across many same-shape solves to
    /// avoid per-call allocations — create via the Allocator ctor.
    ///
    /// Layout: with k = min(m, n), S is length k, M is k x k (the singular-vector matrix — V for a
    /// tall/square system, W for a wide one), U is the left-factor scratch max(m,n) x k (receives the
    /// Golub-Kahan U of the decomposed orientation), and At is the A^T scratch (n x m) used ONLY when A
    /// is wide (m &lt; n); for m &gt;= n At is left as default (unused). This is exactly the (S, M, U, At)
    /// tuple the scratch-primitive overloads expect, bundled so callers don't size them by hand.
    /// </summary>
    public struct fProxySVDCache : IDisposable
    {
        public fProxyN S;
        public fProxyMxN M;
        public fProxyMxN U;
        public fProxyMxN At;

        /// <summary>Allocates an SVD-solver workspace sized for an m x n system. Pair with <see cref="Dispose"/>.</summary>
        public fProxySVDCache(int m, int n, Allocator allocator)
        {
            int k   = m < n ? m : n;
            int big = m < n ? n : m;
            S  = new fProxyN(k, allocator);
            M  = new fProxyMxN(k, k, allocator);
            U  = new fProxyMxN(big, k, allocator);
            At = (m < n) ? new fProxyMxN(n, m, allocator) : default;
        }

        /// <summary>Dispose when done.</summary>
        public void Dispose()
        {
            S.Dispose();
            M.Dispose();
            U.Dispose();
            if (At.IsCreated) At.Dispose();
        }
    }
}
