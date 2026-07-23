using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches the fProxySVDFullCache(m, n, allocator) constructor sizing.</summary>
        static void RequireSvdFullWorkspace(in fProxySVDFullCache ws, int m, int n)
        {
            if (ws.U.M_Rows != m || ws.U.N_Cols != n || ws.S.N != n || ws.V.M_Rows != n || ws.V.N_Cols != n)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use new fProxySVDFullCache(m, n, allocator))");
        }
    }

    /// <summary>
    /// Reusable scratch for the full-SVD-family ops that each compute one Golub-Kahan SVD of an m x n
    /// (m >= n) matrix and slice it: truncated, lowRankApprox, nullspaceBasis, rangeBasis. Allocate
    /// ONCE via the Allocator ctor and reuse across same-shape calls to avoid per-call temp
    /// allocations.
    ///
    /// Layout matches thin's (U, S, V): U is m x n (left singular vectors), S is length n (singular
    /// values), V is n x n (right singular vectors).
    ///
    /// NOTE: removes the per-call U/S/V temp-pool allocations; the inner Golub-Kahan SVD still uses a
    /// little Allocator.Temp scratch of its own, so this is low-alloc rather than strictly zero-alloc.
    /// </summary>
    public struct fProxySVDFullCache : IDisposable
    {
        public fProxyMxN U;
        public fProxyN S;
        public fProxyMxN V;

        /// <summary>Allocates a full-SVD-family workspace for an m x n (m >= n) system. Pair with <see cref="Dispose"/>.</summary>
        public fProxySVDFullCache(int m, int n, Allocator allocator)
        {
            U = new fProxyMxN(m, n, allocator);
            S = new fProxyN(n, allocator);
            V = new fProxyMxN(n, n, allocator);
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            U.Dispose();
            S.Dispose();
            V.Dispose();
        }
    }
}
