using System;

using Unity.Collections;

using LinearAlgebra.ML;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Reusable scratch storage for zero-alloc Lloyd k-means.
    /// Allocate ONCE (sized for the data shape) via the Allocator ctor
    /// and reuse across same-shape calls.
    /// </summary>
    public struct fProxyKMeansCache : IDisposable
    {
        public fProxyMxN Gram;           // N x k  GEMM output X*C^T, patched to scores in-place
        public fProxyN   PointNormSq;    // N      ||x_n||^2 constant (computed once before loop)
        public fProxyN   CentNormSq;     // k      ||c_j||^2 (recomputed each iteration)
        public Indices   PrevAssignment; // N      cluster labels from previous iter (early-exit)
        public fProxyMxN NewCentroids;   // k x D  centroid accumulator (zeroed each iteration)
        public Indices   ClusterCounts;  // k      per-cluster point count (zeroed each iteration)
        public fProxyN   D2Weights;      // N      D^2 distances for k-means++ seeding only

        /// <summary>Allocates a k-means workspace sized for N points, D features, and k clusters. Pair with <see cref="Dispose"/>.</summary>
        public fProxyKMeansCache(int N, int D, int k, Allocator allocator)
        {
            Gram           = new fProxyMxN(N, k, allocator);
            PointNormSq    = new fProxyN(N, allocator);
            CentNormSq     = new fProxyN(k, allocator);
            PrevAssignment = new Indices(N, allocator);
            NewCentroids   = new fProxyMxN(k, D, allocator);
            ClusterCounts  = new Indices(k, allocator);
            D2Weights      = new fProxyN(N, allocator);
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            Gram.Dispose();
            PointNormSq.Dispose();
            CentNormSq.Dispose();
            PrevAssignment.Dispose();
            NewCentroids.Dispose();
            ClusterCounts.Dispose();
            D2Weights.Dispose();
        }
    }
}
