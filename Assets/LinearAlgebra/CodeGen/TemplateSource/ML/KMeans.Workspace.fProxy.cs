using System;

using Unity.Collections;

using LinearAlgebra.ML;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Reusable scratch storage for zero-alloc Lloyd k-means.
    /// Allocate ONCE (sized for the data shape) via <c>Arena.fProxyKMeansCache(N, D, k)</c>
    /// and reuse across same-shape calls. All buffers are arena-owned and disposed with the arena.
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

        /// <summary>Standalone allocation sized identically to <c>Arena.fProxyKMeansCache(N, D, k)</c>. Pair with <see cref="Dispose"/>.</summary>
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

namespace LinearAlgebra
{
    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a k-means workspace sized for <paramref name="N"/> points,
        /// <paramref name="D"/> features, and <paramref name="k"/> clusters.
        /// All buffers are persistent in this arena (disposed with it).
        /// Create once outside hot loops and reuse for same-shape calls.
        /// </summary>
        public static LinearAlgebra.ML.fProxyKMeansCache fProxyKMeansCache(this ref Arena arena, int N, int D, int k)
        {
            return new LinearAlgebra.ML.fProxyKMeansCache
            {
                Gram           = arena.fProxyMat(N, k),
                PointNormSq    = arena.fProxyVec(N),
                CentNormSq     = arena.fProxyVec(k),
                PrevAssignment = arena.Indices(N),
                NewCentroids   = arena.fProxyMat(k, D),
                ClusterCounts  = arena.Indices(k),
                D2Weights      = arena.fProxyVec(N)
            };
        }
    }
}
