using LinearAlgebra.ML;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Reusable scratch storage for zero-alloc Lloyd k-means.
    /// Allocate ONCE (sized for the data shape) via <c>Arena.fProxyKMeans_WS(N, D, k)</c>
    /// and reuse across same-shape calls. All buffers are arena-owned and disposed with the arena.
    ///
    /// Memory layout (all fProxy-scalar counts):
    ///   Gram           N×k  — GEMM output X·Cᵀ, patched in-place to L2² scores each iter
    ///   Ct             D×k  — transposed centroids (refreshed each iteration)
    ///   PointNormSq    N    — ‖xₙ‖² (precomputed once before the Lloyd loop)
    ///   CentNormSq     k    — ‖cⱼ‖² (recomputed each iteration)
    ///   PrevAssignment N    — cluster labels from previous iter (early-exit detection)
    ///   NewCentroids   k×D  — centroid accumulator (zeroed each iteration)
    ///   ClusterCounts  k    — per-cluster point count (zeroed each iteration)
    ///   D2Weights      N    — D² distances used for k-means++ seeding only
    /// </summary>
    public struct fProxyKMeans_WS
    {
        public fProxyMxN Gram;           // N x k  GEMM output X*C^T, patched to scores in-place
        public fProxyMxN Ct;             // D x k  transposed centroids (refreshed each iteration)
        public fProxyN   PointNormSq;    // N      ||x_n||^2 constant (computed once before loop)
        public fProxyN   CentNormSq;     // k      ||c_j||^2 (recomputed each iteration)
        public Indices   PrevAssignment; // N      cluster labels from previous iter (early-exit)
        public fProxyMxN NewCentroids;   // k x D  centroid accumulator (zeroed each iteration)
        public Indices   ClusterCounts;  // k      per-cluster point count (zeroed each iteration)
        public fProxyN   D2Weights;      // N      D^2 distances for k-means++ seeding only
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
        public static LinearAlgebra.ML.fProxyKMeans_WS fProxyKMeans_WS(this ref Arena arena, int N, int D, int k)
        {
            return new LinearAlgebra.ML.fProxyKMeans_WS
            {
                Gram           = arena.fProxyMat(N, k),
                Ct             = arena.fProxyMat(D, k),
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
