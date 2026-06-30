using LinearAlgebra.ML;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Reusable scratch storage for zero-alloc Lloyd k-means.
    /// Allocate ONCE (sized for the data shape) via <c>Arena.doubleKMeans_WS(N, D, k)</c>
    /// and reuse across same-shape calls. All buffers are arena-owned and disposed with the arena.
    ///
    /// Memory layout (all double-scalar counts):
    ///   Gram           N×k  — GEMM output X·Cᵀ, patched in-place to L2² scores each iter
    ///   Ct             D×k  — transposed centroids (refreshed each iteration)
    ///   PointNormSq    N    — ‖xₙ‖² (precomputed once before the Lloyd loop)
    ///   CentNormSq     k    — ‖cⱼ‖² (recomputed each iteration)
    ///   PrevAssignment N    — cluster labels from previous iter (early-exit detection)
    ///   NewCentroids   k×D  — centroid accumulator (zeroed each iteration)
    ///   ClusterCounts  k    — per-cluster point count (zeroed each iteration)
    ///   D2Weights      N    — D² distances used for k-means++ seeding only
    /// </summary>
    public struct doubleKMeans_WS
    {
        public doubleMxN Gram;           // N x k  GEMM output X*C^T, patched to scores in-place
        public doubleMxN Ct;             // D x k  transposed centroids (refreshed each iteration)
        public doubleN   PointNormSq;    // N      ||x_n||^2 constant (computed once before loop)
        public doubleN   CentNormSq;     // k      ||c_j||^2 (recomputed each iteration)
        public Indices   PrevAssignment; // N      cluster labels from previous iter (early-exit)
        public doubleMxN NewCentroids;   // k x D  centroid accumulator (zeroed each iteration)
        public Indices   ClusterCounts;  // k      per-cluster point count (zeroed each iteration)
        public doubleN   D2Weights;      // N      D^2 distances for k-means++ seeding only
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
        public static LinearAlgebra.ML.doubleKMeans_WS doubleKMeans_WS(this ref Arena arena, int N, int D, int k)
        {
            return new LinearAlgebra.ML.doubleKMeans_WS
            {
                Gram           = arena.doubleMat(N, k),
                Ct             = arena.doubleMat(D, k),
                PointNormSq    = arena.doubleVec(N),
                CentNormSq     = arena.doubleVec(k),
                PrevAssignment = arena.Indices(N),
                NewCentroids   = arena.doubleMat(k, D),
                ClusterCounts  = arena.Indices(k),
                D2Weights      = arena.doubleVec(N)
            };
        }
    }
}
