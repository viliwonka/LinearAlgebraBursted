using System;

namespace LinearAlgebra
{
    public static partial class Cholesky
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n pivoted-Cholesky problem. W (n x n,
        /// the symmetric working copy) is needed by choleskyDecompositionPivot; bt (n, the permuted RHS)
        /// by choleskyPivotSolve. Matches Arena.floatCholeskyPivotCache(n).
        /// </summary>
        static void RequireCholeskyPivotWorkspace(in floatCholeskyPivotCache ws, int n,
                                                  bool needW, bool needBt)
        {
            if (needW && (ws.W.M_Rows != n || ws.W.N_Cols != n))
                throw new ArgumentException("Cholesky: workspace W must be n x n (use Arena.floatCholeskyPivotCache(n))");
            if (needBt && ws.bt.N != n)
                throw new ArgumentException("Cholesky: workspace bt must have length n (use Arena.floatCholeskyPivotCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for pivoted (rank-revealing) Cholesky (Cholesky.choleskyDecompositionPivot /
    /// choleskyPivotSolve). Allocate ONCE via Arena.floatCholeskyPivotCache(n) and reuse it across
    /// same-size calls. W (n x n) is the destroyable symmetric working copy the decomposition pivots
    /// on; bt (n) is the permuted right-hand side the solve gathers into.
    ///
    /// NOTE: the rank-deficient minimum-norm solve also forms small rank x rank Gram buffers (g, G, GL)
    /// whose dimension is the runtime numerical rank — these are NOT part of the workspace (the library
    /// has no matrix-view type to slice an n x n buffer to a rank x rank stride) and remain per-call
    /// Allocator.Temp; only the full-rank path is fully zero-alloc with this workspace.
    /// </summary>
    public struct floatCholeskyPivotCache
    {
        public floatMxN W;
        public floatN bt;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a pivoted-Cholesky workspace for an n x n matrix. See
        /// <see cref="floatCholeskyPivotCache"/> for reuse guidance.
        /// </summary>
        public static floatCholeskyPivotCache floatCholeskyPivotCache(this ref Arena arena, int n)
        {
            return new floatCholeskyPivotCache
            {
                W = arena.floatMat(n, n),
                bt = arena.floatVec(n)
            };
        }
    }
}
