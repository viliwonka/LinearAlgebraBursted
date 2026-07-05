using System;

namespace LinearAlgebra
{
    public static partial class CHOP
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n pivoted-Cholesky problem. W (n x n,
        /// the symmetric working copy) is needed by decomp; bt (n, the permuted RHS)
        /// by decompSolve. Matches Arena.doubleCHOPCache(n).
        /// </summary>
        static void RequireCholeskyPivotWorkspace(in doubleCHOPCache ws, int n,
                                                  bool needW, bool needBt)
        {
            if (needW && (ws.W.M_Rows != n || ws.W.N_Cols != n))
                throw new ArgumentException("CHOP: workspace W must be n x n (use Arena.doubleCHOPCache(n))");
            if (needBt && ws.bt.N != n)
                throw new ArgumentException("CHOP: workspace bt must have length n (use Arena.doubleCHOPCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for pivoted (rank-revealing) Cholesky (CHOP.decomp /
    /// decompSolve). Allocate ONCE via Arena.doubleCHOPCache(n) and reuse it across
    /// same-size calls. W (n x n) is the destroyable symmetric working copy the decomposition pivots
    /// on; bt (n) is the permuted right-hand side the solve gathers into.
    ///
    /// NOTE: the rank-deficient minimum-norm solve also forms small rank x rank Gram buffers (g, G, GL)
    /// whose dimension is the runtime numerical rank — these are NOT part of the workspace (the library
    /// has no matrix-view type to slice an n x n buffer to a rank x rank stride) and remain per-call
    /// Allocator.Temp; only the full-rank path is fully zero-alloc with this workspace.
    /// </summary>
    public struct doubleCHOPCache
    {
        public doubleMxN W;
        public doubleN bt;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a pivoted-Cholesky workspace for an n x n matrix. See
        /// <see cref="doubleCHOPCache"/> for reuse guidance.
        /// </summary>
        public static doubleCHOPCache doubleCHOPCache(this ref Arena arena, int n)
        {
            return new doubleCHOPCache
            {
                W = arena.doubleMat(n, n),
                bt = arena.doubleVec(n)
            };
        }
    }
}
