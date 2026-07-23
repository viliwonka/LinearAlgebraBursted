using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class CHOP
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n pivoted-Cholesky problem. W (n x n,
        /// the symmetric working copy) is needed by decomp; bt (n, the permuted RHS)
        /// by decompSolve. Matches Arena.fProxyCHOPCache(n).
        /// </summary>
        static void RequireCholeskyPivotWorkspace(in fProxyCHOPCache ws, int n,
                                                  bool needW, bool needBt)
        {
            if (needW && (ws.W.M_Rows != n || ws.W.N_Cols != n))
                throw new ArgumentException("CHOP: workspace W must be n x n (use Arena.fProxyCHOPCache(n))");
            if (needBt && ws.bt.N != n)
                throw new ArgumentException("CHOP: workspace bt must have length n (use Arena.fProxyCHOPCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for pivoted (rank-revealing) Cholesky (CHOP.decomp /
    /// decompSolve). Allocate ONCE via Arena.fProxyCHOPCache(n) and reuse it across
    /// same-size calls. W (n x n) is the destroyable symmetric working copy the decomposition pivots
    /// on; bt (n) is the permuted right-hand side the solve gathers into.
    ///
    /// NOTE: the rank-deficient minimum-norm solve also forms small rank x rank Gram buffers (g, G, GL)
    /// whose dimension is the runtime numerical rank — these are NOT part of the workspace (the library
    /// has no matrix-view type to slice an n x n buffer to a rank x rank stride) and remain per-call
    /// Allocator.Temp; only the full-rank path is fully zero-alloc with this workspace.
    /// </summary>
    public struct fProxyCHOPCache : IDisposable
    {
        public fProxyMxN W;
        public fProxyN bt;

        /// <summary>Standalone allocation sized identically to <c>Arena.fProxyCHOPCache(n)</c>. Pair with <see cref="Dispose"/>.</summary>
        public fProxyCHOPCache(int n, Allocator allocator)
        {
            W = new fProxyMxN(n, n, allocator);
            bt = new fProxyN(n, allocator);
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            W.Dispose();
            bt.Dispose();
        }
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a pivoted-Cholesky workspace for an n x n matrix. See
        /// <see cref="fProxyCHOPCache"/> for reuse guidance.
        /// </summary>
        public static fProxyCHOPCache fProxyCHOPCache(this ref Arena arena, int n)
        {
            return new fProxyCHOPCache
            {
                W = arena.fProxyMat(n, n),
                bt = arena.fProxyVec(n)
            };
        }
    }
}
