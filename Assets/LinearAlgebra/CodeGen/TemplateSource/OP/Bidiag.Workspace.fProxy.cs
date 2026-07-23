using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class Bidiag
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n bidiagonalization. The common
        /// buffers (W m x n, uVec m, vVec n, wScratch n) are always required; leftU (m x n) is required
        /// only by the full <see cref="decomp"/> (which reconstructs U), not by
        /// <see cref="values"/>. Matches Arena.fProxyBidiagCache(m, n).
        /// </summary>
        static void RequireBidiagWorkspace(in fProxyBidiagCache ws, int m, int n, bool needLeftU)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.uVec.N == m &&
                ws.vVec.N == n &&
                ws.wScratch.N == n &&
                (!needLeftU || (ws.leftU.M_Rows == m && ws.leftU.N_Cols == n));

            if (!ok)
                throw new ArgumentException("Bidiag: workspace must be sized for m x n (use Arena.fProxyBidiagCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for Golub-Kahan-Householder bidiagonalization (Bidiag.decomp /
    /// values). Allocate ONCE via Arena.fProxyBidiagCache(m, n) and reuse it across
    /// same-shape calls so repeated bidiagonalizations are zero-alloc.
    ///
    /// W (m x n) is the working copy of A reduced in place; leftU (m x n) stores the left reflectors
    /// for the U backward pass (used only by decomp, not values); uVec (m),
    /// vVec (n) are the Householder vectors and wScratch (n) is the apply-reflector scratch.
    /// </summary>
    public struct fProxyBidiagCache : IDisposable
    {
        public fProxyMxN W;
        public fProxyMxN leftU;
        public fProxyN uVec;
        public fProxyN vVec;
        public fProxyN wScratch;

        /// <summary>Standalone allocation sized identically to <c>Arena.fProxyBidiagCache(m, n)</c>. Pair with <see cref="Dispose"/>.</summary>
        public fProxyBidiagCache(int m, int n, Allocator allocator)
        {
            W = new fProxyMxN(m, n, allocator);
            leftU = new fProxyMxN(m, n, allocator);
            uVec = new fProxyN(m, allocator);
            vVec = new fProxyN(n, allocator);
            wScratch = new fProxyN(n, allocator);
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            W.Dispose();
            leftU.Dispose();
            uVec.Dispose();
            vVec.Dispose();
            wScratch.Dispose();
        }
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a bidiagonalization workspace for an m x n (m >= n) matrix. See
        /// <see cref="fProxyBidiagCache"/> for reuse guidance.
        /// </summary>
        public static fProxyBidiagCache fProxyBidiagCache(this ref Arena arena, int m, int n)
        {
            return new fProxyBidiagCache
            {
                W = arena.fProxyMat(m, n),
                leftU = arena.fProxyMat(m, n),
                uVec = arena.fProxyVec(m),
                vVec = arena.fProxyVec(n),
                wScratch = arena.fProxyVec(n)
            };
        }
    }
}
