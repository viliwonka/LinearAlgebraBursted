using System;

namespace LinearAlgebra
{
    public static partial class Bidiag
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n bidiagonalization. The common
        /// buffers (W m x n, uVec m, vVec n, wScratch n) are always required; leftU (m x n) is required
        /// only by the full <see cref="bidiagonalize"/> (which reconstructs U), not by
        /// <see cref="bidiagonalizeValues"/>. Matches Arena.doubleBidiag_WS(m, n).
        /// </summary>
        static void RequireBidiagWorkspace(in doubleBidiag_WS ws, int m, int n, bool needLeftU)
        {
            bool ok =
                ws.W.M_Rows == m && ws.W.N_Cols == n &&
                ws.uVec.N == m &&
                ws.vVec.N == n &&
                ws.wScratch.N == n &&
                (!needLeftU || (ws.leftU.M_Rows == m && ws.leftU.N_Cols == n));

            if (!ok)
                throw new ArgumentException("Bidiag: workspace must be sized for m x n (use Arena.doubleBidiag_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for Golub-Kahan-Householder bidiagonalization (Bidiag.bidiagonalize /
    /// bidiagonalizeValues). Allocate ONCE via Arena.doubleBidiag_WS(m, n) and reuse it across
    /// same-shape calls so repeated bidiagonalizations are zero-alloc.
    ///
    /// W (m x n) is the working copy of A reduced in place; leftU (m x n) stores the left reflectors
    /// for the U backward pass (used only by bidiagonalize, not bidiagonalizeValues); uVec (m),
    /// vVec (n) are the Householder vectors and wScratch (n) is the apply-reflector scratch.
    /// </summary>
    public struct doubleBidiag_WS
    {
        public doubleMxN W;
        public doubleMxN leftU;
        public doubleN uVec;
        public doubleN vVec;
        public doubleN wScratch;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a bidiagonalization workspace for an m x n (m >= n) matrix: W (m x n),
        /// leftU (m x n), uVec (m), vVec (n), wScratch (n). The buffers are persistent in this arena
        /// (disposed with it), so create the workspace once outside a hot loop and pass it to the
        /// ref-workspace overloads of bidiagonalize / bidiagonalizeValues.
        /// </summary>
        public static doubleBidiag_WS doubleBidiag_WS(this ref Arena arena, int m, int n)
        {
            return new doubleBidiag_WS
            {
                W = arena.doubleMat(m, n),
                leftU = arena.doubleMat(m, n),
                uVec = arena.doubleVec(m),
                vVec = arena.doubleVec(n),
                wScratch = arena.doubleVec(n)
            };
        }
    }
}
