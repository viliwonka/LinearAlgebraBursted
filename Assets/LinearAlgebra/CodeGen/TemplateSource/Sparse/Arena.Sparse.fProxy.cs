using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // fProxyVec/fProxyMat are declared directly on the Arena struct (see Arena.fProxy.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs.
    public partial struct Arena
    {
        private UnsafeList<fProxyBSM> fProxyBSMs;
        private UnsafeList<fProxyBSMBuilder> fProxyBSMBuilders;
        private UnsafeList<fProxyBlockJacobi> fProxyBlockJacobis;

        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by fProxyBSMBuilder.ToBSM
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBSM fProxyBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false)
        {
            var mat = new fProxyBSM(blockRows, blockCols, BR, BC, nnzb, in this, uninit);
            fProxyBSMs.Add(in mat);
            return mat;
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSM(arena)
        /// once to compress into a fProxyBSM. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBSMBuilder fProxyBSMBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            var builder = new fProxyBSMBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
            fProxyBSMBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        ///
        /// Takes A by `in` (read-only source) but is itself called on a mutable arena
        /// receiver -- an instance method on `partial struct Arena`, same as fProxyBSM/
        /// fProxyBSMBuilder above, so `this` is the caller's real arena, not a defensive copy
        /// (the "ref Arena" lesson from fProxyBSM.ToDense/fProxyBSMBuilder.ToBSM applies to
        /// extension-method-shaped factories; here it is naturally satisfied because the
        /// factory lives directly on Arena).
        /// </summary>
        public fProxyBlockJacobi fProxyBlockJacobi(in fProxyBSM A)
        {
            var pc = new fProxyBlockJacobi(in A, in this);
            fProxyBlockJacobis.Add(in pc);
            return pc;
        }
    }
}
