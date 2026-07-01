using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // doubleVec/doubleMat are declared directly on the Arena struct (see Arena.double.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs.
    public partial struct Arena
    {
        private UnsafeList<doubleBSM> doubleBSMs;
        private UnsafeList<doubleBSMBuilder> doubleBSMBuilders;
        private UnsafeList<doubleBlockJacobi> doubleBlockJacobis;

        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by doubleBSMBuilder.ToBSM
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSM doubleBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false)
        {
            var mat = new doubleBSM(blockRows, blockCols, BR, BC, nnzb, in this, uninit);
            doubleBSMs.Add(in mat);
            return mat;
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSM(arena)
        /// once to compress into a doubleBSM. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSMBuilder doubleBSMBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            var builder = new doubleBSMBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
            doubleBSMBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        ///
        /// Takes A by `in` (read-only source) but is itself called on a mutable arena
        /// receiver -- an instance method on `partial struct Arena`, same as doubleBSM/
        /// doubleBSMBuilder above, so `this` is the caller's real arena, not a defensive copy
        /// (the "ref Arena" lesson from doubleBSM.ToDense/doubleBSMBuilder.ToBSM applies to
        /// extension-method-shaped factories; here it is naturally satisfied because the
        /// factory lives directly on Arena).
        /// </summary>
        public doubleBlockJacobi doubleBlockJacobi(in doubleBSM A)
        {
            var pc = new doubleBlockJacobi(in A, in this);
            doubleBlockJacobis.Add(in pc);
            return pc;
        }
    }
}
