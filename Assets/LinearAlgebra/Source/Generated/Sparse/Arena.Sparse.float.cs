using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<floatBSM> floatBSMs;
        internal UnsafeList<floatBSMBuilder> floatBSMBuilders;
        internal UnsafeList<floatBlockJacobi> floatBlockJacobis;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // floatVec/floatMat are declared directly on the Arena struct (see Arena.float.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by floatBSMBuilder.ToBSM
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public floatBSM floatBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false)
        {
            var mat = new floatBSM(blockRows, blockCols, BR, BC, nnzb, in this, uninit);
            _core->floatBSMs.Add(in mat);
            return mat;
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSM(arena)
        /// once to compress into a floatBSM. Arena-owned: disposed with the arena.
        /// </summary>
        public floatBSMBuilder floatBSMBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            var builder = new floatBSMBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
            _core->floatBSMBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        /// </summary>
        public floatBlockJacobi floatBlockJacobi(in floatBSM A)
        {
            var pc = new floatBlockJacobi(in A, in this);
            _core->floatBlockJacobis.Add(in pc);
            return pc;
        }
    }
}
