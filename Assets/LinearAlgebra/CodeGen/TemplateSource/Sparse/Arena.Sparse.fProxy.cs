using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        internal UnsafeList<fProxyBSM> fProxyBSMs;
        internal UnsafeList<fProxyBSMBuilder> fProxyBSMBuilders;
        internal UnsafeList<fProxyBlockJacobi> fProxyBlockJacobis;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // fProxyVec/fProxyMat are declared directly on the Arena struct (see Arena.fProxy.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by fProxyBSMBuilder.ToBSM
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBSM fProxyBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false, bool symmetric = false)
        {
            var mat = new fProxyBSM(blockRows, blockCols, BR, BC, nnzb, in this, uninit, symmetric);
            _core->fProxyBSMs.Add(in mat);
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
            _core->fProxyBSMBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBlockJacobi fProxyBlockJacobi(in fProxyBSM A)
        {
            var pc = new fProxyBlockJacobi(in A, in this);
            _core->fProxyBlockJacobis.Add(in pc);
            return pc;
        }
    }
}
