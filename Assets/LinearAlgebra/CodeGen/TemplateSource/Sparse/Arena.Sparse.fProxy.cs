using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A), same
        // design as fProxyVecRecords/fProxyMatRecords (see Arena/fProxyRecords.fProxy.cs,
        // Arena/Arena.fProxy.cs) -- fProxyBSR/fProxyBlockJacobi now hold a stable
        // fProxyBSRRecord*/fProxyBlockJacobiRecord* into one of these instead of being tracked by
        // a separate value copy. No temp-pool counterpart (BSR has no isTemp/fProxyTempBSR
        // analogue). fProxyBSRBuilders stays on the OLD growable-list-of-value-copies model
        // DELIBERATELY: a builder's only mutable-relevant field is its heap-Malloc'd `State*`
        // (see fProxyBSRBuilder.cs), which is already pointer-stable and shared identically by
        // every value-copy -- there is no divergence risk (RFC failure mode 1) left to fix by
        // wrapping it in a second layer of record-table indirection, so migrating it would add
        // surface without closing a real bug.
        internal ChunkedRecordTable<fProxyBSRRecord> fProxyBSRRecords;
        internal UnsafeList<fProxyBSRBuilder> fProxyBSRBuilders;
        internal ChunkedRecordTable<fProxyBlockJacobiRecord> fProxyBlockJacobiRecords;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // fProxyVec/fProxyMat are declared directly on the Arena struct (see Arena.fProxy.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by fProxyBSRBuilder.ToBSR
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBSR fProxyBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false, bool symmetric = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyBSR/fProxyBSRBuilder/fProxyBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyBSRRecord* rec = _core->fProxyBSRRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyBSRRecords;
                rec->SelfIndex = slot;
                return new fProxyBSR(blockRows, blockCols, BR, BC, nnzb, rec, Allocator, uninit, symmetric);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSR(arena)
        /// once to compress into a fProxyBSR. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBSRBuilder fProxyBSRBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyBSR/fProxyBSRBuilder/fProxyBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                var builder = new fProxyBSRBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
                _core->fProxyBSRBuilders.Add(in builder);
                return builder;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyBlockJacobi fProxyBlockJacobi(in fProxyBSR A)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.fProxyBSR/fProxyBSRBuilder/fProxyBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                fProxyBlockJacobiRecord* rec = _core->fProxyBlockJacobiRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->fProxyBlockJacobiRecords;
                rec->SelfIndex = slot;
                return new fProxyBlockJacobi(in A, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSR: every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place (BR x BC ->
        /// BC x BR), then re-compressed via <see cref="fProxyBSRBuilder"/> (the same
        /// triplet-sort/compress path as ToBSR). One-time O(nnz) cost -- the payoff is a
        /// cache-friendly FORWARD <see cref="BSR.spMV(in fProxyBSR, in fProxyN, ref fProxyN)"/>
        /// over A^T in place of the scatter-heavy on-the-fly <see cref="BSR.spMVT"/>
        /// traversal on every Krylov iteration -- see <see cref="fProxyBSROperator"/>'s two-arg
        /// constructor and the cgls/lsqr allocating <see cref="fProxyBSR"/> overloads in
        /// Solvers.fProxy.cs, which build A^T once per solve and reuse it every iteration.
        ///
        /// If A.Symmetric (implies square, and A == A^T by construction -- see
        /// fProxyBSR.Symmetric), returns A itself unchanged: transposing symmetric upper-block
        /// storage is a no-op, and materializing a redundant copy would only double memory for
        /// zero benefit. This is safe to feed straight into fProxyBSROperator's two-arg ctor:
        /// BSR.spMV already special-cases Symmetric internally, exactly matching what
        /// spMVT itself does for a symmetric A (forwards straight to spMV).
        ///
        /// NOT itself guarded by the concurrency tripwire (docs/features/dense-types.md):
        /// this is a COMPOSITION of two already-guarded factory calls
        /// (<see cref="fProxyBSRBuilder(int,int,int,int,int)"/>, then
        /// <c>fProxyBSR</c> via <c>builder.ToBSR</c>) run sequentially, each fully entering and
        /// exiting its own guard before the next starts -- wrapping this method too would nest
        /// EnterMutation() on the same thread and trip the tripwire on ourselves.
        /// </summary>
        public unsafe fProxyBSR fProxyBSRTranspose(in fProxyBSR A)
        {
            if (A.Symmetric)
                return A;

            var builder = fProxyBSRBuilder(A.BlockCols, A.BlockRows, A.BC, A.BR, A.Nnzb);

            int blockLen = A.BR * A.BC;
            fProxy* blockT = stackalloc fProxy[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    fProxy* block = A.Values.Ptr + k * blockLen;

                    // Transpose the BR x BC block into blockT (BC x BR, row-major): blockT's
                    // row c, col r == block's row r, col c.
                    for (int r = 0; r < A.BR; r++)
                        for (int c = 0; c < A.BC; c++)
                            blockT[c * A.BR + r] = block[r * A.BC + c];

                    builder.AddBlock(bj, bi, blockT);
                }
            }

            Arena self = this;
            return builder.ToBSR(ref self);
        }
    }
}
