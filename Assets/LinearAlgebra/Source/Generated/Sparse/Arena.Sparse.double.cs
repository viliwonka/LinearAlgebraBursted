using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables, same design as doubleVecRecords/doubleMatRecords
        // (see Arena/doubleRecords.double.cs, Arena/Arena.double.cs) -- doubleBSR/doubleBlockJacobi
        // hold a stable doubleBSRRecord*/doubleBlockJacobiRecord* into one of these instead of being
        // tracked by a separate value copy. No temp-pool counterpart (BSR has no isTemp/doubleTempBSR
        // analogue). doubleBSRBuilders stays on the growable-list-of-value-copies model: a builder's
        // only mutable-relevant field is its heap-Malloc'd `State*` (see doubleBSRBuilder.cs), which
        // is already pointer-stable and shared identically by every value-copy.
        internal ChunkedRecordTable<doubleBSRRecord> doubleBSRRecords;
        internal UnsafeList<doubleBSRBuilder> doubleBSRBuilders;
        internal ChunkedRecordTable<doubleBlockJacobiRecord> doubleBlockJacobiRecords;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // doubleVec/doubleMat are declared directly on the Arena struct (see Arena.double.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by doubleBSRBuilder.ToBSR
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSR doubleBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false, bool symmetric = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleBSR/doubleBSRBuilder/doubleBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleBSRRecord* rec = _core->doubleBSRRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleBSRRecords;
                rec->SelfIndex = slot;
                return new doubleBSR(blockRows, blockCols, BR, BC, nnzb, rec, Allocator, uninit, symmetric);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSR(arena)
        /// once to compress into a doubleBSR. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSRBuilder doubleBSRBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleBSR/doubleBSRBuilder/doubleBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                var builder = new doubleBSRBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
                _core->doubleBSRBuilders.Add(in builder);
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
        public doubleBlockJacobi doubleBlockJacobi(in doubleBSR A)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.doubleBSR/doubleBSRBuilder/doubleBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                doubleBlockJacobiRecord* rec = _core->doubleBlockJacobiRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->doubleBlockJacobiRecords;
                rec->SelfIndex = slot;
                return new doubleBlockJacobi(in A, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSR (O(nnz)): every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place, then
        /// re-compressed via <see cref="doubleBSRBuilder"/>. If A.Symmetric, returns A itself
        /// unchanged (transposing symmetric upper-block storage is a no-op).
        /// </summary>
        public unsafe doubleBSR doubleBSRTranspose(in doubleBSR A)
        {
            if (A.Symmetric)
                return A;

            var builder = doubleBSRBuilder(A.BlockCols, A.BlockRows, A.BC, A.BR, A.Nnzb);

            int blockLen = A.BR * A.BC;
            double* blockT = stackalloc double[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    double* block = A.Values.Ptr + k * blockLen;

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

        /// <summary>
        /// One-time mirror of a SYMMETRIC-storage (upper-block-triangle-only) BSR into an
        /// equivalent FULL-storage BSR: every stored block K at (bi,bj) is kept at (bi,bj), and
        /// if bi != bj its transpose is ALSO materialized at (bj,bi) -- the implicit lower block
        /// <see cref="doubleBSR.ToDense"/> already computes on the fly. O(nnzb*BR*BC), one-time
        /// copy. If A is already full storage (Symmetric == false), returns A unchanged -- no copy.
        /// </summary>
        public unsafe doubleBSR doubleBSRMirrorToFull(in doubleBSR A)
        {
            if (!A.Symmetric)
                return A;

            var builder = doubleBSRBuilder(A.BlockRows, A.BlockCols, A.BR, A.BC, A.Nnzb * 2);

            int blockLen = A.BR * A.BC;
            double* blockT = stackalloc double[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    double* block = A.Values.Ptr + k * blockLen;

                    builder.AddBlock(bi, bj, block);

                    if (bi != bj)
                    {
                        for (int r = 0; r < A.BR; r++)
                            for (int c = 0; c < A.BC; c++)
                                blockT[c * A.BR + r] = block[r * A.BC + c];
                        builder.AddBlock(bj, bi, blockT);
                    }
                }
            }

            Arena self = this;
            return builder.ToBSR(ref self);
        }

        /// <summary>
        /// Builds a symmetric-SOR preconditioner from A (must be square: BlockRows==BlockCols,
        /// BR==BC), with the given relaxation parameter omega in (0, 2). Setup is A's own
        /// diagonal-block inverses (<see cref="doubleBlockJacobi"/>, reused unchanged) plus a
        /// one-time mirror-to-full pass if A is Symmetric-storage
        /// (<see cref="doubleBSRMirrorToFull"/>). Arena-owned: disposed with the arena.
        /// </summary>
        public doubleSSOR doubleSSOR(in doubleBSR A, double omega)
        {
            Arena self = this;
            return new doubleSSOR(in A, omega, ref self);
        }

        /// <summary>doubleSSOR with omega=1 (symmetric Gauss-Seidel).</summary>
        public doubleSSOR doubleSSOR(in doubleBSR A)
        {
            Arena self = this;
            return new doubleSSOR(in A, ref self);
        }
    }
}
