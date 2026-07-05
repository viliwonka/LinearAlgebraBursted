using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/rfc-memory-model.md §4 Option A), same
        // design as doubleVecRecords/doubleMatRecords (see Arena/doubleRecords.double.cs,
        // Arena/Arena.double.cs) -- doubleBSR/doubleBlockJacobi now hold a stable
        // doubleBSRRecord*/doubleBlockJacobiRecord* into one of these instead of being tracked by
        // a separate value copy. No temp-pool counterpart (BSR has no isTemp/doubleTempBSR
        // analogue). doubleBSRBuilders stays on the OLD growable-list-of-value-copies model
        // DELIBERATELY: a builder's only mutable-relevant field is its heap-Malloc'd `State*`
        // (see doubleBSRBuilder.cs), which is already pointer-stable and shared identically by
        // every value-copy -- there is no divergence risk (RFC failure mode 1) left to fix by
        // wrapping it in a second layer of record-table indirection, so migrating it would add
        // surface without closing a real bug.
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
            doubleBSRRecord* rec = _core->doubleBSRRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->doubleBSRRecords;
            rec->SelfIndex = slot;
            return new doubleBSR(blockRows, blockCols, BR, BC, nnzb, rec, Allocator, uninit, symmetric);
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSR(arena)
        /// once to compress into a doubleBSR. Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBSRBuilder doubleBSRBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            var builder = new doubleBSRBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
            _core->doubleBSRBuilders.Add(in builder);
            return builder;
        }

        /// <summary>
        /// Builds a block-Jacobi preconditioner from A's diagonal blocks (A must be square:
        /// BlockRows==BlockCols, BR==BC). Arena-owned: disposed with the arena.
        /// </summary>
        public doubleBlockJacobi doubleBlockJacobi(in doubleBSR A)
        {
            doubleBlockJacobiRecord* rec = _core->doubleBlockJacobiRecords.Allocate(out int slot);
            rec->Owner = _core;
            rec->Table = &_core->doubleBlockJacobiRecords;
            rec->SelfIndex = slot;
            return new doubleBlockJacobi(in A, rec, Allocator);
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSR: every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place (BR x BC ->
        /// BC x BR), then re-compressed via <see cref="doubleBSRBuilder"/> (the same
        /// triplet-sort/compress path as ToBSR). One-time O(nnz) cost -- the payoff is a
        /// cache-friendly FORWARD <see cref="BSR.spMV(in doubleBSR, in doubleN, ref doubleN)"/>
        /// over A^T in place of the scatter-heavy on-the-fly <see cref="BSR.spMVT"/>
        /// traversal on every Krylov iteration -- see <see cref="doubleBSROperator"/>'s two-arg
        /// constructor and the cgls/lsqr allocating <see cref="doubleBSR"/> overloads in
        /// Solvers.double.cs, which build A^T once per solve and reuse it every iteration.
        ///
        /// If A.Symmetric (implies square, and A == A^T by construction -- see
        /// doubleBSR.Symmetric), returns A itself unchanged: transposing symmetric upper-block
        /// storage is a no-op, and materializing a redundant copy would only double memory for
        /// zero benefit. This is safe to feed straight into doubleBSROperator's two-arg ctor:
        /// BSR.spMV already special-cases Symmetric internally, exactly matching what
        /// spMVT itself does for a symmetric A (forwards straight to spMV).
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
    }
}
