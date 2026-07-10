using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables (docs/dev/rfc-memory-model.md §4 Option A), same
        // design as floatVecRecords/floatMatRecords (see Arena/floatRecords.float.cs,
        // Arena/Arena.float.cs) -- floatBSR/floatBlockJacobi now hold a stable
        // floatBSRRecord*/floatBlockJacobiRecord* into one of these instead of being tracked by
        // a separate value copy. No temp-pool counterpart (BSR has no isTemp/floatTempBSR
        // analogue). floatBSRBuilders stays on the OLD growable-list-of-value-copies model
        // DELIBERATELY: a builder's only mutable-relevant field is its heap-Malloc'd `State*`
        // (see floatBSRBuilder.cs), which is already pointer-stable and shared identically by
        // every value-copy -- there is no divergence risk (RFC failure mode 1) left to fix by
        // wrapping it in a second layer of record-table indirection, so migrating it would add
        // surface without closing a real bug.
        internal ChunkedRecordTable<floatBSRRecord> floatBSRRecords;
        internal UnsafeList<floatBSRBuilder> floatBSRBuilders;
        internal ChunkedRecordTable<floatBlockJacobiRecord> floatBlockJacobiRecords;
    }

    // Core bump-allocator primitives for the block-sparse matrix type, mirroring how
    // floatVec/floatMat are declared directly on the Arena struct (see Arena.float.cs).
    // Lifecycle wiring (AllocationsCount / Clear / Dispose) lives in Arena.cs / ArenaCore.
    public unsafe partial struct Arena
    {
        /// <summary>
        /// Allocates a compressed block-sparse (BSR) matrix with the given block-grid shape
        /// and stored-block capacity (nnzb). Typically produced by floatBSRBuilder.ToBSR
        /// rather than called directly. Arena-owned: disposed with the arena.
        /// </summary>
        public floatBSR floatBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, bool uninit = false, bool symmetric = false)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatBSR/floatBSRBuilder/floatBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatBSRRecord* rec = _core->floatBSRRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatBSRRecords;
                rec->SelfIndex = slot;
                return new floatBSR(blockRows, blockCols, BR, BC, nnzb, rec, Allocator, uninit, symmetric);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Allocates a COO-of-blocks assembly builder for a blockRows x blockCols grid of
        /// BR x BC blocks. Accumulate triplets via AddBlock/AddValue, then call ToBSR(arena)
        /// once to compress into a floatBSR. Arena-owned: disposed with the arena.
        /// </summary>
        public floatBSRBuilder floatBSRBuilder(int blockRows, int blockCols, int BR, int BC, int capacityHint = 8)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatBSR/floatBSRBuilder/floatBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                var builder = new floatBSRBuilder(blockRows, blockCols, BR, BC, in this, capacityHint);
                _core->floatBSRBuilders.Add(in builder);
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
        public floatBlockJacobi floatBlockJacobi(in floatBSR A)
        {
            if (_core == null)
                throw new System.InvalidOperationException("Arena.floatBSR/floatBSRBuilder/floatBlockJacobi: arena is not initialized (default or disposed).");
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            _core->EnterMutation();
            try
            {
#endif
                floatBlockJacobiRecord* rec = _core->floatBlockJacobiRecords.Allocate(out int slot);
                rec->Owner = _core;
                rec->Table = &_core->floatBlockJacobiRecords;
                rec->SelfIndex = slot;
                return new floatBlockJacobi(in A, rec, Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSR: every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place (BR x BC ->
        /// BC x BR), then re-compressed via <see cref="floatBSRBuilder"/> (the same
        /// triplet-sort/compress path as ToBSR). One-time O(nnz) cost -- the payoff is a
        /// cache-friendly FORWARD <see cref="BSR.spMV(in floatBSR, in floatN, ref floatN)"/>
        /// over A^T in place of the scatter-heavy on-the-fly <see cref="BSR.spMVT"/>
        /// traversal on every Krylov iteration -- see <see cref="floatBSROperator"/>'s two-arg
        /// constructor and the cgls/lsqr allocating <see cref="floatBSR"/> overloads in
        /// Krylov.float.cs, which build A^T once per solve and reuse it every iteration.
        ///
        /// If A.Symmetric (implies square, and A == A^T by construction -- see
        /// floatBSR.Symmetric), returns A itself unchanged: transposing symmetric upper-block
        /// storage is a no-op, and materializing a redundant copy would only double memory for
        /// zero benefit. This is safe to feed straight into floatBSROperator's two-arg ctor:
        /// BSR.spMV already special-cases Symmetric internally, exactly matching what
        /// spMVT itself does for a symmetric A (forwards straight to spMV).
        ///
        /// NOT itself guarded by the concurrency tripwire (docs/features/dense-types.md):
        /// this is a COMPOSITION of two already-guarded factory calls
        /// (<see cref="floatBSRBuilder(int,int,int,int,int)"/>, then
        /// <c>floatBSR</c> via <c>builder.ToBSR</c>) run sequentially, each fully entering and
        /// exiting its own guard before the next starts -- wrapping this method too would nest
        /// EnterMutation() on the same thread and trip the tripwire on ourselves.
        /// </summary>
        public unsafe floatBSR floatBSRTranspose(in floatBSR A)
        {
            if (A.Symmetric)
                return A;

            var builder = floatBSRBuilder(A.BlockCols, A.BlockRows, A.BC, A.BR, A.Nnzb);

            int blockLen = A.BR * A.BC;
            float* blockT = stackalloc float[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    float* block = A.Values.Ptr + k * blockLen;

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
        /// <see cref="floatBSR.ToDense"/> already computes on the fly (see that method's own
        /// Symmetric branch, which this mirrors exactly). O(nnzb*BR*BC), one-time cost -- the
        /// preconditioner setup this feeds (<see cref="floatSSOR"/>) amortizes it over the whole
        /// solve lifetime, same lifecycle as a factorization (Krylov R3, Q4 ruling: v1
        /// preconditioners are full-storage BSR only; symmetric-storage input pays this one-time
        /// copy instead of a bespoke symmetric-sweep kernel family -- MKL/Eigen practice matches).
        /// If A is already full storage (Symmetric == false), returns A unchanged -- no copy.
        /// Not itself guarded by the concurrency tripwire, for the same reason
        /// <see cref="floatBSRTranspose"/> is not -- see that method's own doc comment.
        /// </summary>
        public unsafe floatBSR floatBSRMirrorToFull(in floatBSR A)
        {
            if (!A.Symmetric)
                return A;

            var builder = floatBSRBuilder(A.BlockRows, A.BlockCols, A.BR, A.BC, A.Nnzb * 2);

            int blockLen = A.BR * A.BC;
            float* blockT = stackalloc float[blockLen];

            for (int bi = 0; bi < A.BlockRows; bi++)
            {
                int rowStart = A.RowPtr[bi];
                int rowEnd = A.RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = A.ColInd[k];
                    float* block = A.Values.Ptr + k * blockLen;

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
        /// diagonal-block inverses (<see cref="floatBlockJacobi"/>, reused unchanged) plus a
        /// one-time mirror-to-full pass if A is Symmetric-storage
        /// (<see cref="floatBSRMirrorToFull"/>). Arena-owned: disposed with the arena.
        /// </summary>
        public floatSSOR floatSSOR(in floatBSR A, float omega)
        {
            Arena self = this;
            return new floatSSOR(in A, omega, ref self);
        }

        /// <summary>floatSSOR with omega=1 (symmetric Gauss-Seidel).</summary>
        public floatSSOR floatSSOR(in floatBSR A)
        {
            Arena self = this;
            return new floatSSOR(in A, ref self);
        }
    }
}
