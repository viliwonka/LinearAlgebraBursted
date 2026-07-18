using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal partial struct ArenaCore
    {
        // Pointer-stable allocation-record tables, same design as fProxyVecRecords/fProxyMatRecords
        // (see Arena/fProxyRecords.fProxy.cs, Arena/Arena.fProxy.cs) -- fProxyBSR/fProxyBlockJacobi
        // hold a stable fProxyBSRRecord*/fProxyBlockJacobiRecord* into one of these instead of being
        // tracked by a separate value copy. No temp-pool counterpart (BSR has no isTemp/fProxyTempBSR
        // analogue). fProxyBSRBuilders stays on the growable-list-of-value-copies model: a builder's
        // only mutable-relevant field is its heap-Malloc'd `State*` (see fProxyBSRBuilder.cs), which
        // is already pointer-stable and shared identically by every value-copy.
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
        /// Non-throwing twin of <see cref="fProxyBlockJacobi(in fProxyBSR)"/>: info carries the
        /// build outcome (see the fProxyBlockJacobi out-info constructor). On failure the returned
        /// struct is unusable and no arena record is retained.
        /// </summary>
        public fProxyBlockJacobi fProxyBlockJacobi(in fProxyBSR A, out PreconditionerInfo info)
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
                var M = new fProxyBlockJacobi(in A, rec, Allocator, out info);
                if (!info.Solved)
                    _core->fProxyBlockJacobiRecords.Free(slot);
                return M;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            }
            finally { _core->ExitMutation(); }
#endif
        }

        /// <summary>
        /// Materializes A^T as its own compressed BSR (O(nnz)): every stored block at (blockRow,
        /// blockCol) becomes a block at (blockCol, blockRow), transposed in place, then
        /// re-compressed via <see cref="fProxyBSRBuilder"/>. If A.Symmetric, returns A itself
        /// unchanged (transposing symmetric lower-block storage is a no-op).
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

        /// <summary>
        /// One-time mirror of a SYMMETRIC-storage (lower-block-triangle-only) BSR into an
        /// equivalent FULL-storage BSR: every stored block K at (bi,bj) is kept at (bi,bj), and
        /// if bi != bj its transpose is ALSO materialized at (bj,bi) -- the implicit upper block
        /// <see cref="fProxyBSR.ToDense"/> already computes on the fly. O(nnzb*BR*BC), one-time
        /// copy. If A is already full storage (Symmetric == false), returns A unchanged -- no copy.
        /// </summary>
        public unsafe fProxyBSR fProxyBSRMirrorToFull(in fProxyBSR A)
        {
            if (!A.Symmetric)
                return A;

            var builder = fProxyBSRBuilder(A.BlockRows, A.BlockCols, A.BR, A.BC, A.Nnzb * 2);

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
        /// diagonal-block inverses (<see cref="fProxyBlockJacobi"/>, reused unchanged) plus a
        /// one-time mirror-to-full pass if A is Symmetric-storage
        /// (<see cref="fProxyBSRMirrorToFull"/>). Arena-owned: disposed with the arena.
        /// </summary>
        public fProxySSOR fProxySSOR(in fProxyBSR A, fProxy omega)
        {
            Arena self = this;
            return new fProxySSOR(in A, omega, ref self);
        }

        /// <summary>fProxySSOR with omega=1 (symmetric Gauss-Seidel).</summary>
        public fProxySSOR fProxySSOR(in fProxyBSR A)
        {
            Arena self = this;
            return new fProxySSOR(in A, ref self);
        }

        /// <summary>
        /// Builds a Chebyshev polynomial preconditioner from A (must be square SPD: every scalar
        /// diagonal entry &gt; 0, every diagonal block stored; Symmetric-storage A is consumed
        /// directly, no mirror needed). See <see cref="fProxyChebyshev"/> for the setup/Apply
        /// contract and <paramref name="opt"/>'s field docs for the tunable defaults. Arena-owned:
        /// disposed with the arena.
        /// </summary>
        public fProxyChebyshev fProxyChebyshev(in fProxyBSR A, in fProxyChebyshevOptions opt)
        {
            Arena self = this;
            return new fProxyChebyshev(in A, in opt, ref self);
        }

        /// <summary>fProxyChebyshev with fProxyChebyshevOptions.Default (degree=3, kappa=30, eigSteps=10, safety=1.1).</summary>
        public fProxyChebyshev fProxyChebyshev(in fProxyBSR A)
        {
            Arena self = this;
            return new fProxyChebyshev(in A, ref self);
        }

        /// <summary>
        /// Builds a block incomplete-Cholesky IC(0) preconditioner from A (must be square SPD
        /// with every diagonal block stored; Symmetric-storage A is consumed zero-copy -- its
        /// stored lower-block pattern IS the IC(0) pattern). See <see cref="fProxyIC0"/> for the
        /// breakdown/diagonal-shift contract. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyIC0 fProxyIC0(in fProxyBSR A)
        {
            Arena self = this;
            return new fProxyIC0(in A, ref self);
        }

        /// <summary>Non-throwing twin of <see cref="fProxyIC0(in fProxyBSR)"/>: info carries the
        /// build outcome (Success, or NotPositiveDefinite on factorization breakdown).</summary>
        public fProxyIC0 fProxyIC0(in fProxyBSR A, out PreconditionerInfo info)
        {
            Arena self = this;
            return new fProxyIC0(in A, ref self, out info);
        }

        /// <summary>
        /// Builds a block incomplete-LU ILU(0) preconditioner from A (square, every diagonal
        /// block stored) — the nonsymmetric sibling of <see cref="fProxyIC0"/>, for
        /// Krylov.pbiCGStab. See <see cref="fProxyILU0"/> for the breakdown/shift contract.
        /// Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyILU0 fProxyILU0(in fProxyBSR A)
        {
            Arena self = this;
            return new fProxyILU0(in A, ref self);
        }

        /// <summary>Non-throwing twin of <see cref="fProxyILU0(in fProxyBSR)"/>: info carries the
        /// build outcome (Success, or Singular on factorization breakdown).</summary>
        public fProxyILU0 fProxyILU0(in fProxyBSR A, out PreconditionerInfo info)
        {
            Arena self = this;
            return new fProxyILU0(in A, ref self, out info);
        }

        /// <summary>
        /// Builds a factored sparse approximate inverse (FSAI) preconditioner from A (must be
        /// square SPD with every diagonal block stored; Symmetric-storage A is consumed zero-copy).
        /// Uses <see cref="SaiOptions.Default"/>. See <see cref="fProxyFSAI"/> for the
        /// breakdown/diagonal-shift contract. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxyFSAI fProxyFSAI(in fProxyBSR A)
        {
            Arena self = this;
            return new fProxyFSAI(in A, ref self);
        }

        /// <summary>Non-throwing twin of <see cref="fProxyFSAI(in fProxyBSR)"/>: info carries the
        /// build outcome (Success, or NotPositiveDefinite on factorization breakdown).</summary>
        public fProxyFSAI fProxyFSAI(in fProxyBSR A, out PreconditionerInfo info)
        {
            Arena self = this;
            return new fProxyFSAI(in A, ref self, out info);
        }

        /// <summary>Same as <see cref="fProxyFSAI(in fProxyBSR)"/> with explicit <see cref="SaiOptions"/>.</summary>
        public fProxyFSAI fProxyFSAI(in fProxyBSR A, in SaiOptions opts)
        {
            Arena self = this;
            return new fProxyFSAI(in A, ref self, in opts);
        }

        /// <summary>Non-throwing twin of <see cref="fProxyFSAI(in fProxyBSR, in SaiOptions)"/>.</summary>
        public fProxyFSAI fProxyFSAI(in fProxyBSR A, in SaiOptions opts, out PreconditionerInfo info)
        {
            Arena self = this;
            return new fProxyFSAI(in A, ref self, in opts, out info);
        }

        /// <summary>
        /// Builds a row-oriented sparse approximate inverse (SPAI) preconditioner from A (square,
        /// every diagonal block stored) -- the nonsymmetric sibling of <see cref="fProxyFSAI"/>,
        /// for Krylov.pbiCGStab. Uses <see cref="SaiOptions.Default"/>. See <see cref="fProxySPAI"/>
        /// for the breakdown/shift contract. Arena-owned: disposed with the arena.
        /// </summary>
        public fProxySPAI fProxySPAI(in fProxyBSR A)
        {
            Arena self = this;
            return new fProxySPAI(in A, ref self);
        }

        /// <summary>Non-throwing twin of <see cref="fProxySPAI(in fProxyBSR)"/>: info carries the
        /// build outcome (Success, or Singular on factorization breakdown).</summary>
        public fProxySPAI fProxySPAI(in fProxyBSR A, out PreconditionerInfo info)
        {
            Arena self = this;
            return new fProxySPAI(in A, ref self, out info);
        }

        /// <summary>Same as <see cref="fProxySPAI(in fProxyBSR)"/> with explicit <see cref="SaiOptions"/>.</summary>
        public fProxySPAI fProxySPAI(in fProxyBSR A, in SaiOptions opts)
        {
            Arena self = this;
            return new fProxySPAI(in A, ref self, in opts);
        }

        /// <summary>Non-throwing twin of <see cref="fProxySPAI(in fProxyBSR, in SaiOptions)"/>.</summary>
        public fProxySPAI fProxySPAI(in fProxyBSR A, in SaiOptions opts, out PreconditionerInfo info)
        {
            Arena self = this;
            return new fProxySPAI(in A, ref self, in opts, out info);
        }
    }
}
