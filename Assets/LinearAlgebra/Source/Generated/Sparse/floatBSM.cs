using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Block-CSR (BSR) sparse matrix: a uniform grid of BlockRows x BlockCols dense blocks,
    /// each BR x BC, stored compressed like CSR but with one column index PER BLOCK instead
    /// of per scalar (RowPtr/ColInd/Values -- "CSR of dense blocks").
    ///
    /// Block interior layout is ROW-MAJOR: block k's entry (r, c) lives at
    /// Values[k*BR*BC + r*BC + c] -- matching the library's row-major dense convention
    /// (floatMxN.Data[r*N_Cols+c]). Do not mix layouts. Blocks within a block-row are stored
    /// in ascending ColInd (enables transpose-SpMV and future binary-search block lookup).
    ///
    /// Logical scalar dims: M_Rows = BlockRows*BR, N_Cols = BlockCols*BC. Rectangular blocks
    /// (BR != BC) are supported. Phase 1 ships FULL storage only -- no symmetric-upper-block
    /// mode yet (see docs/spec-sparse-bsm.md).
    ///
    /// Lifecycle: build via floatBSMBuilder.ToBSM(arena). This type is the compressed,
    /// matvec-ready form -- there is no cheap incremental pattern edit after compression; go
    /// back through the builder to add/remove blocks.
    /// </summary>
    public partial struct floatBSM : IDisposable
    {
        public int BlockRows;  // mb: number of block-rows
        public int BlockCols;  // nb: number of block-cols
        public int BR;         // rows per block
        public int BC;         // cols per block

        public int M_Rows => BlockRows * BR;
        public int N_Cols => BlockCols * BC;

        /// <summary>Number of stored (nonzero) blocks.</summary>
        public int Nnzb => ColInd.Length;

        // CSR-of-blocks index structure (arena-owned UnsafeLists):
        public UnsafeList<int> RowPtr;     // length BlockRows+1
        public UnsafeList<int> ColInd;     // length nnzb (block-column of each stored block)
        public UnsafeList<float> Values;  // length nnzb*BR*BC (flat, row-major per block)

        [NativeDisableUnsafePtrRestriction]
        private unsafe Arena* _arenaPtr;

        /// <summary>
        /// Allocates a compressed BSR matrix with the given block-grid shape and a fixed
        /// number of stored blocks (nnzb). Typically produced by floatBSMBuilder.ToBSM
        /// rather than called directly -- the caller is expected to fill RowPtr/ColInd/Values.
        /// </summary>
        public unsafe floatBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, Allocator allocator, bool uninit = false)
        {
            _arenaPtr = null;
            BlockRows = blockRows;
            BlockCols = blockCols;
            this.BR = BR;
            this.BC = BC;

            var options = uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory;

            var rowPtr = new UnsafeList<int>(blockRows + 1, allocator, options);
            rowPtr.Resize(blockRows + 1, options);
            RowPtr = rowPtr;

            var colInd = new UnsafeList<int>(nnzb, allocator, options);
            colInd.Resize(nnzb, options);
            ColInd = colInd;

            int valuesLen = nnzb * BR * BC;
            var values = new UnsafeList<float>(valuesLen, allocator, options);
            values.Resize(valuesLen, options);
            Values = values;
        }

        /// <summary>
        /// Creates a new BSR matrix of the given shape from an arena. Same allocation shape
        /// as the Allocator overload, but tracked by the arena for disposal.
        /// </summary>
        public unsafe floatBSM(int blockRows, int blockCols, int BR, int BC, int nnzb, in Arena arena, bool uninit = false)
        {
            fixed (Arena* arenaPtr = &arena)
                _arenaPtr = arenaPtr;

            BlockRows = blockRows;
            BlockCols = blockCols;
            this.BR = BR;
            this.BC = BC;

            var allocator = arena.Allocator;
            var options = uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory;

            var rowPtr = new UnsafeList<int>(blockRows + 1, allocator, options);
            rowPtr.Resize(blockRows + 1, options);
            RowPtr = rowPtr;

            var colInd = new UnsafeList<int>(nnzb, allocator, options);
            colInd.Resize(nnzb, options);
            ColInd = colInd;

            int valuesLen = nnzb * BR * BC;
            var values = new UnsafeList<float>(valuesLen, allocator, options);
            values.Resize(valuesLen, options);
            Values = values;
        }

        public void Dispose()
        {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < Values.Length; i++) Values[i] = float.NaN;
#endif
            RowPtr.Dispose();
            ColInd.Dispose();
            Values.Dispose();
        }

        /// <summary>
        /// Expands this BSM to a dense M_Rows x N_Cols matrix: zero-filled, then every stored
        /// block scattered into place. Used by tests and as a general-purpose densify helper.
        ///
        /// Takes the arena by `ref`, NOT `in`: it calls the mutating arena.floatMat(...)
        /// allocator internally, and `Arena`'s allocator methods are not `readonly` -- an `in`
        /// parameter here would force the compiler to defensively copy the arena before that
        /// call, so the returned matrix's internal arena pointer would capture the address of a
        /// dead temporary instead of the caller's real arena (a dangling-pointer bug caught by
        /// the test suite). Same reasoning as why ArenaExtensions factory methods take
        /// `this ref Arena`, not `this in Arena`.
        /// </summary>
        public floatMxN ToDense(ref Arena arena)
        {
            var dense = arena.floatMat(M_Rows, N_Cols); // zero-initialized

            for (int br = 0; br < BlockRows; br++)
            {
                int rowStart = RowPtr[br];
                int rowEnd = RowPtr[br + 1];
                int baseRow = br * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = ColInd[k];
                    int baseCol = bc * BC;
                    int blockOffset = k * BR * BC;

                    for (int r = 0; r < BR; r++)
                    {
                        for (int c = 0; c < BC; c++)
                        {
                            dense[baseRow + r, baseCol + c] = Values[blockOffset + r * BC + c];
                        }
                    }
                }
            }

            return dense;
        }
    }
}
