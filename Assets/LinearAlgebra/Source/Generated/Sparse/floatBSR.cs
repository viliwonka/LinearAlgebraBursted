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
    /// (BR != BC) are supported. Set Symmetric=true to opt into upper-block-triangle-only
    /// storage (halves memory and single-threaded matvec FLOPs for symmetric matrices) --
    /// requires BR==BC and a square block grid (BlockRows==BlockCols). See spec-sparse-bsm.md
    /// §2.3.
    ///
    /// Lifecycle: build via floatBSRBuilder.ToBSR(arena). This type is the compressed,
    /// matvec-ready form -- there is no cheap incremental pattern edit after compression; go
    /// back through the builder to add/remove blocks.
    /// </summary>
    public partial struct floatBSR : IDisposable
    {
        public int BlockRows;  // mb: number of block-rows
        public int BlockCols;  // nb: number of block-cols
        public int BR;         // rows per block
        public int BC;         // cols per block

        public bool Symmetric;  // true => only the upper block-triangle (ColInd >= blockRow) is stored

        public int M_Rows => BlockRows * BR;
        public int N_Cols => BlockCols * BC;

        /// <summary>Number of stored (nonzero) blocks.</summary>
        public int Nnzb => ColInd.Length;

        // Arena-tracked path: a stable pointer into the arena's ChunkedRecordTable<floatBSRRecord>
        // (docs/rfc-memory-model.md §4 Option A). null for a standalone (non-arena) matrix, in which
        // case RowPtr/ColInd/Values resolve to the inline fields below instead -- see those
        // properties. Replaces the old `Arena _arena` handle field: retiring it keeps this struct's
        // size unchanged (both are a single pointer-width field), and the record's own `Owner`
        // back-pointer is where a future Copy()/cross-type shortcut would resolve through instead.
        [NativeDisableUnsafePtrRestriction] private unsafe floatBSRRecord* _rec;

        // Standalone-path backing store -- the ONLY thing that changes for a non-arena matrix.
        // Stay default(UnsafeList<...>) whenever _rec != null (arena-tracked).
        private UnsafeList<int> _inlineRowPtr;
        private UnsafeList<int> _inlineColInd;
        private UnsafeList<float> _inlineValues;

        // CSR-of-blocks index structure. Dual-mode: arena-tracked resolves through the record,
        // standalone keeps the inline field untouched -- mirrors floatN.Data (Arena/floatN.cs).
        // Indexed reads/writes (e.g. RowPtr[i] = ...) still mutate the underlying native buffer
        // through the returned UnsafeList's own internal pointer even though this getter returns a
        // header copy -- only a WHOLE-FIELD reassignment from outside this file would be unsafe,
        // and there is none (grepped repo-wide).
        public unsafe UnsafeList<int> RowPtr
        {
            get => _rec != null ? _rec->RowPtr : _inlineRowPtr;
            private set { if (_rec != null) _rec->RowPtr = value; else _inlineRowPtr = value; }
        }

        public unsafe UnsafeList<int> ColInd
        {
            get => _rec != null ? _rec->ColInd : _inlineColInd;
            private set { if (_rec != null) _rec->ColInd = value; else _inlineColInd = value; }
        }

        public unsafe UnsafeList<float> Values
        {
            get => _rec != null ? _rec->Values : _inlineValues;
            private set { if (_rec != null) _rec->Values = value; else _inlineValues = value; }
        }

        /// <summary>
        /// Allocates a compressed BSR matrix with the given block-grid shape and a fixed
        /// number of stored blocks (nnzb). Typically produced by floatBSRBuilder.ToBSR
        /// rather than called directly -- the caller is expected to fill RowPtr/ColInd/Values.
        /// </summary>
        public unsafe floatBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, Allocator allocator, bool uninit = false, bool symmetric = false)
        {
            _rec = null;
            _inlineRowPtr = default;
            _inlineColInd = default;
            _inlineValues = default;

            BlockRows = blockRows;
            BlockCols = blockCols;
            this.BR = BR;
            this.BC = BC;

            if (symmetric && (BR != BC || blockRows != blockCols))
                throw new ArgumentException("floatBSR: symmetric storage requires BR==BC and blockRows==blockCols");
            Symmetric = symmetric;

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
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.floatBSR) -- this ctor only fills in the
        /// record's RowPtr/ColInd/Values, it does not allocate or own the slot itself. Same
        /// pre-allocated-record contract as floatN's arena-tracked ctor (Arena/floatN.cs).
        /// </summary>
        internal unsafe floatBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, floatBSRRecord* rec, Allocator allocator, bool uninit = false, bool symmetric = false)
        {
            _rec = rec;
            _inlineRowPtr = default;
            _inlineColInd = default;
            _inlineValues = default;

            BlockRows = blockRows;
            BlockCols = blockCols;
            this.BR = BR;
            this.BC = BC;

            if (symmetric && (BR != BC || blockRows != blockCols))
                throw new ArgumentException("floatBSR: symmetric storage requires BR==BC and blockRows==blockCols");
            Symmetric = symmetric;

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

        public unsafe void Dispose()
        {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < Values.Length; i++) Values[i] = float.NaN;
#endif
            if (_rec != null)
            {
                // Cache the lists BEFORE Free(): Free() marks the slot dead and does not itself
                // clear the record's payload -- read them into locals first rather than reading
                // _rec-> again after the slot is dead. Free() runs BEFORE the native Dispose()
                // calls: an ALIASED double-dispose (a different struct copy sharing this SAME
                // record) throws HERE, from the table's own double-Free guard, before any native
                // memory would be touched a second time -- see floatN.Dispose() (Arena/floatN.cs)
                // for the full ordering rationale, which this mirrors exactly.
                var rowPtr = _rec->RowPtr;
                var colInd = _rec->ColInd;
                var values = _rec->Values;
                _rec->Table->Free(_rec->SelfIndex);
                rowPtr.Dispose();
                colInd.Dispose();
                values.Dispose();
                _rec = null;
            }
            else
            {
                _inlineRowPtr.Dispose();
                _inlineColInd.Dispose();
                _inlineValues.Dispose();
            }
        }

        /// <summary>
        /// Expands this BSR to a dense M_Rows x N_Cols matrix: zero-filled, then every stored
        /// block scattered into place. Used by tests and as a general-purpose densify helper.
        /// Kept as `ref Arena` for API stability, though `in Arena` would work equally well now
        /// that Arena is a thin handle to a heap-allocated ArenaCore (see Arena.cs).
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

                    if (Symmetric && bc != br)
                    {
                        for (int r = 0; r < BR; r++)
                            for (int c = 0; c < BC; c++)
                                dense[baseCol + c, baseRow + r] = Values[blockOffset + r * BC + c];
                    }
                }
            }

            return dense;
        }
    }
}
