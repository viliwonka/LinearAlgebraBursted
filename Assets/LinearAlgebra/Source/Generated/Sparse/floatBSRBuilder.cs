using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using Unity.Mathematics;
using System;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// COO-of-blocks assembly builder for floatBSR. Accumulates (blockRow, blockCol, BR x BC
    /// block) triplets in growable allocator-backed lists; call ToBSR(arena) ONCE to sort and
    /// compress into block-CSR. Duplicate triplets at the same (blockRow, blockCol) are summed
    /// on compression -- this is the "sparse matrix is a graph" editable phase (add/remove a
    /// node = add/remove triplets).
    ///
    /// Editing the pattern after compression is out of scope for Phase 1 -- go back through the
    /// builder (re-stamping VALUES on a fixed pattern without a rebuild is a later phase).
    ///
    /// The growable triplet state lives behind a single heap-allocated State* shared by every
    /// value-copy of this struct (Malloc'd once in the constructor, Free'd once in Dispose), so
    /// UnsafeList growth from AddBlock/AddValue is visible to every copy -- including the arena's
    /// own tracked copy -- instead of diverging.
    /// </summary>
    public partial struct floatBSRBuilder : IDisposable
    {
        // Heap-owned, single-identity mutable state -- see the type doc above. Every
        // floatBSRBuilder value-copy (this instance, the arena's tracked copy, the caller's
        // copy, ...) shares the SAME pointee, so UnsafeList growth from AddBlock/AddValue is
        // visible everywhere, including to the arena's own bookkeeping copy.
        private struct State
        {
            public int BlockRows;  // mb: number of block-rows
            public int BlockCols;  // nb: number of block-cols
            public int BR;         // rows per block
            public int BC;         // cols per block

            public Allocator Allocator;

            public UnsafeList<int> triBlockRow;
            public UnsafeList<int> triBlockCol;

            // flat, row-major per block: triplet t's block occupies
            // triValues[t*BR*BC .. t*BR*BC + BR*BC)
            public UnsafeList<float> triValues;
        }

        [NativeDisableUnsafePtrRestriction]
        private unsafe State* _state;

        public unsafe int BlockRows => _state->BlockRows;
        public unsafe int BlockCols => _state->BlockCols;
        public unsafe int BR => _state->BR;
        public unsafe int BC => _state->BC;

        public int M_Rows => BlockRows * BR;
        public int N_Cols => BlockCols * BC;

        /// <summary>
        /// Number of accumulated (blockRow, blockCol, block) triplets PRE-compression --
        /// duplicates at the same (blockRow, blockCol) are still separate entries here.
        /// </summary>
        public unsafe int TripletCount => _state->triBlockRow.Length;

        // Value handle to the shared ArenaCore, not a raw pointer (see Arena.cs); copies stay live.
        // Unrelated to the _state indirection above.
        private Arena _arena;

        public unsafe floatBSRBuilder(int blockRows, int blockCols, int BR, int BC, Allocator allocator, int capacityHint = 8)
        {
            _arena = default;

            _state = (State*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<State>(), UnsafeUtility.AlignOf<State>(), allocator);
            _state->BlockRows = blockRows;
            _state->BlockCols = blockCols;
            _state->BR = BR;
            _state->BC = BC;
            _state->Allocator = allocator;

            _state->triBlockRow = new UnsafeList<int>(capacityHint, allocator);
            _state->triBlockCol = new UnsafeList<int>(capacityHint, allocator);
            _state->triValues = new UnsafeList<float>(capacityHint * BR * BC, allocator);
        }

        public unsafe floatBSRBuilder(int blockRows, int blockCols, int BR, int BC, in Arena arena, int capacityHint = 8)
        {
            _arena = arena;

            var allocator = arena.Allocator;

            _state = (State*)UnsafeUtility.Malloc(UnsafeUtility.SizeOf<State>(), UnsafeUtility.AlignOf<State>(), allocator);
            _state->BlockRows = blockRows;
            _state->BlockCols = blockCols;
            _state->BR = BR;
            _state->BC = BC;
            _state->Allocator = allocator;

            _state->triBlockRow = new UnsafeList<int>(capacityHint, allocator);
            _state->triBlockCol = new UnsafeList<int>(capacityHint, allocator);
            _state->triValues = new UnsafeList<float>(capacityHint * BR * BC, allocator);
        }

        /// <summary>
        /// Appends a (br, bc, block) triplet. block must point to BR*BC values in row-major
        /// order (block[r*BC+c]). Multiple triplets at the same (br, bc) are summed on ToBSR.
        /// </summary>
        public unsafe void AddBlock(int br, int bc, [NoAlias] float* block)
        {
            if (br < 0 || br >= BlockRows)
                throw new ArgumentException("AddBlock: blockRow out of bounds");
            if (bc < 0 || bc >= BlockCols)
                throw new ArgumentException("AddBlock: blockCol out of bounds");

            _state->triBlockRow.Add(br);
            _state->triBlockCol.Add(bc);

            int blockLen = BR * BC;
            for (int i = 0; i < blockLen; i++)
                _state->triValues.Add(block[i]);
        }

        /// <summary>
        /// Appends a (br, bc, block) triplet from a dense BR x BC view.
        /// </summary>
        public unsafe void AddBlock(int br, int bc, in floatMxN block)
        {
            if (block.M_Rows != BR || block.N_Cols != BC)
                throw new ArgumentException("AddBlock: block dimensions must be BR x BC");

            AddBlock(br, bc, block.Data.Ptr);
        }

        /// <summary>
        /// Convenience: routes a single scalar at global (row, col) into its owning block,
        /// appending a fresh triplet that is zero everywhere except that one entry. Summed
        /// with any other triplet touching the same block on ToBSR -- fine for occasional
        /// edits, wasteful for many scalar adds into the same block (prefer AddBlock for that).
        /// </summary>
        public unsafe void AddValue(int globalRow, int globalCol, float v)
        {
            if (globalRow < 0 || globalRow >= M_Rows)
                throw new ArgumentException("AddValue: globalRow out of bounds");
            if (globalCol < 0 || globalCol >= N_Cols)
                throw new ArgumentException("AddValue: globalCol out of bounds");

            int br = globalRow / BR;
            int localR = globalRow % BR;
            int bc = globalCol / BC;
            int localC = globalCol % BC;

            _state->triBlockRow.Add(br);
            _state->triBlockCol.Add(bc);

            int targetIdx = localR * BC + localC;
            int blockLen = BR * BC;
            for (int i = 0; i < blockLen; i++)
                _state->triValues.Add(i == targetIdx ? v : (float)0);
        }

        /// <summary>
        /// Sorts triplets by (blockRow, blockCol), sums duplicates at the same (blockRow,
        /// blockCol), and builds the compressed floatBSR (RowPtr/ColInd/Values). Counting-sort
        /// by block-row + insertion-sort by block-col within each row bucket: deterministic,
        /// O(nnz + sum of row-degree^2) -- fine for the one-time assembly-to-compressed
        /// transition this represents (see the type doc re: Phase 1 pattern-edit scope).
        ///
        /// Kept as `ref Arena` for API stability, but this is no longer load-bearing -- see the
        /// matching comment on floatBSR.ToDense: Arena is now a thin copyable handle to a
        /// heap-allocated ArenaCore, so `in Arena` would resolve correctly here too.
        /// </summary>
        public unsafe floatBSR ToBSR(ref Arena arena) => ToBSRCore(ref arena, symmetric: false);

        /// <summary>
        /// Same as ToBSR, but builds SYMMETRIC upper-block storage (see floatBSR.Symmetric / spec
        /// §2.3): requires BR==BC and BlockRows==BlockCols, and requires every accumulated triplet's
        /// blockCol >= blockRow (upper triangle + diagonal only). A lower-triangle triplet
        /// (blockCol < blockRow) throws immediately -- we do NOT silently fold it into its transpose
        /// position, because that would mask caller bugs (e.g. accidentally adding both A_ij and A_ji
        /// for what the caller believes is a symmetric matrix, when they actually differ). Callers
        /// building a symmetric matrix must AddBlock/AddValue only at (blockRow, blockCol) with
        /// blockCol >= blockRow.
        ///
        /// Each stored DIAGONAL block must itself be symmetric (block[r,c] == block[c,r]). Upper-block
        /// storage represents the implicit lower block (bj,bi) as block(bi,bj)^T, so the matrix is
        /// symmetric ONLY IF the diagonal blocks are -- and spMVT forwards to spMV assuming A==A^T, so
        /// a non-symmetric diagonal block would silently make spMVT return A*x, not A^T*x. A
        /// non-symmetric diagonal block therefore throws (same "don't mask caller bugs" stance as the
        /// lower-triangle guard). The check is on the duplicate-SUMMED diagonal block, so a symmetric
        /// block assembled from several AddBlock/AddValue contributions is accepted.
        /// </summary>
        public unsafe floatBSR ToBSRSymmetric(ref Arena arena)
        {
            if (BR != BC || BlockRows != BlockCols)
                throw new ArgumentException("ToBSRSymmetric: requires BR==BC and BlockRows==BlockCols (square blocks on a square block grid)");

            int n = TripletCount;
            for (int t = 0; t < n; t++)
                if (_state->triBlockCol[t] < _state->triBlockRow[t])
                    throw new ArgumentException("ToBSRSymmetric: found a lower-triangle triplet (blockCol < blockRow); symmetric build only accepts blocks with blockCol >= blockRow (upper triangle + diagonal). Add the block at its transpose position (blockRow<->blockCol swapped) instead, or use ToBSR() for full storage.");

            var bsm = ToBSRCore(ref arena, symmetric: true);

            // Validate diagonal-block symmetry on the compressed (duplicate-summed) blocks. A
            // relative tolerance absorbs assembly roundoff while still catching a genuinely
            // non-symmetric block (whose block[r,c]-block[c,r] is O(block magnitude)).
            int blockLen = BR * BC;
            for (int row = 0; row < BlockRows; row++)
            {
                int rs = bsm.RowPtr[row], re = bsm.RowPtr[row + 1];
                for (int k = rs; k < re; k++)
                {
                    if (bsm.ColInd[k] != row) continue;   // diagonal blocks only
                    int off = k * blockLen;
                    for (int r = 0; r < BR; r++)
                        for (int c = r + 1; c < BC; c++)
                        {
                            float a = bsm.Values[off + r * BC + c];
                            float b = bsm.Values[off + c * BC + r];
                            float tolAbs = (float)8 * Consts.floatZeroThreshold * ((float)1 + math.abs(a) + math.abs(b));
                            if (math.abs(a - b) > tolAbs)
                                throw new ArgumentException("ToBSRSymmetric: a diagonal block is not symmetric (block[r,c] != block[c,r]). Symmetric upper-block storage stores the lower triangle implicitly as the transpose, so diagonal blocks must be symmetric; symmetrize the block (e.g. (K+K^T)/2), or use ToBSR() for full storage.");
                        }
                }
            }

            return bsm;
        }

        private unsafe floatBSR ToBSRCore(ref Arena arena, bool symmetric)
        {
            int n = TripletCount;
            int blockLen = BR * BC;

            // 1. Counting-sort triplet indices into per-block-row buckets (rowStart[i] is the
            //    bucket boundary, à la CSR construction from COO).
            var rowStart = new NativeArray<int>(BlockRows + 1, Allocator.Temp, NativeArrayOptions.ClearMemory);
            for (int t = 0; t < n; t++)
                rowStart[_state->triBlockRow[t] + 1]++;
            for (int i = 0; i < BlockRows; i++)
                rowStart[i + 1] += rowStart[i];

            var order = new NativeArray<int>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var cursor = new NativeArray<int>(BlockRows, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < BlockRows; i++) cursor[i] = rowStart[i];
            for (int t = 0; t < n; t++)
            {
                int row = _state->triBlockRow[t];
                order[cursor[row]] = t;
                cursor[row]++;
            }
            cursor.Dispose();

            // 2. Insertion-sort each row bucket by blockCol (row degree is small in practice).
            for (int row = 0; row < BlockRows; row++)
            {
                int s = rowStart[row];
                int e = rowStart[row + 1];
                for (int i = s + 1; i < e; i++)
                {
                    int cur = order[i];
                    int curCol = _state->triBlockCol[cur];
                    int j = i - 1;
                    while (j >= s && _state->triBlockCol[order[j]] > curCol)
                    {
                        order[j + 1] = order[j];
                        j--;
                    }
                    order[j + 1] = cur;
                }
            }

            // 3. Count distinct stored blocks (nnzb) after de-duplication.
            int nnzb = 0;
            for (int row = 0; row < BlockRows; row++)
            {
                int s = rowStart[row];
                int e = rowStart[row + 1];
                int prevCol = -1;
                for (int i = s; i < e; i++)
                {
                    int col = _state->triBlockCol[order[i]];
                    if (col != prevCol) { nnzb++; prevCol = col; }
                }
            }

            var bsm = arena.floatBSR(BlockRows, BlockCols, BR, BC, nnzb, true, symmetric);

            // Cache the three lists into local variables ONCE: RowPtr/ColInd/Values are dual-mode
            // properties (floatBSR.cs), and a local UnsafeList<T> copy is addressable and shares
            // the SAME underlying native buffer, so indexing through these locals still mutates
            // bsm's real storage -- and it's cheaper too (no repeated property dispatch inside the
            // loop below).
            var rowPtr = bsm.RowPtr;
            var colInd = bsm.ColInd;
            var values = bsm.Values;

            // 4. Fill RowPtr/ColInd/Values, summing consecutive same-column entries.
            int outIdx = 0;
            for (int row = 0; row < BlockRows; row++)
            {
                rowPtr[row] = outIdx;
                int s = rowStart[row];
                int e = rowStart[row + 1];
                int prevCol = -1;

                for (int i = s; i < e; i++)
                {
                    int t = order[i];
                    int col = _state->triBlockCol[t];
                    int srcOff = t * blockLen;

                    if (col != prevCol)
                    {
                        colInd[outIdx] = col;
                        int dstOff = outIdx * blockLen;
                        for (int k = 0; k < blockLen; k++)
                            values[dstOff + k] = _state->triValues[srcOff + k];
                        prevCol = col;
                        outIdx++;
                    }
                    else
                    {
                        int dstOff = (outIdx - 1) * blockLen;
                        for (int k = 0; k < blockLen; k++)
                            values[dstOff + k] += _state->triValues[srcOff + k];
                    }
                }
            }
            rowPtr[BlockRows] = outIdx;

            order.Dispose();
            rowStart.Dispose();

            return bsm;
        }

        /// <summary>
        /// Frees the shared triplet state and the State block itself. Idempotent on the same
        /// struct copy. The owning arena disposes builders it created; callers are not expected
        /// to dispose the value returned by arena.floatBSRBuilder(...) themselves.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_state == null)
                return;

            _state->triBlockRow.Dispose();
            _state->triBlockCol.Dispose();
            _state->triValues.Dispose();

            UnsafeUtility.Free(_state, _state->Allocator);
            _state = null;
        }
    }
}
