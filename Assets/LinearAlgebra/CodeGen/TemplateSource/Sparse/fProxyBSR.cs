using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.InteropServices;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Block-CSR (BSR) sparse matrix: a uniform grid of BlockRows x BlockCols dense blocks,
    /// each BR x BC, stored compressed like CSR but with one column index PER BLOCK instead
    /// of per scalar (RowPtr/ColInd/Values -- "CSR of dense blocks").
    ///
    /// Block interior layout is ROW-MAJOR: block k's entry (r, c) lives at
    /// Values[k*BR*BC + r*BC + c] -- matching the library's row-major dense convention
    /// (fProxyMxN.Data[r*N_Cols+c]). Do not mix layouts. Blocks within a block-row are stored
    /// in ascending ColInd (enables transpose-SpMV and future binary-search block lookup).
    ///
    /// Logical scalar dims: M_Rows = BlockRows*BR, N_Cols = BlockCols*BC. Rectangular blocks
    /// (BR != BC) are supported. Set Symmetric=true to opt into lower-block-triangle-only
    /// storage (halves memory and single-threaded matvec FLOPs for symmetric matrices) --
    /// requires BR==BC and a square block grid (BlockRows==BlockCols).
    ///
    /// Lifecycle: build via fProxyBSRBuilder.ToBSR. This type is the compressed, matvec-ready
    /// form -- there is no cheap incremental pattern edit after compression; go back through
    /// the builder to add/remove blocks.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public partial struct fProxyBSR : IDisposable
    {
        public int BlockRows;  // mb: number of block-rows
        public int BlockCols;  // nb: number of block-cols
        public int BR;         // rows per block
        public int BC;         // cols per block

        public bool Symmetric;  // true => only the lower block-triangle (ColInd <= blockRow) is stored

        public int M_Rows => BlockRows * BR;
        public int N_Cols => BlockCols * BC;

        /// <summary>Number of stored (nonzero) blocks.</summary>
        public int Nnzb => ColInd.Length;

        private UnsafeList<int> _rowPtr;
        private UnsafeList<int> _colInd;
        private UnsafeList<fProxy> _values;

        public unsafe UnsafeList<int> RowPtr
        {
            get => _rowPtr;
            private set => _rowPtr = value;
        }

        public unsafe UnsafeList<int> ColInd
        {
            get => _colInd;
            private set => _colInd = value;
        }

        public unsafe UnsafeList<fProxy> Values
        {
            get => _values;
            private set => _values = value;
        }

        /// <summary>
        /// Allocates a compressed BSR matrix with the given block-grid shape and a fixed
        /// number of stored blocks (nnzb). Typically produced by fProxyBSRBuilder.ToBSR
        /// rather than called directly -- the caller is expected to fill RowPtr/ColInd/Values.
        /// </summary>
        public unsafe fProxyBSR(int blockRows, int blockCols, int BR, int BC, int nnzb, Allocator allocator, bool uninit = false, bool symmetric = false)
        {
            _rowPtr = default;
            _colInd = default;
            _values = default;

            BlockRows = blockRows;
            BlockCols = blockCols;
            this.BR = BR;
            this.BC = BC;

            if (symmetric && (BR != BC || blockRows != blockCols))
                throw new ArgumentException("fProxyBSR: symmetric storage requires BR==BC and blockRows==blockCols");
            Symmetric = symmetric;

            var options = uninit ? NativeArrayOptions.UninitializedMemory : NativeArrayOptions.ClearMemory;

            var rowPtr = new UnsafeList<int>(blockRows + 1, allocator, options);
            rowPtr.Resize(blockRows + 1, options);
            RowPtr = rowPtr;

            var colInd = new UnsafeList<int>(nnzb, allocator, options);
            colInd.Resize(nnzb, options);
            ColInd = colInd;

            int valuesLen = nnzb * BR * BC;
            var values = new UnsafeList<fProxy>(valuesLen, allocator, options);
            values.Resize(valuesLen, options);
            Values = values;
        }

        public unsafe void Dispose()
        {
            _rowPtr.Dispose();
            _colInd.Dispose();
            _values.Dispose();
        }

        /// <summary>
        /// Expands this BSR to a dense M_Rows x N_Cols matrix backed by
        /// <paramref name="allocator"/>: zero-filled, then every stored block scattered into place.
        /// Caller owns disposal for non-Temp allocators.
        /// </summary>
        public fProxyMxN ToDense(Allocator allocator = Allocator.Temp)
        {
            var dense = new fProxyMxN(M_Rows, N_Cols, allocator); // zero-initialized

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

        /// <summary>
        /// Materializes Aᵀ (block grid and per-block dimensions swapped, each stored block
        /// transposed) through a fresh <paramref name="allocator"/>-backed builder. If Symmetric,
        /// returns this matrix unchanged (transposing symmetric lower-block storage is a no-op) --
        /// the result then ALIASES this matrix's own buffers; Dispose only one of them.
        /// </summary>
        public unsafe fProxyBSR Transpose(Allocator allocator)
        {
            if (Symmetric)
                return this;

            var builder = new fProxyBSRBuilder(BlockCols, BlockRows, BC, BR, allocator, Nnzb);

            int blockLen = BR * BC;
            fProxy* blockT = stackalloc fProxy[blockLen];

            for (int bi = 0; bi < BlockRows; bi++)
            {
                int rowStart = RowPtr[bi];
                int rowEnd = RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = ColInd[k];
                    fProxy* block = Values.Ptr + k * blockLen;

                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BC; c++)
                            blockT[c * BR + r] = block[r * BC + c];

                    builder.AddBlock(bj, bi, blockT);
                }
            }

            var result = builder.ToBSR(allocator);
            builder.Dispose();
            return result;
        }

        /// <summary>
        /// Mirrors a SYMMETRIC-storage (lower-block-triangle-only) matrix into full storage
        /// through a fresh <paramref name="allocator"/>-backed builder. If not Symmetric, returns
        /// this matrix unchanged (no copy) -- the result then ALIASES this matrix's own buffers;
        /// Dispose only one of them.
        /// </summary>
        public unsafe fProxyBSR MirrorToFull(Allocator allocator)
        {
            if (!Symmetric)
                return this;

            var builder = new fProxyBSRBuilder(BlockRows, BlockCols, BR, BC, allocator, Nnzb * 2);

            int blockLen = BR * BC;
            fProxy* blockT = stackalloc fProxy[blockLen];

            for (int bi = 0; bi < BlockRows; bi++)
            {
                int rowStart = RowPtr[bi];
                int rowEnd = RowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = ColInd[k];
                    fProxy* block = Values.Ptr + k * blockLen;

                    builder.AddBlock(bi, bj, block);

                    if (bi != bj)
                    {
                        for (int r = 0; r < BR; r++)
                            for (int c = 0; c < BC; c++)
                                blockT[c * BR + r] = block[r * BC + c];
                        builder.AddBlock(bj, bi, blockT);
                    }
                }
            }

            var result = builder.ToBSR(allocator);
            builder.Dispose();
            return result;
        }
    }
}
