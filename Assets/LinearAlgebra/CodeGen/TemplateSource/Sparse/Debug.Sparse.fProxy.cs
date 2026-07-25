using Unity.Collections;
using BULA.Sparse;

namespace BULA
{
    // Burst-safe block-structure printing for fProxyBSR (block-CSR sparse matrix), mirroring
    // Print.Spy/Print.Log for dense matrices (Debug.fProxy.cs) but at BLOCK granularity: one char
    // per BR x BC block instead of per scalar. See Sparse/fProxyBSR.cs for the RowPtr/ColInd/Values
    // (CSR-of-blocks, row-major block interior) layout this reads. Lives in namespace LinearAlgebra
    // (not BULA.Sparse) so it merges into the same `Print` partial class as the dense
    // overloads -- fProxyBSR is brought in via the `using BULA.Sparse;` above.
    public static partial class Print
    {
        // Scans ColInd[RowPtr[row]..RowPtr[row+1)) for `col`. ColInd is ascending within a row
        // (see fProxyBSR.cs), so this early-exits once it passes col instead of scanning the whole
        // row. Used both directly (non-symmetric) and with (row,col) swapped to mirror the stored
        // lower block-triangle into the upper triangle for Symmetric matrices.
        static bool BsrBlockStored(in fProxyBSR m, int row, int col)
        {
            int start = m.RowPtr[row];
            int end = m.RowPtr[row + 1];
            for (int k = start; k < end; k++)
            {
                int c = m.ColInd[k];
                if (c == col) return true;
                if (c > col) break;
            }
            return false;
        }

        // NB: the literal 8 below is the truncation budget -- bytes always kept free so a truncation
        // notice ("...\n") never overflows the buffer. It is inlined (not a const field) to avoid a
        // duplicate const across the float/double partials.

        // Appends the header + block-sparsity grid ('X' stored / ' ' absent) into str. One char per
        // block; for Symmetric matrices (only the lower block-triangle is stored, see fProxyBSR.cs)
        // the upper triangle is mirrored in the DISPLAY only -- storage itself is untouched. Caps
        // against the FixedString4096Bytes budget and appends "..." instead of overflowing. Returns
        // true if the grid was truncated (caller can skip anything more expensive after that).
        static bool AppendBsrSpyGrid(in fProxyBSR m, ref FixedString4096Bytes str)
        {
            int mb = m.BlockRows;
            int nb = m.BlockCols;
            int nnzb = m.Nnzb;
            int totalBlocks = mb * nb;
            fProxy density = totalBlocks > 0 ? (fProxy)nnzb / (fProxy)totalBlocks : (fProxy)0;

            FixedString128Bytes header1 = "BSR block sparsity print\n";
            FixedString128Bytes header2 = $"Dim | Rows:{m.M_Rows} Cols:{m.N_Cols}  Block:{m.BR}x{m.BC}  Grid:{mb}x{nb}\n";
            FixedString128Bytes header3 = $"Nnzb:{nnzb}  Symmetric:{(m.Symmetric ? 1 : 0)}  Density:{density:G3}\n";

            str.Append(header1);
            str.Append(header2);
            str.Append(header3);
            str.Append('\n');

            bool truncated = false;

            for (int br = 0; br < mb; br++)
            {
                // a row needs nb + 3 more chars (brackets + newline); keep the truncation trailer free too.
                if (str.Length + nb + 3 + 8 > str.Capacity)
                {
                    truncated = true;
                    break;
                }

                str.Append('[');
                for (int bc = 0; bc < nb; bc++)
                {
                    bool present;
                    if (m.Symmetric && br < bc)
                        present = BsrBlockStored(in m, bc, br); // mirror: (bc,br) is the stored lower block
                    else
                        present = BsrBlockStored(in m, br, bc);

                    str.Append(present ? 'X' : ' ');
                }
                str.Append(']');
                str.Append('\n');
            }

            if (truncated)
            {
                FixedString32Bytes trailer = "...\n";
                str.Append(trailer);
            }

            return truncated;
        }

        /// <summary>
        /// MATLAB-style block sparsity grid: one char per BR x BC block ('X' stored, ' ' absent),
        /// with a header giving M_Rows x N_Cols, block size BR x BC, block grid BlockRows x
        /// BlockCols, Nnzb and block density. Symmetric matrices (lower-block-triangle-only
        /// storage) are mirrored into the upper triangle for the display.
        /// </summary>
        public static void Spy(in fProxyBSR m)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();
            AppendBsrSpyGrid(in m, ref str);
            UnityEngine.Debug.Log($"{str}");
        }

        /// <summary>
        /// Spy's block sparsity grid PLUS the actual stored block values ("block(br,bc): [v0, v1,
        /// ...]"), iterated directly off RowPtr/ColInd/Values (no symmetric mirroring -- this shows
        /// what is actually stored). Caps against the FixedString4096Bytes budget and appends "..."
        /// instead of overflowing, both for the grid and for the value dump.
        /// </summary>
        public static void Log(in fProxyBSR m)
        {
            FixedString4096Bytes str = new FixedString4096Bytes();
            bool truncated = AppendBsrSpyGrid(in m, ref str);

            if (!truncated)
            {
                str.Append('\n');
                FixedString128Bytes valuesHeader = "Stored blocks:\n";
                str.Append(valuesHeader);

                int blockLen = m.BR * m.BC;
                bool valuesTruncated = false;

                for (int br = 0; br < m.BlockRows && !valuesTruncated; br++)
                {
                    int rowStart = m.RowPtr[br];
                    int rowEnd = m.RowPtr[br + 1];

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        // "block(999999,999999): [" is well under 48 bytes -- safe margin.
                        if (str.Length + 48 + 8 > str.Capacity)
                        {
                            valuesTruncated = true;
                            break;
                        }

                        int bc = m.ColInd[k];
                        FixedString64Bytes blockHeader = $"block({br},{bc}): [";
                        str.Append(blockHeader);

                        int blockOffset = k * blockLen;
                        bool blockTruncated = false;

                        for (int i = 0; i < blockLen; i++)
                        {
                            // a G4-formatted fProxy value plus ", " separator is at most ~24 bytes.
                            if (str.Length + 24 + 8 > str.Capacity)
                            {
                                blockTruncated = true;
                                valuesTruncated = true;
                                break;
                            }

                            fProxy v = m.Values[blockOffset + i];
                            FixedString64Bytes vs;
                            if (i == blockLen - 1)
                                vs = $"{v:G4}";
                            else
                                vs = $"{v:G4}, ";
                            str.Append(vs);
                        }

                        if (blockTruncated)
                            break;

                        str.Append(']');
                        str.Append('\n');
                    }
                }

                if (valuesTruncated)
                {
                    FixedString32Bytes trailer = "...\n";
                    str.Append(trailer);
                }
            }

            UnityEngine.Debug.Log($"{str}");
        }
    }
}
