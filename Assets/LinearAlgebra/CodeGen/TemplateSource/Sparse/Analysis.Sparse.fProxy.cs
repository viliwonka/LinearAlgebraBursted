using System;
using BULA.Sparse;

namespace BULA
{
    public static partial class Analysis
    {
        /// <summary>
        /// Trace of a BSR matrix: Σ aᵢᵢ. Requires square blocks on a square grid (BR == BC,
        /// BlockRows == BlockCols) so every global diagonal entry lives in a block-diagonal
        /// block; absent diagonal blocks contribute 0. Works for symmetric storage unchanged
        /// (diagonal blocks are always stored explicitly).
        /// </summary>
        public static fProxy trace(in fProxyBSR A)
        {
            if (A.BR != A.BC || A.BlockRows != A.BlockCols)
                throw new ArgumentException("trace: requires square blocks on a square block grid (BR == BC, BlockRows == BlockCols)");

            var rowPtr = A.RowPtr; var colInd = A.ColInd; var values = A.Values;
            int blockLen = A.BR * A.BC;

            fProxy sum = 0;
            for (int row = 0; row < A.BlockRows; row++)
            {
                int s = rowPtr[row], e = rowPtr[row + 1];
                for (int k = s; k < e; k++)
                {
                    if (colInd[k] != row) continue;
                    int off = k * blockLen;
                    for (int r = 0; r < A.BR; r++)
                        sum += values[off + r * A.BC + r];
                    break;   // at most one diagonal block per block-row
                }
            }
            return sum;
        }

        /// <summary>
        /// Extracts the main diagonal of a BSR matrix into d (d.N == A.M_Rows). Same squareness
        /// requirement as <see cref="trace(in fProxyBSR)"/>; entries in absent diagonal blocks
        /// come out 0.
        /// </summary>
        public static void diagonal(in fProxyBSR A, ref fProxyN d)
        {
            if (A.BR != A.BC || A.BlockRows != A.BlockCols)
                throw new ArgumentException("diagonal: requires square blocks on a square block grid (BR == BC, BlockRows == BlockCols)");
            if (d.N != A.M_Rows)
                throw new ArgumentException("diagonal: d.N must equal A.M_Rows");

            for (int i = 0; i < d.N; i++) d[i] = 0;

            var rowPtr = A.RowPtr; var colInd = A.ColInd; var values = A.Values;
            int blockLen = A.BR * A.BC;

            for (int row = 0; row < A.BlockRows; row++)
            {
                int s = rowPtr[row], e = rowPtr[row + 1];
                for (int k = s; k < e; k++)
                {
                    if (colInd[k] != row) continue;
                    int off = k * blockLen;
                    for (int r = 0; r < A.BR; r++)
                        d[row * A.BR + r] = values[off + r * A.BC + r];
                    break;
                }
            }
        }

        /// <summary>
        /// Extracts the main diagonal of a dense matrix into d (d.N == min(M_Rows, N_Cols)) —
        /// dense counterpart of the BSR overload.
        /// </summary>
        public static void diagonal(in fProxyMxN A, ref fProxyN d)
        {
            int n = A.M_Rows < A.N_Cols ? A.M_Rows : A.N_Cols;
            if (d.N != n)
                throw new ArgumentException("diagonal: d.N must equal min(A.M_Rows, A.N_Cols)");

            for (int i = 0; i < n; i++)
                d[i] = A[i, i];
        }
    }
}
