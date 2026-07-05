using System;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Sparse matvec kernels over doubleBSR (block-CSR). The shape mirrors Blas.dot's
    /// dense matVec overloads (in A, in x, ref y) on purpose -- a future generic
    /// IdoubleLinearOperator wrapper (Phase 2) can forward Apply/ApplyT straight to spMV/spMVT.
    /// </summary>
    public static partial class BSR
    {
        // ---- y = A * x ----

        // ref-dest primitive. Guard: y must not alias x (each x[k] feeds every block-row that
        // stores a block in column-block k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spMV(in doubleBSR A, in doubleN x, ref doubleN y)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (y.N != A.M_Rows)
                throw new ArgumentException("spMV: y.N must equal A.M_Rows");

            unsafe
            {
                if (y.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("spMV: y must not alias x");

                // bsrMatVec accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<double>());

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* xPtr = x.Data.Ptr;
                double* yPtr = y.Data.Ptr;

                if (A.Symmetric)
                {
                    // Symmetric storage requires BR==BC by construction (doubleBSR ctor), so
                    // dispatching on BR alone is sufficient here.
                    switch (A.BR)
                    {
                        case 1: UnsafeOP.bsrMatVecSymB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: UnsafeOP.bsrMatVecSymB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: UnsafeOP.bsrMatVecSymB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: UnsafeOP.bsrMatVecSymB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: UnsafeOP.bsrMatVecSymB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: UnsafeOP.bsrMatVecSym(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR); break;
                    }
                }
                else if (A.BR == A.BC)
                {
                    // Register-tile specializations only apply to square blocks -- rectangular
                    // BR != BC always falls through to the general kernel below.
                    switch (A.BR)
                    {
                        case 1: UnsafeOP.bsrMatVecB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: UnsafeOP.bsrMatVecB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: UnsafeOP.bsrMatVecB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: UnsafeOP.bsrMatVecB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: UnsafeOP.bsrMatVecB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: UnsafeOP.bsrMatVec(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC); break;
                    }
                }
                else
                {
                    UnsafeOP.bsrMatVec(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN spMV(in doubleBSR A, in doubleN x)
        {
            doubleN result = x.doubleTempVec(A.M_Rows);
            spMV(in A, in x, ref result);
            return result;
        }

        /// <summary>
        /// Squared L2 norm of each column of the block-sparse A: d2[j] = Σ_i A[i,j]² = diag(AᵀA)[j],
        /// computed directly from the stored blocks in a single pass over the nonzeros (no AᵀA
        /// formed, no transpose-matvecs). Written into the caller's d2 (length A.N_Cols), no
        /// allocation. Feeds an AᵀA-Jacobi (column-equilibration) least-squares preconditioner
        /// (see <see cref="doubleColScaledOperator{TInner}"/> / <c>Blas.buildJacobiScale</c>).
        ///
        /// NOT supported for Symmetric (upper-block-triangle-only) storage: the implicit lower
        /// blocks are not materialized, so a single pass would under-count every column -- throws
        /// in that case. Jacobi-LS preconditioning targets rectangular / non-symmetric least
        /// squares, where Symmetric is false.
        /// </summary>
        public static void columnNormsSquared(in doubleBSR A, ref doubleN d2)
        {
            if (d2.N != A.N_Cols)
                throw new ArgumentException("columnNormsSquared: d2.N must equal A.N_Cols");
            if (A.Symmetric)
                throw new ArgumentException("columnNormsSquared: not supported for Symmetric (upper-block-only) storage -- the implicit lower blocks would be under-counted");

            int BR = A.BR, BC = A.BC;
            int blockSize = BR * BC;

            unsafe
            {
                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* d2Ptr = d2.Data.Ptr;

                UnsafeUtility.MemClear(d2Ptr, (long)d2.Data.Length * UnsafeUtility.SizeOf<double>());

                for (int bi = 0; bi < A.BlockRows; bi++)
                {
                    for (int k = rowPtr[bi]; k < rowPtr[bi + 1]; k++)
                    {
                        int colBase = colInd[k] * BC;         // global column of block-interior col 0
                        double* block = values + (long)k * blockSize;
                        for (int r = 0; r < BR; r++)
                            for (int c = 0; c < BC; c++)
                            {
                                double v = block[r * BC + c]; // row-major block interior
                                d2Ptr[colBase + c] += v * v;
                            }
                    }
                }
            }
        }

        // ---- y = A^T * x ----

        // ref-dest primitive. Guard: y must not alias x (each x[k] feeds every block-column
        // that stores a block in block-row k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spMVT(in doubleBSR A, in doubleN x, ref doubleN y)
        {
            if (A.Symmetric)
            {
                // A == A^T for symmetric upper-block storage -- forward straight to spMV. Its guards
                // (Assume.SameDim(A.N_Cols, x.N), y.N != A.M_Rows) are equivalent to spMVT's own
                // (Assume.SameDim(A.M_Rows, x.N), y.N != A.N_Cols) here because Symmetric implies
                // A.M_Rows == A.N_Cols.
                spMV(in A, in x, ref y);
                return;
            }

            Assume.SameDim(A.M_Rows, x.N);

            if (y.N != A.N_Cols)
                throw new ArgumentException("spMVT: y.N must equal A.N_Cols");

            unsafe
            {
                if (y.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("spMVT: y must not alias x");

                // bsrMatVecT accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<double>());

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* xPtr = x.Data.Ptr;
                double* yPtr = y.Data.Ptr;

                if (A.BR == A.BC)
                {
                    // Register-tile specializations only apply to square blocks -- rectangular
                    // BR != BC always falls through to the general kernel below.
                    switch (A.BR)
                    {
                        case 1: UnsafeOP.bsrMatVecTB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: UnsafeOP.bsrMatVecTB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: UnsafeOP.bsrMatVecTB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: UnsafeOP.bsrMatVecTB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: UnsafeOP.bsrMatVecTB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: UnsafeOP.bsrMatVecT(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC); break;
                    }
                }
                else
                {
                    UnsafeOP.bsrMatVecT(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN spMVT(in doubleBSR A, in doubleN x)
        {
            doubleN result = x.doubleTempVec(A.N_Cols);
            spMVT(in A, in x, ref result);
            return result;
        }
    }
}
