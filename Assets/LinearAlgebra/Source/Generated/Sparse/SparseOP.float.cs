using System;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Sparse matvec kernels over floatBSM (block-CSR). The shape mirrors Linear_OP.dot's
    /// dense matVec overloads (in A, in x, ref y) on purpose -- a future generic
    /// IfloatLinearOperator wrapper (Phase 2) can forward Apply/ApplyT straight to spMV/spMVT.
    /// </summary>
    public static partial class Sparse_OP
    {
        // ---- y = A * x ----

        // ref-dest primitive. Guard: y must not alias x (each x[k] feeds every block-row that
        // stores a block in column-block k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spMV(in floatBSM A, in floatN x, ref floatN y)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (y.N != A.M_Rows)
                throw new ArgumentException("spMV: y.N must equal A.M_Rows");

            unsafe
            {
                if (y.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("spMV: y must not alias x");

                // bsmMatVec accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<float>());

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                float* values = A.Values.Ptr;
                float* xPtr = x.Data.Ptr;
                float* yPtr = y.Data.Ptr;

                if (A.Symmetric)
                {
                    // Symmetric storage requires BR==BC by construction (floatBSM ctor), so
                    // dispatching on BR alone is sufficient here.
                    switch (A.BR)
                    {
                        case 1: Unsafe_OP.bsmMatVecSymB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: Unsafe_OP.bsmMatVecSymB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: Unsafe_OP.bsmMatVecSymB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: Unsafe_OP.bsmMatVecSymB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: Unsafe_OP.bsmMatVecSymB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: Unsafe_OP.bsmMatVecSym(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR); break;
                    }
                }
                else if (A.BR == A.BC)
                {
                    // Register-tile specializations only apply to square blocks -- rectangular
                    // BR != BC always falls through to the general kernel below.
                    switch (A.BR)
                    {
                        case 1: Unsafe_OP.bsmMatVecB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: Unsafe_OP.bsmMatVecB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: Unsafe_OP.bsmMatVecB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: Unsafe_OP.bsmMatVecB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: Unsafe_OP.bsmMatVecB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: Unsafe_OP.bsmMatVec(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC); break;
                    }
                }
                else
                {
                    Unsafe_OP.bsmMatVec(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN spMV(in floatBSM A, in floatN x)
        {
            floatN result = x.tempfloatVec(A.M_Rows);
            spMV(in A, in x, ref result);
            return result;
        }

        // ---- y = A^T * x ----

        // ref-dest primitive. Guard: y must not alias x (each x[k] feeds every block-column
        // that stores a block in block-row k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spMVT(in floatBSM A, in floatN x, ref floatN y)
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

                // bsmMatVecT accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<float>());

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                float* values = A.Values.Ptr;
                float* xPtr = x.Data.Ptr;
                float* yPtr = y.Data.Ptr;

                if (A.BR == A.BC)
                {
                    // Register-tile specializations only apply to square blocks -- rectangular
                    // BR != BC always falls through to the general kernel below.
                    switch (A.BR)
                    {
                        case 1: Unsafe_OP.bsmMatVecTB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: Unsafe_OP.bsmMatVecTB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: Unsafe_OP.bsmMatVecTB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: Unsafe_OP.bsmMatVecTB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: Unsafe_OP.bsmMatVecTB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: Unsafe_OP.bsmMatVecT(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC); break;
                    }
                }
                else
                {
                    Unsafe_OP.bsmMatVecT(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static floatN spMVT(in floatBSM A, in floatN x)
        {
            floatN result = x.tempfloatVec(A.N_Cols);
            spMVT(in A, in x, ref result);
            return result;
        }
    }
}
