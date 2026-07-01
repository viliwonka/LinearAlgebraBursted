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

                if (A.Symmetric)
                    Unsafe_OP.bsmMatVecSym(A.RowPtr.Ptr, A.ColInd.Ptr, A.Values.Ptr, x.Data.Ptr, y.Data.Ptr, A.BlockRows, A.BR);
                else
                    Unsafe_OP.bsmMatVec(A.RowPtr.Ptr, A.ColInd.Ptr, A.Values.Ptr, x.Data.Ptr, y.Data.Ptr,
                                         A.BlockRows, A.BR, A.BC);
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

                Unsafe_OP.bsmMatVecT(A.RowPtr.Ptr, A.ColInd.Ptr, A.Values.Ptr, x.Data.Ptr, y.Data.Ptr,
                                      A.BlockRows, A.BR, A.BC);
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
