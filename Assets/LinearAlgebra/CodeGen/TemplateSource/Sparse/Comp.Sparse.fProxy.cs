using System;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra.Internal;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    // Componentwise ops on a BSR matrix's stored entries. Only ops with f(0) = 0 belong here —
    // they leave the implicit zero blocks untouched, so the result equals the dense op applied
    // to the expanded matrix. Ops like exp or add-a-scalar would logically densify the matrix
    // and are deliberately absent.
    public static partial class fProxyComp
    {
        /// <summary>A *= s over the stored entries (pattern-preserving).</summary>
        public static void mulInPlace(this in fProxyBSR A, fProxy s)
        {
            unsafe { UnsafeOP.scalMul(A.Values.Ptr, A.Values.Length, s); }
        }

        /// <summary>A = -A over the stored entries (pattern-preserving).</summary>
        public static void signFlipInPlace(this in fProxyBSR A)
        {
            unsafe { UnsafeOP.signFlipInPlace(A.Values.Ptr, A.Values.Length); }
        }

        /// <summary>A = |A| over the stored entries (pattern-preserving).</summary>
        public static void absInPlace(this in fProxyBSR A)
        {
            unsafe { UnsafeMathOP.abs(A.Values.Ptr, A.Values.Length); }
        }

        /// <summary>
        /// y += a * x over the stored entries. Requires x and y to share an IDENTICAL sparsity
        /// pattern (same block grid and same block placements — see
        /// <see cref="LinearAlgebra.Sparse.BSR.samePattern"/>); throws otherwise. The typical use
        /// is perturbing a matrix (A' = A + eps*B) ahead of a warm re-solve.
        /// </summary>
        public static void addScaledInPlace(this in fProxyBSR y, fProxy a, in fProxyBSR x)
        {
            if (!Sparse.BSR.samePattern(in y, in x))
                throw new ArgumentException("addScaledInPlace: x and y must share an identical BSR sparsity pattern");

            unsafe { UnsafeOP.axpy(y.Values.Ptr, x.Values.Ptr, a, x.Values.Length); }
        }
    }
}

namespace LinearAlgebra.Sparse
{
    public static partial class BSR
    {
        /// <summary>
        /// True iff A and B use the same storage form (Symmetric flag), the same block grid
        /// (BlockRows/BlockCols/BR/BC) and identical block placements (RowPtr and ColInd match
        /// element-for-element). Values are not compared.
        /// </summary>
        public static bool samePattern(in fProxyBSR A, in fProxyBSR B)
        {
            if (A.Symmetric != B.Symmetric)
                return false;
            if (A.BlockRows != B.BlockRows || A.BlockCols != B.BlockCols ||
                A.BR != B.BR || A.BC != B.BC || A.Nnzb != B.Nnzb)
                return false;

            unsafe
            {
                var aRow = A.RowPtr; var bRow = B.RowPtr;
                var aCol = A.ColInd; var bCol = B.ColInd;
                if (UnsafeUtility.MemCmp(aRow.Ptr, bRow.Ptr, (long)aRow.Length * sizeof(int)) != 0)
                    return false;
                if (aCol.Length == 0) return true;
                return UnsafeUtility.MemCmp(aCol.Ptr, bCol.Ptr, (long)aCol.Length * sizeof(int)) == 0;
            }
        }
    }
}
