#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;
using System;

using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    /// <summary>
    /// Inpl = inplace
    /// </summary>
    public static partial class int_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int dot(intN a, intN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return Unsafe_OP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        // ---- outer product: a (col) * b (row) -> M x N ----

        // ref-dest primitive. No alias guard: result is a matrix, inputs are vectors,
        // so they can never share a buffer.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void outerDot(in intN a, in intN b, ref intMxN result)
        {
            if (result.M_Rows != a.N || result.N_Cols != b.N)
                throw new ArgumentException("outerDot: result must be a.N x b.N");

            unsafe
            {
                Unsafe_OP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN outerDot(intN a, intN b)
        {
            intMxN result = a.tempintMat(a.N, b.N, true);
            outerDot(in a, in b, ref result);
            return result;
        }

        // ---- matrix * vector -> vector ----

        // ref-dest primitive. Guard: result must not alias x (each x[k] feeds every row).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in intMxN A, in intN x, ref intN result)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (result.N != A.M_Rows)
                throw new ArgumentException("dot: result.N must equal A.M_Rows");

            unsafe {
                if (result.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("dot: result must not alias x");

                // matVecDot accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(result.Data.Ptr, (long)result.Data.Length * UnsafeUtility.SizeOf<int>());

                Unsafe_OP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN dot(intMxN A, intN x)
        {
            intN result = x.tempintVec(A.M_Rows);
            dot(in A, in x, ref result);
            return result;
        }

        // ---- vector * matrix -> vector ----

        // ref-dest primitive. Guard: result must not alias y (each y[i] feeds every column).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in intN y, in intMxN A, ref intN result)
        {
            Assume.SameDim(A.M_Rows, y.N);

            if (result.N != A.N_Cols)
                throw new ArgumentException("dot: result.N must equal A.N_Cols");

            unsafe {
                if (result.Data.Ptr == y.Data.Ptr)
                    throw new ArgumentException("dot: result must not alias y");

                // vecMatDot accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(result.Data.Ptr, (long)result.Data.Length * UnsafeUtility.SizeOf<int>());

                Unsafe_OP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN dot(intN y, intMxN A)
        {
            intN result = y.tempintVec(A.N_Cols);
            dot(in y, in A, ref result);
            return result;
        }

        // ---- matrix * matrix -> matrix ----

        // ref-dest primitive. Guard: c must not alias a or b (each input entry feeds a
        // whole row/column of the product).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in intMxN a, in intMxN b, ref intMxN c, bool transposeA = false)
        {
            // C = Aᵀ·B contracts over the rows of A and B, so a.M_Rows must equal b.M_Rows.
            // (The non-transposed path contracts a.N_Cols against b.M_Rows.)
            if(transposeA)
                Assume.SameDim(a.M_Rows, b.M_Rows);
            else
                Assume.SameDim(a.N_Cols, b.M_Rows);

            int m, n, k;

            if (transposeA)
            {
                m = a.N_Cols; n = a.M_Rows ; k = b.N_Cols;
            }
            else {
                m = a.M_Rows; n = a.N_Cols; k = b.N_Cols;
            }

            if (c.M_Rows != m || c.N_Cols != k)
                throw new ArgumentException("dot: destination must be m x k");

            unsafe
            {
                if (c.Data.Ptr == a.Data.Ptr || c.Data.Ptr == b.Data.Ptr)
                    throw new ArgumentException("dot: destination must not alias an input");

                // matMatDot / matMatDotTransA accumulate (+=), so c must start zeroed.
                UnsafeUtility.MemClear(c.Data.Ptr, (long)c.Data.Length * UnsafeUtility.SizeOf<int>());

                if(transposeA)
                    Unsafe_OP.matMatDotTransA(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
                else
                    Unsafe_OP.matMatDot(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN dot(intMxN a, intMxN b, bool transposeA = false)
        {
            int m = transposeA ? a.N_Cols : a.M_Rows;
            int k = b.N_Cols;

            intMxN c = a.tempintMat(m, k);
            dot(in a, in b, ref c, transposeA);
            return c;
        }

        // ---- transpose -> matrix ----

        // ref-dest primitive. Guard: T must not alias A. Transpose is a permutation, so
        // even though each entry is read once, writing T[i,j] would clobber A[i,j] which
        // is still needed as T[j,i].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void trans(in intMxN A, ref intMxN T)
        {
            if (T.M_Rows != A.N_Cols || T.N_Cols != A.M_Rows)
                throw new ArgumentException("trans: destination must be A.N_Cols x A.M_Rows");

            unsafe
            {
                if (T.Data.Ptr == A.Data.Ptr)
                    throw new ArgumentException("trans: destination must not alias the input");

                Unsafe_OP.matTrans(A.Data.Ptr, T.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN trans(intMxN A)
        {
            var T = A.tempintMat(A.N_Cols, A.M_Rows, true);
            trans(in A, ref T);
            return T;
        }
    }
}
