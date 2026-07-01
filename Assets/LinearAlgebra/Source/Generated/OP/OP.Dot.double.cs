#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;
using System;

using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// Inpl = inplace
    /// </summary>
    public static partial class Linear_OP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double dot(doubleN a, doubleN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return Unsafe_OP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double dot(doubleN a, doubleN b, int start, int end = -1) {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            if(end == -1)
                end = a.N;

            unsafe {
                return Unsafe_OP.vecDotRange(a.Data.Ptr, b.Data.Ptr, start, end);
            }
        }

        // ---- outer product: a (col) * b (row) -> M x N ----

        // ref-dest primitive. No alias guard: result is a matrix, inputs are vectors,
        // so they can never share a buffer.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void outerDot(in doubleN a, in doubleN b, ref doubleMxN result)
        {
            if (result.M_Rows != a.N || result.N_Cols != b.N)
                throw new ArgumentException("outerDot: result must be a.N x b.N");

            unsafe
            {
                Unsafe_OP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN outerDot(doubleN a, doubleN b)
        {
            doubleMxN result = a.tempdoubleMat(a.N, b.N, true);
            outerDot(in a, in b, ref result);
            return result;
        }

        // ---- matrix * vector -> vector ----

        // ref-dest primitive. Guard: result must not alias x (each x[k] feeds every row).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in doubleMxN A, in doubleN x, ref doubleN result)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (result.N != A.M_Rows)
                throw new ArgumentException("dot: result.N must equal A.M_Rows");

            unsafe {
                if (result.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("dot: result must not alias x");

                // matVecDot accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(result.Data.Ptr, (long)result.Data.Length * UnsafeUtility.SizeOf<double>());

                Unsafe_OP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN dot(doubleMxN A, doubleN x)
        {
            doubleN result = x.tempdoubleVec(A.M_Rows);
            dot(in A, in x, ref result);
            return result;
        }

        // ---- vector * matrix -> vector ----

        // ref-dest primitive. Guard: result must not alias y (each y[i] feeds every column).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in doubleN y, in doubleMxN A, ref doubleN result)
        {
            Assume.SameDim(A.M_Rows, y.N);

            if (result.N != A.N_Cols)
                throw new ArgumentException("dot: result.N must equal A.N_Cols");

            unsafe {
                if (result.Data.Ptr == y.Data.Ptr)
                    throw new ArgumentException("dot: result must not alias y");

                // vecMatDot accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(result.Data.Ptr, (long)result.Data.Length * UnsafeUtility.SizeOf<double>());

                Unsafe_OP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN dot(doubleN y, doubleMxN A)
        {
            doubleN result = y.tempdoubleVec(A.N_Cols);
            dot(in y, in A, ref result);
            return result;
        }

        // ---- matrix * matrix -> matrix ----

        // ref-dest primitive. Guard: c must not alias a or b (each input entry feeds a
        // whole row/column of the product).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in doubleMxN a, in doubleMxN b, ref doubleMxN c, bool transposeA = false)
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
                UnsafeUtility.MemClear(c.Data.Ptr, (long)c.Data.Length * UnsafeUtility.SizeOf<double>());

                if(transposeA)
                    Unsafe_OP.matMatDotTransA(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
                else
                    Unsafe_OP.matMatDot(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN dot(doubleMxN a, doubleMxN b, bool transposeA = false)
        {
            int m = transposeA ? a.N_Cols : a.M_Rows;
            int k = b.N_Cols;

            doubleMxN c = a.tempdoubleMat(m, k);
            dot(in a, in b, ref c, transposeA);
            return c;
        }

        // ---- transpose -> matrix ----

        // ref-dest primitive. Guard: T must not alias A. Transpose is a permutation, so
        // even though each entry is read once, writing T[i,j] would clobber A[i,j] which
        // is still needed as T[j,i].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void trans(in doubleMxN A, ref doubleMxN T)
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
        public static doubleMxN trans(doubleMxN A)
        {
            var T = A.tempdoubleMat(A.N_Cols, A.M_Rows, true);
            trans(in A, ref T);
            return T;
        }

        // Applies a single Householder reflection matrix -style transform directly to `matrix`:
        // matrix -= (2 / uᵀu) · u·uᵀ · matrix. Standalone primitive (not part of QR's internal
        // incremental reflector pipeline — see QR.applyReflectorRight for that).
        public static void householderInpl(ref doubleMxN matrix, in doubleN u)
        {
            if(matrix.IsSquare == false)
                throw new ArgumentException("Linear_OP.householderInpl: Matrix must be square");

            if(matrix.M_Rows < matrix.N_Cols)
                throw new ArgumentException("Linear_OP.householderInpl: Matrix must be square or tall (more or equal rows than cols)");

            var maxDim = math.max(matrix.M_Rows, matrix.N_Cols);

            if(u.N < maxDim)
                throw new ArgumentException("Linear_OP.householderInpl: Vector must be at least as long as the largest dimension of the matrix");

            double vTv = dot(u, u); // Inline dot product calculation

            // Degenerate (zero / near-zero) reflector -> identity transform; leave matrix unchanged.
            // NaN-safe (!(vTv > t) is true for NaN); avoids 2/0 = Inf poisoning the matrix.
            if (!(vTv > Consts.doubleZeroThreshold))
                return;

            double scaleFactor = 2 / vTv;

            for (int i = 0; i < matrix.M_Rows; i++)
            {
                for (int j = 0; j < matrix.N_Cols; j++)
                {
                    double vvT_element = scaleFactor * u[i] * u[j];
                    matrix[i, j] -= vvT_element; // Apply directly to matrix
                }
            }
        }
    }
}
