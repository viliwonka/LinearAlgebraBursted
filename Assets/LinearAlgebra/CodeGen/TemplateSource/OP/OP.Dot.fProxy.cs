using System.Runtime.CompilerServices;
using System;

using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// Dot products, outer product, matrix multiply, transpose, and in-place Householder reflection.
    /// </summary>
    public static partial class Blas {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy dot(fProxyN a, fProxyN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return UnsafeOP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy dot(fProxyN a, fProxyN b, int start, int end = -1) {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            if(end == -1)
                end = a.N;

            unsafe {
                return UnsafeOP.vecDotRange(a.Data.Ptr, b.Data.Ptr, start, end);
            }
        }

        // ---- outer product: a (col) * b (row) -> M x N ----

        // ref-dest primitive. No alias guard: result is a matrix, inputs are vectors,
        // so they can never share a buffer.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void outerDot(in fProxyN a, in fProxyN b, ref fProxyMxN result)
        {
            if (result.M_Rows != a.N || result.N_Cols != b.N)
                throw new ArgumentException("outerDot: result must be a.N x b.N");

            unsafe
            {
                UnsafeOP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN outerDot(fProxyN a, fProxyN b)
        {
            fProxyMxN result = a.fProxyTempMat(a.N, b.N, true);
            outerDot(in a, in b, ref result);
            return result;
        }

        // ---- matrix * vector -> vector ----

        // ref-dest primitive. Guard: result must not alias x (each x[k] feeds every row).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in fProxyMxN A, in fProxyN x, ref fProxyN result)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (result.N != A.M_Rows)
                throw new ArgumentException("dot: result.N must equal A.M_Rows");

            unsafe {
                if (result.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("dot: result must not alias x");

                // matVecDot accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(result.Data.Ptr, (long)result.Data.Length * UnsafeUtility.SizeOf<fProxy>());

                UnsafeOP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN dot(fProxyMxN A, fProxyN x)
        {
            fProxyN result = x.fProxyTempVec(A.M_Rows);
            dot(in A, in x, ref result);
            return result;
        }

        // y = A x, PLUS dot(x, y) computed as part of the same call. Composes a plain matVecDot pass
        // with a separate vecDot pass rather than a single fused kernel. Requires A square (the
        // trailing dot(x, y) needs x.N == y.N).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy dotSelf(in fProxyMxN A, in fProxyN x, ref fProxyN y)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (y.N != A.M_Rows)
                throw new ArgumentException("dotSelf: y.N must equal A.M_Rows");

            if (!A.IsSquare)
                throw new ArgumentException("dotSelf: A must be square");

            unsafe
            {
                if (y.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("dotSelf: y must not alias x");

                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<fProxy>());
                UnsafeOP.matVecDot(A.Data.Ptr, x.Data.Ptr, y.Data.Ptr, A.M_Rows, A.N_Cols);
            }
            return dot(x, y);
        }

        // ---- vector * matrix -> vector ----

        // ref-dest primitive. Guard: result must not alias y (each y[i] feeds every column).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in fProxyN y, in fProxyMxN A, ref fProxyN result)
        {
            Assume.SameDim(A.M_Rows, y.N);

            if (result.N != A.N_Cols)
                throw new ArgumentException("dot: result.N must equal A.N_Cols");

            unsafe {
                if (result.Data.Ptr == y.Data.Ptr)
                    throw new ArgumentException("dot: result must not alias y");

                // vecMatDot zeroes the destination itself before accumulating.
                UnsafeOP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyN dot(fProxyN y, fProxyMxN A)
        {
            fProxyN result = y.fProxyTempVec(A.N_Cols);
            dot(in y, in A, ref result);
            return result;
        }

        // ---- matrix * matrix -> matrix ----

        // ref-dest primitive. Guard: c must not alias a or b (each input entry feeds a
        // whole row/column of the product).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dot(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c, bool transposeA = false)
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
                UnsafeUtility.MemClear(c.Data.Ptr, (long)c.Data.Length * UnsafeUtility.SizeOf<fProxy>());

                if (a.Data.Ptr == b.Data.Ptr && transposeA)
                {
                    // Aᵀ·A: dedicated SYRK-shape kernel — one input pointer, no copy.
                    UnsafeOP.matAtA(a.Data.Ptr, c.Data.Ptr, m, n);
                }
                else if (a.Data.Ptr == b.Data.Ptr)
                {
                    // A·A: matMatDot promises [NoAlias] on every pointer and has no single-input
                    // twin, so feed it a Temp copy of b — an O(n²) copy against the O(n³) product.
                    var bCopy = new fProxyMxN(b.M_Rows, b.N_Cols, Unity.Collections.Allocator.Temp, true);
                    bCopy.Data.CopyFrom(b.Data);
                    UnsafeOP.matMatDot(a.Data.Ptr, bCopy.Data.Ptr, c.Data.Ptr, m, n, k);
                    bCopy.Dispose();
                }
                else if (transposeA)
                    UnsafeOP.matMatDotTransA(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
                else
                    UnsafeOP.matMatDot(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
            }
        }

        /// <summary>
        /// C = Aᵀ·B for products that are symmetric BY CONSTRUCTION (B = Q·A with symmetric Q —
        /// the ΓᵀQΓ / ZᵀQZ / MᵀRM shapes). Computes the upper triangle and mirrors it, so it runs
        /// ~2x faster than <c>dot(..., transposeA: true)</c> and the output is exactly symmetric.
        /// CALLER CONTRACT: the true product must be symmetric — asymmetric inputs get a
        /// symmetrized wrong answer. Requires a.N_Cols == b.N_Cols (square C); destination must
        /// not alias an input. C is overwritten.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dotSym(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c)
        {
            Assume.SameDim(a.M_Rows, b.M_Rows);
            if (a.N_Cols != b.N_Cols)
                throw new ArgumentException("dotSym: a.N_Cols must equal b.N_Cols (square result)");

            int m = a.N_Cols, n = a.M_Rows, k = b.N_Cols;
            if (c.M_Rows != m || c.N_Cols != k)
                throw new ArgumentException("dotSym: destination must be m x m");

            unsafe
            {
                if (c.Data.Ptr == a.Data.Ptr || c.Data.Ptr == b.Data.Ptr)
                    throw new ArgumentException("dotSym: destination must not alias an input");

                UnsafeUtility.MemClear(c.Data.Ptr, (long)c.Data.Length * UnsafeUtility.SizeOf<fProxy>());

                if (a.Data.Ptr == b.Data.Ptr)
                    UnsafeOP.matAtA(a.Data.Ptr, c.Data.Ptr, m, n);
                else
                    UnsafeOP.matMatDotTransASym(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
            }
        }

        // Row-limited mat*mat for block operator applies (LOBPCG): C[0:rows,:] = a[0:rows,:] · b,
        // i.e. only the first `rows` rows of a are read and only the first `rows` rows of c are
        // written. c's remaining rows [rows, c.M_Rows) are left UNTOUCHED -- callers (e.g. LOBPCG)
        // keep locked-pair data there that a whole-buffer dot would clobber. c must not alias a or b.
        // Contract mirrors the non-transposed dot: contracts a.N_Cols against b.M_Rows.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void dotRows(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c, int rows)
        {
            Assume.SameDim(a.N_Cols, b.M_Rows);
            if (c.N_Cols != b.N_Cols)
                throw new ArgumentException("dotRows: destination cols must equal b.N_Cols");
            if (rows < 0 || rows > a.M_Rows || rows > c.M_Rows)
                throw new ArgumentException("dotRows: rows must be within a.M_Rows and c.M_Rows");

            int nn = a.N_Cols, kk = b.N_Cols;
            unsafe
            {
                if (c.Data.Ptr == a.Data.Ptr || c.Data.Ptr == b.Data.Ptr)
                    throw new ArgumentException("dotRows: destination must not alias an input");

                // matMatDot accumulates (+=), so zero just the rows we are about to write.
                UnsafeUtility.MemClear(c.Data.Ptr, (long)rows * kk * UnsafeUtility.SizeOf<fProxy>());
                UnsafeOP.matMatDot(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, rows, nn, kk);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN dot(fProxyMxN a, fProxyMxN b, bool transposeA = false)
        {
            int m = transposeA ? a.N_Cols : a.M_Rows;
            int k = b.N_Cols;

            fProxyMxN c = a.fProxyTempMat(m, k);
            dot(in a, in b, ref c, transposeA);
            return c;
        }

        // ---- transpose -> matrix ----

        // ref-dest primitive. Guard: T must not alias A. Transpose is a permutation, so
        // even though each entry is read once, writing T[i,j] would clobber A[i,j] which
        // is still needed as T[j,i].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void trans(in fProxyMxN A, ref fProxyMxN T)
        {
            if (T.M_Rows != A.N_Cols || T.N_Cols != A.M_Rows)
                throw new ArgumentException("trans: destination must be A.N_Cols x A.M_Rows");

            unsafe
            {
                if (T.Data.Ptr == A.Data.Ptr)
                    throw new ArgumentException("trans: destination must not alias the input");

                UnsafeOP.matTrans(A.Data.Ptr, T.Data.Ptr, A.M_Rows, A.N_Cols);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxyMxN trans(fProxyMxN A)
        {
            var T = A.fProxyTempMat(A.N_Cols, A.M_Rows, true);
            trans(in A, ref T);
            return T;
        }

        // Applies a single Householder reflection matrix -style transform directly to `matrix`:
        // matrix -= (2 / uᵀu) · u·uᵀ · matrix. Standalone primitive (not part of QR's internal
        // incremental reflector pipeline — see QR.applyReflectorRight for that).
        public static void householderInPlace(ref fProxyMxN matrix, in fProxyN u)
        {
            if(matrix.IsSquare == false)
                throw new ArgumentException("Blas.householderInPlace: Matrix must be square");

            if(u.N < matrix.N_Cols)
                throw new ArgumentException("Blas.householderInPlace: Vector must be at least as long as the matrix dimension");

            fProxy vTv = dot(u, u);

            // Degenerate (zero / near-zero) reflector -> identity transform; leave matrix unchanged.
            // NaN-safe (!(vTv > t) is true for NaN); avoids 2/0 = Inf poisoning the matrix.
            if (!(vTv > Consts.fProxyZeroThreshold))
                return;

            fProxy scaleFactor = 2 / vTv;

            for (int j = 0; j < matrix.N_Cols; j++)
            {
                fProxy proj = 0;
                for (int i = 0; i < matrix.M_Rows; i++)
                    proj += u[i] * matrix[i, j];

                fProxy scaledProj = scaleFactor * proj;
                for (int i = 0; i < matrix.M_Rows; i++)
                    matrix[i, j] -= u[i] * scaledProj;
            }
        }
    }
}
