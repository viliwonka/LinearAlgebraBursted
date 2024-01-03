#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using System;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class doubleOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double dot(doubleN a, doubleN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return UnsafeOP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double dot(doubleN a, doubleN b, int start, int end = -1) {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            if(end == -1)
                end = a.N;

            unsafe {
                return UnsafeOP.vecDotRange(a.Data.Ptr, b.Data.Ptr, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN outerDot(doubleN a, doubleN b)
        {
            doubleMxN result = a.tempdoubleMat(a.N, b.N, true);

            unsafe
            {
                UnsafeOP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN dot(doubleMxN A, doubleN x)
        {
            Assume.SameDim(A.N_Cols, x.N);

            doubleN result = x.tempdoubleVec(A.M_Rows);

            unsafe {
                
                UnsafeOP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN dot(doubleN y, doubleMxN A)
        {
            Assume.SameDim(A.M_Rows, y.N);

            doubleN result = y.tempdoubleVec(A.N_Cols);

            unsafe {
                UnsafeOP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN dot(doubleMxN a, doubleMxN b, bool transposeA = false)
        {
            if(transposeA)
                Assume.SameDim(a.N_Cols, b.N_Cols);
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
            doubleMxN c = a.tempdoubleMat(m, k);

            unsafe
            {
                if(transposeA)
                    UnsafeOP.matMatDotTransA(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
                else
                    UnsafeOP.matMatDot(a.Data.Ptr, b.Data.Ptr, c.Data.Ptr, m, n, k);
            }

            return c;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleMxN trans(doubleMxN A)
        {
            var T = A.tempdoubleMat(A.N_Cols, A.M_Rows, true);

            unsafe
            {
                UnsafeOP.matTrans(A.Data.Ptr, T.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return T;
        }
    }
}
