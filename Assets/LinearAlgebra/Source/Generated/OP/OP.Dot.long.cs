#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using System;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class longOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long dot(longN a, longN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return UnsafeOP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN outerDot(longN a, longN b)
        {
            longMxN result = a.templongMat(a.N, b.N, true);
            
            unsafe
            {
                UnsafeOP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN dot(longMxN A, longN x)
        {
            Assume.SameDim(A.N_Cols, x.N);

            longN result = x.templongVec(A.M_Rows);

            unsafe {
                
                UnsafeOP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longN dot(longN y, longMxN A)
        {
            Assume.SameDim(A.M_Rows, y.N);

            longN result = y.templongVec(A.N_Cols);

            unsafe {
                UnsafeOP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static longMxN dot(longMxN a, longMxN b, bool transposeA = false)
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
            longMxN c = a.templongMat(m, k);

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
        public static longMxN trans(longMxN A)
        {
            var T = A.templongMat(A.N_Cols, A.M_Rows, true);

            unsafe
            {
                UnsafeOP.matTrans(A.Data.Ptr, T.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return T;
        }
    }
}
