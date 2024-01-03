#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using System;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class shortOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short dot(shortN a, shortN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return UnsafeOP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN outerDot(shortN a, shortN b)
        {
            shortMxN result = a.tempshortMat(a.N, b.N, true);
            
            unsafe
            {
                UnsafeOP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN dot(shortMxN A, shortN x)
        {
            Assume.SameDim(A.N_Cols, x.N);

            shortN result = x.tempshortVec(A.M_Rows);

            unsafe {
                
                UnsafeOP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortN dot(shortN y, shortMxN A)
        {
            Assume.SameDim(A.M_Rows, y.N);

            shortN result = y.tempshortVec(A.N_Cols);

            unsafe {
                UnsafeOP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static shortMxN dot(shortMxN a, shortMxN b, bool transposeA = false)
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
            shortMxN c = a.tempshortMat(m, k);

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
        public static shortMxN trans(shortMxN A)
        {
            var T = A.tempshortMat(A.N_Cols, A.M_Rows, true);

            unsafe
            {
                UnsafeOP.matTrans(A.Data.Ptr, T.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return T;
        }
    }
}
