#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System.Runtime.CompilerServices;
using System;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class intOP {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int dot(intN a, intN b)
        {
            if (a.N != b.N)
                throw new ArgumentException("dot: Vector must have same dimension");

            unsafe {
                return UnsafeOP.vecDot(a.Data.Ptr, b.Data.Ptr, a.Data.Length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN outerDot(intN a, intN b)
        {
            intMxN result = a.tempintMat(a.N, b.N, true);
            
            unsafe
            {
                UnsafeOP.vecOuterDot(a.Data.Ptr, b.Data.Ptr, result.Data.Ptr, a.N, b.N);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN dot(intMxN A, intN x)
        {
            Assume.SameDim(A.N_Cols, x.N);

            intN result = x.tempintVec(A.M_Rows);

            unsafe {
                
                UnsafeOP.matVecDot(A.Data.Ptr, x.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intN dot(intN y, intMxN A)
        {
            Assume.SameDim(A.M_Rows, y.N);

            intN result = y.tempintVec(A.N_Cols);

            unsafe {
                UnsafeOP.vecMatDot(y.Data.Ptr, A.Data.Ptr, result.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static intMxN dot(intMxN a, intMxN b, bool transposeA = false)
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
            intMxN c = a.tempintMat(m, k);

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
        public static intMxN trans(intMxN A)
        {
            var T = A.tempintMat(A.N_Cols, A.M_Rows, true);

            unsafe
            {
                UnsafeOP.matTrans(A.Data.Ptr, T.Data.Ptr, A.M_Rows, A.N_Cols);
            }

            return T;
        }
    }
}
