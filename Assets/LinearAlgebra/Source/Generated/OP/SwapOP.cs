#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;
//singularFile//

namespace LinearAlgebra
{
    public static class SwapOP {

        

        // just for completeness, swap two elements in a vector
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Vec(ref floatN vec, int i, int j) {

            if (i < 0 || i >= vec.N) {
                throw new System.Exception("i and j must be bounded inside vector dimensions");
            }

            if (i == j) {
                // do nothing
                return;
            }

            float temp = vec[i];
            vec[i] = vec[j];
            vec[j] = temp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Rows(ref floatMxN mat, int i, int j, int start = 0, int end = -1) {

            if(i < 0 || i >= mat.M_Rows || j < 0 || j >= mat.M_Rows)
                throw new System.Exception("i and j must be bounded inside matrix rows dimensions");
            
            if(end != -1 && start > end)
                throw new System.Exception("start must be less than end");
            
            if(start < 0)
                throw new System.Exception("start must be greater than 0");

            if (end > mat.N_Cols)
                throw new System.Exception("end must be less than matrix columns");
                
            if (i == j) {
                // do nothing
                return;
            }

            unsafe {

                UnsafeOP.swapRows(mat.Data.Ptr, i, j, mat.N_Cols, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Columns(ref floatMxN mat, int i, int j, int start = 0, int end = -1) {

            if(i < 0 || i >= mat.N_Cols || j < 0 || j >= mat.N_Cols)
                throw new System.Exception("i and j must be bounded inside matrix columns dimensions");
            
            if (end != -1 && start > end)
                throw new System.Exception("start must be less than end");
            
            if(start < 0) 
                throw new System.Exception("start must be greater than 0");
            
            if (end > mat.M_Rows)
                throw new System.Exception("end must be less than matrix rows");
            
            if(i == j) {
                // do nothing
                return;
            }

            unsafe {

                UnsafeOP.swapColumns(mat.Data.Ptr, i, j, mat.M_Rows, mat.N_Cols);
            }
        }

        

        // just for completeness, swap two elements in a vector
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Vec(ref doubleN vec, int i, int j) {

            if (i < 0 || i >= vec.N) {
                throw new System.Exception("i and j must be bounded inside vector dimensions");
            }

            if (i == j) {
                // do nothing
                return;
            }

            double temp = vec[i];
            vec[i] = vec[j];
            vec[j] = temp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Rows(ref doubleMxN mat, int i, int j, int start = 0, int end = -1) {

            if(i < 0 || i >= mat.M_Rows || j < 0 || j >= mat.M_Rows)
                throw new System.Exception("i and j must be bounded inside matrix rows dimensions");
            
            if(end != -1 && start > end)
                throw new System.Exception("start must be less than end");
            
            if(start < 0)
                throw new System.Exception("start must be greater than 0");

            if (end > mat.N_Cols)
                throw new System.Exception("end must be less than matrix columns");
                
            if (i == j) {
                // do nothing
                return;
            }

            unsafe {

                UnsafeOP.swapRows(mat.Data.Ptr, i, j, mat.N_Cols, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Columns(ref doubleMxN mat, int i, int j, int start = 0, int end = -1) {

            if(i < 0 || i >= mat.N_Cols || j < 0 || j >= mat.N_Cols)
                throw new System.Exception("i and j must be bounded inside matrix columns dimensions");
            
            if (end != -1 && start > end)
                throw new System.Exception("start must be less than end");
            
            if(start < 0) 
                throw new System.Exception("start must be greater than 0");
            
            if (end > mat.M_Rows)
                throw new System.Exception("end must be less than matrix rows");
            
            if(i == j) {
                // do nothing
                return;
            }

            unsafe {

                UnsafeOP.swapColumns(mat.Data.Ptr, i, j, mat.M_Rows, mat.N_Cols);
            }
        }

        

    }
}
