#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.CompilerServices;
using LinearAlgebra.Internal;
//singularFile//

namespace LinearAlgebra
{
    public static class Swap_OP {

        

        // just for completeness, swap two elements in a vector
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Vec(ref floatN vec, int i, int j) {

            if (i < 0 || i >= vec.N) {
                throw new ArgumentOutOfRangeException("i and j must be bounded inside vector dimensions");
            }

            if (j < 0 || j >= vec.N) {
                throw new ArgumentOutOfRangeException("i and j must be bounded inside vector dimensions");
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
                throw new System.ArgumentException("i and j must be bounded inside matrix rows dimensions");

            if (end == -1)
                end = mat.N_Cols;

            if (start < 0 || start > end || end > mat.N_Cols)
                throw new System.ArgumentException("start and end must satisfy 0 <= start <= end <= N_Cols");

            if (i == j) {
                // do nothing
                return;
            }

            unsafe {

                Unsafe_OP.swapRows(mat.Data.Ptr, i, j, mat.N_Cols, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Columns(ref floatMxN mat, int i, int j, int start = 0, int end = -1) {

            if(i < 0 || i >= mat.N_Cols || j < 0 || j >= mat.N_Cols)
                throw new System.ArgumentException("i and j must be bounded inside matrix columns dimensions");

            if (end == -1)
                end = mat.M_Rows;

            if (start < 0 || start > end || end > mat.M_Rows)
                throw new System.ArgumentException("start and end must satisfy 0 <= start <= end <= M_Rows");

            if(i == j) {
                // do nothing
                return;
            }

            unsafe {

                Unsafe_OP.swapColumns(mat.Data.Ptr, i, j, mat.M_Rows, mat.N_Cols, start, end);
            }
        }

        

        // just for completeness, swap two elements in a vector
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Vec(ref doubleN vec, int i, int j) {

            if (i < 0 || i >= vec.N) {
                throw new ArgumentOutOfRangeException("i and j must be bounded inside vector dimensions");
            }

            if (j < 0 || j >= vec.N) {
                throw new ArgumentOutOfRangeException("i and j must be bounded inside vector dimensions");
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
                throw new System.ArgumentException("i and j must be bounded inside matrix rows dimensions");

            if (end == -1)
                end = mat.N_Cols;

            if (start < 0 || start > end || end > mat.N_Cols)
                throw new System.ArgumentException("start and end must satisfy 0 <= start <= end <= N_Cols");

            if (i == j) {
                // do nothing
                return;
            }

            unsafe {

                Unsafe_OP.swapRows(mat.Data.Ptr, i, j, mat.N_Cols, start, end);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Columns(ref doubleMxN mat, int i, int j, int start = 0, int end = -1) {

            if(i < 0 || i >= mat.N_Cols || j < 0 || j >= mat.N_Cols)
                throw new System.ArgumentException("i and j must be bounded inside matrix columns dimensions");

            if (end == -1)
                end = mat.M_Rows;

            if (start < 0 || start > end || end > mat.M_Rows)
                throw new System.ArgumentException("start and end must satisfy 0 <= start <= end <= M_Rows");

            if(i == j) {
                // do nothing
                return;
            }

            unsafe {

                Unsafe_OP.swapColumns(mat.Data.Ptr, i, j, mat.M_Rows, mat.N_Cols, start, end);
            }
        }

        

    }
}
