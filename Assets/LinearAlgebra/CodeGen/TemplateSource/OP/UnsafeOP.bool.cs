//singularFile//
#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;


namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap rows in a matrix 
        public static void swapRows([NoAlias] bool* target, int rowA, int rowB, int nCols, int colStart = 0, int colEnd = -1) {

            int rowIndexA = rowA * nCols;
            int rowIndexB = rowB * nCols;

            if (colEnd == -1)
                colEnd = nCols;

            for (int i = colStart; i < colEnd; i++) {
                bool temp = target[rowIndexA + i];
                target[rowIndexA + i] = target[rowIndexB + i];
                target[rowIndexB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]

        // Swap columns in a matrix
        public static void swapColumns([NoAlias] bool* target, int colA, int colB, int nRows, int nCols, int start = 0, int end = -1) {
            int startA = colA;
            int startB = colB;

            if (end == -1)
                end = nRows;

            for (int i = start; i < end; i++) {
                bool temp = target[startA + i * nCols];
                target[startA + i * nCols] = target[startB + i * nCols];
                target[startB + i * nCols] = temp;
            }
        }
    }
}