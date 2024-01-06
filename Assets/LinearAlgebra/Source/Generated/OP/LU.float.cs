#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class LU {

        // LU decomposition with partial pivoting (row pivoting)
        /// <summary>
        /// U = A
        /// L = I
        /// A = L * U
        /// U will be modified in place
        /// L will be modified in place
        public static void luDecomposition(ref floatMxN U, ref floatMxN L)
        {
            if (!U.IsSquare)
                throw new System.Exception("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.Exception("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.Exception("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;

            // Pivot used for row swapping
            // Initialized as array with [0, 1, 2, 3, 4,... ] indices that will be swapped
            Pivot RP = new Pivot(m, Allocator.Temp);
            
            for(int k = 0; k < m - 1; k++) {

                /*int pivotIndex = k;
                float pivotValue = math.abs(U[RP[k], k]);

                // Find best pivot in rows
                for(int r = k + 1; r < dim; r++) {
                    float absValue = math.abs(U[RP[r], k]);
                    if(absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Swap rows
                RP.Swap(k, pivotIndex);*/

                // Calculate L and U
                float Ukk = U[RP[k], k];
                
                for(int j = k + 1; j < m; j++) {

                    float Ljk = U[RP[j], k] / Ukk;
                    
                    // fine
                    L[RP[j], k] = Ljk;

                    for (int i = k; i < m; i++) {
                        U[RP[j], i] -= Ljk * U[RP[k], i];
                    }
                }
            }

            RP.Dispose();
        }

    }

}
