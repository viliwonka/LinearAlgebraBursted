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

        // LU decomposition with no pivoting
        /// <summary>
        /// U = A
        /// L = I
        /// A = L * U
        /// U will be modified in place
        /// L will be modified in place
        public static void luDecompositionNoPivot(ref fProxyMxN U, ref fProxyMxN L)
        {
            if (!U.IsSquare)
                throw new System.Exception("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.Exception("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.Exception("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;
                        
            for(int k = 0; k < m - 1; k++) {

                // Calculate L and U
                fProxy Ukk = U[k, k];
                
                for(int j = k + 1; j < m; j++) {

                    fProxy Ljk = U[j, k] / Ukk;
                    
                    // fine
                    L[j, k] = Ljk;

                    for (int i = k; i < m; i++) {
                        U[j, i] -= Ljk * U[k, i];
                    }
                }
            }
        }

        public static void luDecomposition(ref fProxyMxN U, ref fProxyMxN L, ref Pivot P) {
            if (!U.IsSquare)
                throw new System.Exception("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.Exception("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.Exception("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                fProxy pivotValue = math.abs(U[k, k]);

                // Find best pivot in rows
                for(int r = k + 1; r < m; r++) {
                    fProxy absValue = math.abs(U[r, k]);
                    if(absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Swap rows
                P.Swap(k, pivotIndex);

                // have to test all the swapping ops
                SwapOP.Rows(ref U, k, pivotIndex, k);
                SwapOP.Rows(ref L, k, pivotIndex, 0, k - 1);

                // Calculate L and U
                fProxy Ukk = U[k, k];

                for (int j = k + 1; j < m; j++) {

                    fProxy Ljk = U[j, k] / Ukk;

                    L[j, k] = Ljk;
                        
                    for (int i = k; i < m; i++) {
                        U[j, i] -= Ljk * U[k, i];
                    }
                }
            }
        }

    }

}
