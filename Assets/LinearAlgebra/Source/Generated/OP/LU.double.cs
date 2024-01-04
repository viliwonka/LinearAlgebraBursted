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
        /// U is initially original A matrix, L is initially identity matrix
        /// A = L * U
        /// U will be modified in place
        /// L will be modified in place
        public static void luDecomposition(ref doubleMxN U, ref doubleMxN L)
        {
            if (!U.IsSquare)
                throw new System.Exception("luDecomposition: A needs to be square, so has to be LU");

            int dim = U.M_Rows;

            // Pivot used for row swapping
            // Initialized as array with [0, 1, 2, 3, 4,... ] indices that will be swapped
            Pivot RP = new Pivot(dim, Allocator.Temp);
            RP.Reset();
            for(int c = 0; c < dim; c++) {

                /*int pivotIndex = k;
                double pivotValue = math.abs(U[RP[k], k]);

                // Find best pivot in rows
                for(int r = k + 1; r < dim; r++) {
                    double absValue = math.abs(U[RP[r], k]);
                    if(absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Swap rows
                RP.Swap(k, pivotIndex);*/

                // Calculate L and U
                double Lkk = U[RP[c], c];
                // Lkk should be checked for zero, but we assume that the matrix is non-singular

                for(int r = c + 1; r < dim; r++) {

                    double Lki = U[RP[r], c] / Lkk;
                    L[RP[r], c] = Lki;

                    for (int j = c + 1; j < dim; j++) {
                        U[RP[r], j] -= Lki * U[RP[c], j];
                    }
                }

                for(int j = c + 1; j < dim; j++) {
                    U[RP[c], j] = U[RP[c], j] / Lkk;
                }
            }

            RP.Dispose();
        }

    }

}
