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

        // PA = L * U
        // U is originally A
        // L is originally I
        // P is pivot, that is reset, and is modified in place
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

                // Find largest pivot in rows
                for(int r = k + 1; r < m; r++) {
                    fProxy absValue = math.abs(U[r, k]);
                    if(absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Swap rows
                P.Swap(k, pivotIndex);

                // swap submatrix U rows
                SwapOP.Rows(ref U, k, pivotIndex, k, m);
                
                // swap already calculated L rows
                SwapOP.Rows(ref L, k, pivotIndex, 0, k);

                if (L[k,k] != 1f)
                    throw new Exception("L[k,k] must be 1");

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

        // A = LU
        // LU is originally A
        // P is pivot, that is reset, and is modified in place
        public static void luDecompositionInplace(ref fProxyMxN LU, ref Pivot P) {

            if (!LU.IsSquare)
                throw new System.Exception("luDecomposition: LU (A) needs to be square");

            int m = LU.M_Rows;

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                fProxy pivotValue = math.abs(LU[P[k], k]);

                // Find largest pivot in rows
                for (int r = k + 1; r < m; r++) {
                    fProxy absValue = math.abs(LU[P[r], k]); 
                    if (absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Swap rows
                P.Swap(k, pivotIndex);

                int Pk = P[k];

                // Calculate L and U
                fProxy Ukk = LU[Pk, k];
                for (int j = k + 1; j < m; j++) {

                    int Pj = P[j];

                    fProxy Ljk = LU[Pj, k] / Ukk;

                    for (int i = k; i < m; i++) {
                        LU[Pj, i] -= Ljk * LU[Pk, i];
                    }

                    LU[Pj, k] = Ljk;
                }
            }
        }

        public static void luDecompositionInplace2(ref fProxyMxN LU, ref Pivot P) {

            if (!LU.IsSquare)
                throw new System.Exception("luDecomposition: LU (A) needs to be square");

            int m = LU.M_Rows;

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                fProxy pivotValue = math.abs(LU[P[k], k]);

                // Find largest pivot in rows
                for (int r = k + 1; r < m; r++) {
                    fProxy absValue = math.abs(LU[P[r], k]);
                    if (absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Swap rows
                P.Swap(k, pivotIndex);

                SwapOP.Rows(ref LU, k, pivotIndex);

                // Calculate L and U
                fProxy Ukk = LU[k, k];
                for (int j = k + 1; j < m; j++) {

                    fProxy Ljk = LU[j, k] / Ukk;

                    for (int i = k; i < m; i++) {
                        LU[j, i] -= Ljk * LU[k, i];
                    }

                    LU[j, k] = Ljk;
                }
            }
        }

        // solve LUx = b for x
        // both L and U are in single matrix LU
        public static void LUSolve(ref fProxyMxN LU, ref Pivot pivot, ref fProxyN b) {

            pivot.ApplyInverseVec(ref b);
            
            // Solve Ly = b
            Solvers.SolveLowerTriangularLU(ref LU, ref pivot, ref b);
            // Solve Ux = y
            Solvers.SolveUpperTriangularLU(ref LU, ref pivot, ref b);

        }

        // solve LUx = b for x
        // both L and U are in single matrix LU
        public static void LUSolve2(ref fProxyMxN LU, ref Pivot pivot, ref fProxyN b) {

            pivot.ApplyVec(ref b);

            // Solve Ly = b
            Solvers.SolveLowerTriangularLU(ref LU, ref b);
            // Solve Ux = y
            Solvers.SolveUpperTriangular(ref LU, ref b);

        }

        // Solve LUx = Pb for x
        // b is overwritten with x
        public static void LUSolve(ref fProxyMxN L, ref fProxyMxN U, in Pivot pivot, ref fProxyN b) {

            // apply pivot to b
            pivot.ApplyInverseVec(ref b);

            // Solver linear system LUx = b, b is overwritten with x

            // Solve Ly = b
            Solvers.SolveLowerTriangular(ref L, ref b);
            // Solve Ux = y
            Solvers.SolveUpperTriangular(ref U, ref b);

        }
    }
}
