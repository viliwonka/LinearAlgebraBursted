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
        /// U = A (input matrix, overwritten with upper triangular U)
        /// L = I (identity matrix, overwritten with lower triangular L)
        /// A = L * U
        /// Returns true on success; false if a zero pivot is encountered (singular matrix).
        /// On false: no NaN/Inf is written.
        /// </summary>
        public static bool luDecompositionNoPivot(ref floatMxN U, ref floatMxN L)
        {
            if (!U.IsSquare)
                throw new System.ArgumentException("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.ArgumentException("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.ArgumentException("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;

            if (m == 0) return true;

            for(int k = 0; k < m - 1; k++) {

                // Calculate L and U
                float Ukk = U[k, k];

                if (Ukk == 0)
                    return false;

                for(int j = k + 1; j < m; j++) {

                    float Ljk = U[j, k] / Ukk;

                    L[j, k] = Ljk;

                    for (int i = k + 1; i < m; i++) {
                        U[j, i] -= Ljk * U[k, i];
                    }

                    // U is exactly upper-triangular
                    U[j, k] = 0;
                }
            }

            // Check last diagonal
            if (U[m - 1, m - 1] == 0)
                return false;

            return true;
        }

        // PA = L * U
        // U is originally A
        // L is originally I
        // P is pivot, that is reset, and is modified in place
        /// <summary>
        /// Performs LU decomposition with partial pivoting.
        /// Returns true on success; false if a zero pivot is encountered (singular matrix).
        /// On false: no NaN/Inf is written, P remains a valid permutation.
        /// </summary>
        public static bool luDecomposition(ref floatMxN U, ref floatMxN L, ref Pivot P) {
            if (!U.IsSquare)
                throw new System.ArgumentException("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.ArgumentException("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.ArgumentException("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;

            if (P.N != m) throw new System.ArgumentException("pivot size must equal matrix dimension");

            P.Reset();

            if (m == 0) return true;

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                float pivotValue = math.abs(U[k, k]);

                // Find largest pivot in rows
                for(int r = k + 1; r < m; r++) {
                    float absValue = math.abs(U[r, k]);
                    if(absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Check for zero pivot before any division
                if (pivotValue == 0)
                    return false;

                // Swap rows
                P.Swap(k, pivotIndex);

                // swap submatrix U rows
                SwapOP.Rows(ref U, k, pivotIndex, k, m);

                // swap already calculated L rows
                SwapOP.Rows(ref L, k, pivotIndex, 0, k);

                // Calculate L and U
                float Ukk = U[k, k];
                for (int j = k + 1; j < m; j++) {

                    float Ljk = U[j, k] / Ukk;

                    L[j, k] = Ljk;

                    for (int i = k + 1; i < m; i++) {
                        U[j, i] -= Ljk * U[k, i];
                    }

                    // U is exactly upper-triangular
                    U[j, k] = 0;
                }
            }

            // Check last diagonal
            if (U[m - 1, m - 1] == 0)
                return false;

            return true;
        }

        // A = LU
        // LU is originally A
        // P is pivot, that is reset, and is modified in place
        /// <summary>
        /// Performs LU decomposition inplace with partial pivoting (compact LU form).
        /// Factor row i lives at physical row P[i].
        /// Returns true on success; false if a zero pivot is encountered (singular matrix).
        /// On false: no NaN/Inf is written, P remains a valid permutation.
        /// </summary>
        public static bool luDecompositionInplace(ref floatMxN LU, ref Pivot P) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("luDecomposition: LU (A) needs to be square");

            int m = LU.M_Rows;

            if (P.N != m) throw new System.ArgumentException("pivot size must equal matrix dimension");

            P.Reset();

            if (m == 0) return true;

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                float pivotValue = math.abs(LU[P[k], k]);

                // Find largest pivot in rows
                for (int r = k + 1; r < m; r++) {
                    float absValue = math.abs(LU[P[r], k]);
                    if (absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Check for zero pivot before any division
                if (pivotValue == 0)
                    return false;

                // Swap rows
                P.Swap(k, pivotIndex);

                int Pk = P[k];

                // Calculate L and U
                float Ukk = LU[Pk, k];
                for (int j = k + 1; j < m; j++) {

                    int Pj = P[j];

                    float Ljk = LU[Pj, k] / Ukk;

                    for (int i = k + 1; i < m; i++) {
                        LU[Pj, i] -= Ljk * LU[Pk, i];
                    }

                    LU[Pj, k] = Ljk;
                }
            }

            // Check last diagonal (the k < m-1 loop never inspects it)
            if (LU[P[m - 1], m - 1] == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Solve LUx = b for x using the compact inplace LU form with pivot.
        /// b is overwritten with x.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        public static void LUSolve(ref floatMxN LU, in Pivot pivot, ref floatN b) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("LUSolve: LU must be square");

            if (b.N != LU.M_Rows)
                throw new System.ArgumentException("LUSolve: b.N must equal LU.M_Rows");

            if (pivot.N != b.N)
                throw new System.ArgumentException("LUSolve: pivot.N must equal b.N");

            pivot.ApplyInverseVec(ref b);

            // Solve Ly = b
            Solvers.SolveLowerTriangularLU(ref LU, in pivot, ref b);
            // Solve Ux = y
            Solvers.SolveUpperTriangularLU(ref LU, in pivot, ref b);

        }

        /// <summary>
        /// Solve LUx = Pb for x using separate L and U matrices with pivot.
        /// b is overwritten with x.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        public static void LUSolve(ref floatMxN L, ref floatMxN U, in Pivot pivot, ref floatN b) {

            if (!U.IsSquare)
                throw new System.ArgumentException("LUSolve: U must be square");

            if (b.N != U.M_Rows)
                throw new System.ArgumentException("LUSolve: b.N must equal U.M_Rows");

            if (pivot.N != b.N)
                throw new System.ArgumentException("LUSolve: pivot.N must equal b.N");

            // apply pivot to b
            pivot.ApplyInverseVec(ref b);

            // Solver linear system LUx = b, b is overwritten with x

            // Solve Ly = b
            Solvers.SolveLowerTriangular(ref L, ref b);
            // Solve Ux = y
            Solvers.SolveUpperTriangular(ref U, ref b);

        }

        /// <summary>
        /// Compute the determinant from the compact inplace LU form with pivot.
        /// Returns P.Sign * product of diagonal elements LU[P[i], i].
        /// Throws ArgumentException if LU is not square or P.N != LU.M_Rows.
        /// </summary>
        public static float determinant(in floatMxN LU, in Pivot P) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("determinant: LU must be square");

            if (P.N != LU.M_Rows)
                throw new System.ArgumentException("determinant: P.N must equal LU.M_Rows");

            int m = LU.M_Rows;
            float det = P.Sign;

            for (int i = 0; i < m; i++)
                det *= LU[P[i], i];

            return det;
        }
    }
}
