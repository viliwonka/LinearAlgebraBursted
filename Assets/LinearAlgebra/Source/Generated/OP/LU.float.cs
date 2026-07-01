#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

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
                Swap_OP.Rows(ref U, k, pivotIndex, k, m);

                // swap already calculated L rows
                Swap_OP.Rows(ref L, k, pivotIndex, 0, k);

                // Calculate L and U. The trailing-row elimination U[j, k+1:] -= Ljk * U[k, k+1:] is
                // an axpy over two DISTINCT rows (j > k) along the unit-stride column axis; routed
                // through the vectorising Unsafe_OP.axpy ([NoAlias], the GEMM pointer path) so Burst
                // SIMD-vectorises this O(n^3) hot loop (float ~2x double). Bitwise identical to the
                // scalar form: each column i is updated independently, and (-Ljk)*U[k,i] added to
                // U[j,i] equals U[j,i] - Ljk*U[k,i] exactly in IEEE.
                float Ukk = U[k, k];
                unsafe
                {
                    float* up = U.Data.Ptr;
                    float* rowK = up + (long)k * m;
                    int len = m - (k + 1);
                    for (int j = k + 1; j < m; j++) {

                        float Ljk = U[j, k] / Ukk;

                        L[j, k] = Ljk;

                        float* rowJ = up + (long)j * m;
                        Unsafe_OP.axpy(rowJ + (k + 1), rowK + (k + 1), -Ljk, len);

                        // U is exactly upper-triangular
                        U[j, k] = 0;
                    }
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
        public static bool luDecompositionInpl(ref floatMxN LU, ref Pivot P) {

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

                // Calculate L and U. Same vectorised axpy elimination as luDecomposition, but on the
                // physical (pivot-indirected) rows Pj, Pk — still distinct (Pj != Pk), so [NoAlias]
                // holds. Bitwise identical to the scalar form.
                float Ukk = LU[Pk, k];
                unsafe
                {
                    float* lup = LU.Data.Ptr;
                    float* rowPk = lup + (long)Pk * m;
                    int len = m - (k + 1);
                    for (int j = k + 1; j < m; j++) {

                        int Pj = P[j];

                        float Ljk = LU[Pj, k] / Ukk;

                        float* rowPj = lup + (long)Pj * m;
                        Unsafe_OP.axpy(rowPj + (k + 1), rowPk + (k + 1), -Ljk, len);

                        LU[Pj, k] = Ljk;
                    }
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
        public static void luSolve(ref floatMxN LU, in Pivot pivot, ref floatN b) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("luSolve: LU must be square");

            if (b.N != LU.M_Rows)
                throw new System.ArgumentException("luSolve: b.N must equal LU.M_Rows");

            if (pivot.N != b.N)
                throw new System.ArgumentException("luSolve: pivot.N must equal b.N");

            pivot.ApplyInverseVec(ref b);

            // Solve Ly = b
            Solvers.solveLowerTriangularLU(ref LU, in pivot, ref b);
            // Solve Ux = y
            Solvers.solveUpperTriangularLU(ref LU, in pivot, ref b);

        }

        /// <summary>
        /// Solve LUx = Pb for x using separate L and U matrices with pivot.
        /// b is overwritten with x.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        public static void luSolve(ref floatMxN L, ref floatMxN U, in Pivot pivot, ref floatN b) {

            if (!U.IsSquare)
                throw new System.ArgumentException("luSolve: U must be square");

            if (b.N != U.M_Rows)
                throw new System.ArgumentException("luSolve: b.N must equal U.M_Rows");

            if (pivot.N != b.N)
                throw new System.ArgumentException("luSolve: pivot.N must equal b.N");

            // apply pivot to b
            pivot.ApplyInverseVec(ref b);

            // Solver linear system LUx = b, b is overwritten with x

            // Solve Ly = b
            Solvers.solveLowerTriangular(ref L, ref b);
            // Solve Ux = y
            Solvers.solveUpperTriangular(ref U, ref b);

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
