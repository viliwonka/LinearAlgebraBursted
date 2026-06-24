#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Cholesky factorization A = L * Lᵀ for symmetric positive-definite (SPD) matrices.
    /// Cheaper and more stable than LU for SPD systems (no pivoting needed).
    /// Inpl = inplace
    /// </summary>
    public static partial class Cholesky {

        /// <summary>
        /// Cholesky factorization A = L * Lᵀ for a symmetric positive-definite matrix A.
        /// L (caller-allocated, square, same dimension as A) is overwritten with the lower-triangular
        /// factor; its strict upper triangle is set to zero. A is read-only — only its lower triangle
        /// is referenced (the matrix is assumed symmetric, so the upper triangle is ignored).
        ///
        /// Returns true on success; false if A is not positive-definite (a non-positive pivot is
        /// encountered, which also catches NaN). On false: no NaN/Inf is written, since the check
        /// happens before the sqrt.
        ///
        /// L may alias A (in-place factorization): each A entry is read before it is overwritten, so
        /// passing the same matrix as both A and L is safe — but then A's strict upper triangle is
        /// destroyed (zeroed). On a false (non-PD) return with L aliasing A, the lower triangle is
        /// left partially overwritten, so treat A as destroyed on failure.
        /// </summary>
        public static bool choleskyDecomposition(in floatMxN A, ref floatMxN L) {
            if (!A.IsSquare)
                throw new ArgumentException("choleskyDecomposition: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("choleskyDecomposition: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("choleskyDecomposition: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (n == 0) return true;

            for (int j = 0; j < n; j++) {

                // Diagonal: L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
                float diag = A[j, j];
                for (int k = 0; k < j; k++) {
                    float Ljk = L[j, k];
                    diag -= Ljk * Ljk;
                }

                // Not positive-definite. !(diag > 0) is also true for NaN, so this rejects
                // non-finite inputs before the sqrt can produce a NaN.
                if (!(diag > 0))
                    return false;

                float Ljj = math.sqrt(diag);
                L[j, j] = Ljj;

                // Below diagonal: L[i,j] = (A[i,j] - sum_{k<j} L[i,k] * L[j,k]) / L[j,j]
                for (int i = j + 1; i < n; i++) {

                    float sum = A[i, j];
                    for (int k = 0; k < j; k++)
                        sum -= L[i, k] * L[j, k];

                    L[i, j] = sum / Ljj;

                    // L is exactly lower-triangular
                    L[j, i] = 0;
                }
            }

            return true;
        }

        /// <summary>
        /// Solve A x = b for x given the Cholesky factor L (A = L * Lᵀ) from choleskyDecomposition.
        /// b is overwritten with x. Use this overload to solve for multiple right-hand sides without
        /// refactoring. Solves L y = b (forward substitution), then Lᵀ x = y (back substitution).
        ///
        /// PRECONDITION: L must be a valid factor from a choleskyDecomposition that returned true.
        /// Passing an invalid or partially-computed L (e.g. from a decomposition that returned false)
        /// divides by a zero/garbage diagonal and silently produces NaN/Inf — always check the bool.
        /// </summary>
        public static void choleskySolve(ref floatMxN L, ref floatN b) {
            if (!L.IsSquare)
                throw new ArgumentException("choleskySolve: L must be square");

            if (b.N != L.M_Rows)
                throw new ArgumentException("choleskySolve: b.N must equal L.M_Rows");

            // L y = b
            Solvers.SolveLowerTriangular(ref L, ref b);
            // Lᵀ x = y
            SolveUpperTriangularTransposed(ref L, ref b);
        }

        /// <summary>
        /// Factor SPD A = L * Lᵀ into caller-allocated L and solve A x = b in one call.
        /// b is overwritten with x. Returns false without solving if A is not positive-definite.
        /// </summary>
        public static bool choleskySolve(in floatMxN A, ref floatMxN L, ref floatN b) {
            if (!choleskyDecomposition(in A, ref L))
                return false;

            choleskySolve(ref L, ref b);
            return true;
        }

        // Solve Lᵀ x = b for x in place, where L is lower-triangular (so Lᵀ is upper-triangular).
        // Reads L transposed: (Lᵀ)[r,c] = L[c,r]. Avoids materializing the transpose.
        static void SolveUpperTriangularTransposed(ref floatMxN L, ref floatN x) {
            int n = L.M_Rows;

            for (int r = n - 1; r >= 0; r--) {
                float sum = 0;

                for (int c = r + 1; c < n; c++)
                    sum += L[c, r] * x[c];

                x[r] = (x[r] - sum) / L[r, r];
            }
        }
    }
}
