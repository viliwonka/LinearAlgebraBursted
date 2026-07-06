#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Triangular-solve primitives (forward/back substitution) — the innermost kernels the direct
    // factorization solvers (LU/CHO/QR/LQ) call once they hold a triangular factor. They live on
    // Blas as the level-2 substitution counterpart to its GEMM/GEMV kernels; the higher-level
    // "solve Ax=b" entry points live on LU/CHO/QR/QRCP. Iterative (Krylov) solvers are on Krylov.
    public static partial class Blas {

        // Solve Ux = b for x
        // U may be tall (M_Rows >= N_Cols): only the top N_Cols x N_Cols block is read,
        // which is the R block produced by QR on overdetermined systems.
        // PRECONDITION: U is non-singular — every diagonal U[r,r] must be nonzero. A zero diagonal
        // (a singular/rank-deficient triangular factor) divides by zero and yields Inf/NaN; this
        // primitive does not guard it. For rank-deficient systems use the rank-revealing paths
        // (QRCP.decompInPlace, SVD.pinvSolve, or CHOP.solveInPlace).
        // Always reports DirectSolveStatus.Success — this primitive assumes a valid (non-singular)
        // triangular factor and does not itself detect a bad one.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo triUpper(ref floatMxN U, ref floatN b_to_x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new ArgumentException("Blas.triUpper: Matrix must be square or tall (M_Rows >= N_Cols)");

            if(U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpper: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--)
            {
                float sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[r, c] * b_to_x[c];

                b_to_x[r] = (b_to_x[r] - sum) / U[r, r];
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve Lx = b for x
        // PRECONDITION: L is non-singular — every diagonal L[r,r] must be nonzero (see
        // triUpper; a zero diagonal divides by zero -> Inf/NaN, unguarded).
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo triLower(ref floatMxN L, ref floatN b_to_x)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLower: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLower: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++)
            {
                float sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[r, c] * b_to_x[c];

                b_to_x[r] = (b_to_x[r] - sum) / L[r, r];
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve Ly = b for, where y = Ux
        // RP = Row Pivot
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo triLowerLU(ref floatMxN L, in Pivot RP, ref floatN b_to_x) {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLU: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLowerLU: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                float sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * b_to_x[c];

                b_to_x[r] = (b_to_x[r] - sum);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo triUpperLU(ref floatMxN U, in Pivot RP, ref floatN b_to_x) {
            if(U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLU: Matrix must be square");

            if (U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpperLU: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--) {
                float sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[RP[r], c] * b_to_x[c];

                b_to_x[r] = (b_to_x[r] - sum) / U[RP[r], r];
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }
    }
}
