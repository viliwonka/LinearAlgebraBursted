#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Mathematics;
using LinearAlgebra.Internal;

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
        public static DirectSolveInfo triUpper(ref fProxyMxN U, ref fProxyN b_to_x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new ArgumentException("Blas.triUpper: Matrix must be square or tall (M_Rows >= N_Cols)");

            if(U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpper: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--)
            {
                fProxy sum = 0;

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
        public static DirectSolveInfo triLower(ref fProxyMxN L, ref fProxyN b_to_x)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLower: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLower: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++)
            {
                fProxy sum = 0;

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
        public static DirectSolveInfo triLowerLU(ref fProxyMxN L, in Pivot RP, ref fProxyN b_to_x) {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLU: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLowerLU: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                fProxy sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * b_to_x[c];

                b_to_x[r] = (b_to_x[r] - sum);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo triUpperLU(ref fProxyMxN U, in Pivot RP, ref fProxyN b_to_x) {
            if(U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLU: Matrix must be square");

            if (U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpperLU: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--) {
                fProxy sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[RP[r], c] * b_to_x[c];

                b_to_x[r] = (b_to_x[r] - sum) / U[RP[r], r];
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // ---- multi-RHS (TRSM) forms: solve for a whole matrix of right-hand sides at once ----
        //
        // Each right-hand side is a COLUMN of B_to_X (n x k, row-major). The substitution runs the
        // same recurrence as the vector forms, but every scalar update on component r becomes a
        // unit-stride axpy across the k columns — B_to_X row r is contiguous in memory, so the inner
        // loop over the k right-hand sides vectorises (UnsafeOP.axpy, the GEMM pointer path). This is
        // the level-2 (TRSV) -> level-3 (TRSM) jump: each factor entry U[r,c] is loaded once and
        // reused across all k right-hand sides, and the O(n^2) triangular solve streams the RHS block
        // instead of one vector. Result differs from looping the vector form column-by-column only by
        // summation-order rounding (the trailing contributions are subtracted incrementally rather
        // than accumulated into one sum first — a different, equally-valid order; see the blocked
        // LU/CHO cores for the same convention).

        // Solve U X = B for X (multi-RHS). U may be tall (only the top N_Cols x N_Cols block is read).
        // See the vector triUpper for the non-singular-diagonal precondition (unguarded).
        /// <param name="B_to_X">On entry B (N_Cols rows x k cols); on exit the solution X.</param>
        public static unsafe DirectSolveInfo triUpper(ref fProxyMxN U, ref fProxyMxN B_to_X)
        {
            if (U.M_Rows < U.N_Cols)
                throw new ArgumentException("Blas.triUpper: Matrix must be square or tall (M_Rows >= N_Cols)");

            if (U.N_Cols != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triUpper: U.N_Cols must equal B_to_X.M_Rows");

            int n = U.N_Cols;
            int k = B_to_X.N_Cols;
            fProxy* Xp = B_to_X.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                fProxy* Xr = Xp + (long)r * k;

                for (int c = r + 1; c < n; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -U[r, c], k);

                fProxy inv = (fProxy)1 / U[r, r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve L X = B for X (multi-RHS). See the vector triLower for the non-singular-diagonal
        // precondition (unguarded).
        /// <param name="B_to_X">On entry B (M_Rows rows x k cols); on exit the solution X.</param>
        public static unsafe DirectSolveInfo triLower(ref fProxyMxN L, ref fProxyMxN B_to_X)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLower: Matrix must be square");

            if (L.M_Rows != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triLower: L.M_Rows must equal B_to_X.M_Rows");

            int n = L.M_Rows;
            int k = B_to_X.N_Cols;
            fProxy* Xp = B_to_X.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                fProxy* Xr = Xp + (long)r * k;

                for (int c = 0; c < r; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -L[r, c], k);

                fProxy inv = (fProxy)1 / L[r, r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve L Y = B (unit-lower, row-pivoted), multi-RHS — the compact-LU forward step. B_to_X rows
        // are logical (X[r,:] is the r-th component of every RHS); only the factor L is pivot-indirected.
        /// <param name="B_to_X">On entry B; on exit Y (the forward-substitution result).</param>
        public static unsafe DirectSolveInfo triLowerLU(ref fProxyMxN L, in Pivot RP, ref fProxyMxN B_to_X)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLU: Matrix must be square");

            if (L.M_Rows != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triLowerLU: L.M_Rows must equal B_to_X.M_Rows");

            int n = L.M_Rows;
            int k = B_to_X.N_Cols;
            fProxy* Xp = B_to_X.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                fProxy* Xr = Xp + (long)r * k;
                for (int c = 0; c < r; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -L[RP[r], c], k);
                // unit diagonal: no scale
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve U X = Y (row-pivoted), multi-RHS — the compact-LU back step.
        /// <param name="B_to_X">On entry Y; on exit the solution X.</param>
        public static unsafe DirectSolveInfo triUpperLU(ref fProxyMxN U, in Pivot RP, ref fProxyMxN B_to_X)
        {
            if (U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLU: Matrix must be square");

            if (U.N_Cols != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triUpperLU: U.N_Cols must equal B_to_X.M_Rows");

            int n = U.N_Cols;
            int k = B_to_X.N_Cols;
            fProxy* Xp = B_to_X.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                fProxy* Xr = Xp + (long)r * k;
                for (int c = r + 1; c < n; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -U[RP[r], c], k);

                fProxy inv = (fProxy)1 / U[RP[r], r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }
    }
}
