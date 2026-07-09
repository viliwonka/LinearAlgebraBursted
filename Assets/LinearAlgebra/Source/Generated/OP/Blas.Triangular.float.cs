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
        public static unsafe DirectSolveInfo triUpper(ref floatMxN U, ref floatN b_to_x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new ArgumentException("Blas.triUpper: Matrix must be square or tall (M_Rows >= N_Cols)");

            if(U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpper: Matrix and vector must have same number of columns");

            // The trailing sum_{c>r} U[r,c]*x[c] is a dot of row r's tail (contiguous, row-major) with the
            // already-solved x tail -> route through the SIMD vecDotRange. U and b_to_x never alias.
            int n = U.N_Cols;
            int stride = U.N_Cols;
            float* Up = U.Data.Ptr;
            float* bp = b_to_x.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                float* Ur = Up + (long)r * stride;
                float sum = UnsafeOP.vecDotRange(Ur, bp, r + 1, n);
                bp[r] = (bp[r] - sum) / Ur[r];
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve Lx = b for x
        // PRECONDITION: L is non-singular — every diagonal L[r,r] must be nonzero (see
        // triUpper; a zero diagonal divides by zero -> Inf/NaN, unguarded).
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static unsafe DirectSolveInfo triLower(ref floatMxN L, ref floatN b_to_x)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLower: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLower: Matrix and vector must have same number of rows");

            // Leading sum_{c<r} L[r,c]*x[c] = dot of row r's head with the solved x head -> vecDotRange.
            int n = L.M_Rows;
            int stride = L.N_Cols;
            float* Lp = L.Data.Ptr;
            float* bp = b_to_x.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                float* Lr = Lp + (long)r * stride;
                float sum = UnsafeOP.vecDotRange(Lr, bp, 0, r);
                bp[r] = (bp[r] - sum) / Lr[r];
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve Ly = b for, where y = Ux
        // RP = Row Pivot
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static unsafe DirectSolveInfo triLowerLU(ref floatMxN L, in Pivot RP, ref floatN b_to_x) {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLU: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLowerLU: Matrix and vector must have same number of rows");

            // Pivot only selects WHICH row (columns stay contiguous), so row RP[r] head is still a
            // contiguous dot with the solved x head -> vecDotRange. Unit diagonal: no divide.
            int n = L.M_Rows;
            int stride = L.N_Cols;
            float* Lp = L.Data.Ptr;
            float* bp = b_to_x.Data.Ptr;

            for (int r = 0; r < n; r++) {
                float* Lr = Lp + (long)RP[r] * stride;
                float sum = UnsafeOP.vecDotRange(Lr, bp, 0, r);
                bp[r] = bp[r] - sum;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static unsafe DirectSolveInfo triUpperLU(ref floatMxN U, in Pivot RP, ref floatN b_to_x) {
            if(U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLU: Matrix must be square");

            if (U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpperLU: Matrix and vector must have same number of columns");

            // Row RP[r] tail dotted with the solved x tail (see triLowerLU on the pivot); divide by
            // the pivoted diagonal U[RP[r],r].
            int n = U.N_Cols;
            int stride = U.N_Cols;
            float* Up = U.Data.Ptr;
            float* bp = b_to_x.Data.Ptr;

            for (int r = n - 1; r >= 0; r--) {
                float* Ur = Up + (long)RP[r] * stride;
                float sum = UnsafeOP.vecDotRange(Ur, bp, r + 1, n);
                bp[r] = (bp[r] - sum) / Ur[r];
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
        public static unsafe DirectSolveInfo triUpper(ref floatMxN U, ref floatMxN B_to_X)
        {
            if (U.M_Rows < U.N_Cols)
                throw new ArgumentException("Blas.triUpper: Matrix must be square or tall (M_Rows >= N_Cols)");

            if (U.N_Cols != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triUpper: U.N_Cols must equal B_to_X.M_Rows");

            int n = U.N_Cols;
            int k = B_to_X.N_Cols;
            float* Xp = B_to_X.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                float* Xr = Xp + (long)r * k;

                for (int c = r + 1; c < n; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -U[r, c], k);

                float inv = (float)1 / U[r, r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve L X = B for X (multi-RHS). See the vector triLower for the non-singular-diagonal
        // precondition (unguarded).
        /// <param name="B_to_X">On entry B (M_Rows rows x k cols); on exit the solution X.</param>
        public static unsafe DirectSolveInfo triLower(ref floatMxN L, ref floatMxN B_to_X)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLower: Matrix must be square");

            if (L.M_Rows != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triLower: L.M_Rows must equal B_to_X.M_Rows");

            int n = L.M_Rows;
            int k = B_to_X.N_Cols;
            float* Xp = B_to_X.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                float* Xr = Xp + (long)r * k;

                for (int c = 0; c < r; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -L[r, c], k);

                float inv = (float)1 / L[r, r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve L Y = B (unit-lower, row-pivoted), multi-RHS — the compact-LU forward step. B_to_X rows
        // are logical (X[r,:] is the r-th component of every RHS); only the factor L is pivot-indirected.
        /// <param name="B_to_X">On entry B; on exit Y (the forward-substitution result).</param>
        public static unsafe DirectSolveInfo triLowerLU(ref floatMxN L, in Pivot RP, ref floatMxN B_to_X)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLU: Matrix must be square");

            if (L.M_Rows != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triLowerLU: L.M_Rows must equal B_to_X.M_Rows");

            int n = L.M_Rows;
            int k = B_to_X.N_Cols;
            float* Xp = B_to_X.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                float* Xr = Xp + (long)r * k;
                for (int c = 0; c < r; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -L[RP[r], c], k);
                // unit diagonal: no scale
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve U X = Y (row-pivoted), multi-RHS — the compact-LU back step.
        /// <param name="B_to_X">On entry Y; on exit the solution X.</param>
        public static unsafe DirectSolveInfo triUpperLU(ref floatMxN U, in Pivot RP, ref floatMxN B_to_X)
        {
            if (U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLU: Matrix must be square");

            if (U.N_Cols != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triUpperLU: U.N_Cols must equal B_to_X.M_Rows");

            int n = U.N_Cols;
            int k = B_to_X.N_Cols;
            float* Xp = B_to_X.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                float* Xr = Xp + (long)r * k;
                for (int c = r + 1; c < n; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -U[RP[r], c], k);

                float inv = (float)1 / U[RP[r], r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // ---- transposed compact-LU solves: Uᵀw=v then Lᵀw=(that) -- the getrs(trans='T') triangular
        // steps, called by LU.decompSolveTransA. RIGHT-LOOKING (axpy) formulation, not the row-dot
        // form triLowerLU/triUpperLU above use: a row of Uᵀ (or Lᵀ) is a COLUMN of the row-major
        // compact factor, which is strided and does not vectorise. Instead each step finalizes ONE
        // component then pushes its contribution into the remaining, not-yet-solved components via a
        // contiguous ROW of the factor -- an axpy, the same shape LU's own factorization elimination
        // uses. Operates in ROW order (the logical index, not the pivoted one): the caller applies the
        // pivot once, after both passes (see LU.decompSolveTransA) -- these primitives never touch it.

        // Solve Uᵀw = v in place (forward step). Diagonal is LU[RP[r], r] -- the same physical entry
        // triUpperLU reads as U[RP[r], r].
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry v; on exit w = U⁻ᵀv, still in ROW order.</param>
        public static unsafe DirectSolveInfo triUpperLUTransA(ref floatMxN U, in Pivot RP, ref floatN b_to_x)
        {
            if (U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLUTransA: Matrix must be square");

            if (U.N_Cols != b_to_x.N)
                throw new ArgumentException("Blas.triUpperLUTransA: Matrix and vector must have same number of columns");

            int n = U.N_Cols;
            int stride = U.N_Cols;
            float* Up = U.Data.Ptr;
            float* bp = b_to_x.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                float* Ur = Up + (long)RP[r] * stride;
                bp[r] = bp[r] / Ur[r];

                int len = n - (r + 1);
                if (len > 0)
                    UnsafeOP.axpy(bp + (r + 1), Ur + (r + 1), -bp[r], len);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve Lᵀw = z in place (backward step; unit diagonal, no divide). Same right-looking
        // rationale as triUpperLUTransA above.
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="b_to_x">On entry z (= U⁻ᵀv); on exit w = L⁻ᵀz, still in ROW order.</param>
        public static unsafe DirectSolveInfo triLowerLUTransA(ref floatMxN L, in Pivot RP, ref floatN b_to_x)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLUTransA: Matrix must be square");

            if (L.M_Rows != b_to_x.N)
                throw new ArgumentException("Blas.triLowerLUTransA: Matrix and vector must have same number of rows");

            int n = L.M_Rows;
            int stride = L.N_Cols;
            float* Lp = L.Data.Ptr;
            float* bp = b_to_x.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                // unit diagonal: no scale
                if (r > 0)
                {
                    float* Lr = Lp + (long)RP[r] * stride;
                    UnsafeOP.axpy(bp, Lr, -bp[r], r);
                }
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // ---- transposed compact-LU solves, multi-RHS (TRSM) forms: same right-looking rationale as
        // the vector forms above, swept across a whole k-wide row-block per step (one axpy per
        // contributing row instead of one per contributing scalar) -- the same level-3 shape the
        // forward triLowerLU/triUpperLU multi-RHS overloads use.

        // Solve Uᵀ W = V in place (forward step, multi-RHS).
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="B_to_X">On entry V (N_Cols rows x k cols); on exit W = U⁻ᵀV, still in ROW order.</param>
        public static unsafe DirectSolveInfo triUpperLUTransA(ref floatMxN U, in Pivot RP, ref floatMxN B_to_X)
        {
            if (U.IsSquare == false)
                throw new ArgumentException("Blas.triUpperLUTransA: Matrix must be square");

            if (U.N_Cols != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triUpperLUTransA: U.N_Cols must equal B_to_X.M_Rows");

            int n = U.N_Cols;
            int k = B_to_X.N_Cols;
            float* Xp = B_to_X.Data.Ptr;

            for (int r = 0; r < n; r++)
            {
                float* Xr = Xp + (long)r * k;
                float inv = (float)1 / U[RP[r], r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;

                for (int c = r + 1; c < n; c++)
                    UnsafeOP.axpy(Xp + (long)c * k, Xr, -U[RP[r], c], k);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Solve Lᵀ W = Z in place (backward step, multi-RHS; unit diagonal, no scale).
        // Always reports DirectSolveStatus.Success — see triUpper.
        /// <param name="B_to_X">On entry Z; on exit W = L⁻ᵀZ, still in ROW order.</param>
        public static unsafe DirectSolveInfo triLowerLUTransA(ref floatMxN L, in Pivot RP, ref floatMxN B_to_X)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Blas.triLowerLUTransA: Matrix must be square");

            if (L.M_Rows != B_to_X.M_Rows)
                throw new ArgumentException("Blas.triLowerLUTransA: L.M_Rows must equal B_to_X.M_Rows");

            int n = L.M_Rows;
            int k = B_to_X.N_Cols;
            float* Xp = B_to_X.Data.Ptr;

            for (int r = n - 1; r >= 0; r--)
            {
                float* Xr = Xp + (long)r * k;
                // unit diagonal: no scale
                for (int c = 0; c < r; c++)
                    UnsafeOP.axpy(Xp + (long)c * k, Xr, -L[RP[r], c], k);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }
    }
}
