#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// Cholesky factorization A = L * Lᵀ for symmetric positive-definite (SPD) matrices.
    /// Cheaper and more stable than LU for SPD systems (no pivoting needed).
    /// </summary>
    public static partial class CHO {

        /// <summary>
        /// Cholesky factorization A = L * Lᵀ for a symmetric positive-definite matrix A.
        /// L (caller-allocated, square, same dimension as A) is overwritten with the lower-triangular
        /// factor; its strict upper triangle is set to zero. A is read-only — only its lower triangle
        /// is referenced (the matrix is assumed symmetric, so the upper triangle is ignored).
        ///
        /// Returns Success; NotPositiveDefinite if A is not positive-definite (a non-positive pivot
        /// is encountered, which also catches NaN). On NotPositiveDefinite: no NaN/Inf is written,
        /// since the check happens before the sqrt.
        ///
        /// For the in-place variant (factor into A's own storage), use decompInPlace instead.
        /// </summary>
        public static DirectSolveInfo decomp(in doubleMxN A, ref doubleMxN L) {
            if (!A.IsSquare)
                throw new ArgumentException("decomp: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("decomp: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("decomp: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (n == 0) return new DirectSolveInfo { status = DirectSolveStatus.Success };

            // RIGHT-LOOKING (outer-product) Cholesky. The left-looking form's hot loop is a dot
            // (reduction over already-computed columns), which stays effectively scalar under strict
            // FloatMode (loop-carried accumulator). This form instead, once column j is known,
            // immediately subtracts its rank-1 contribution from the trailing LOWER triangle as a set
            // of row-wise axpys: L[i, j+1..i] -= L[i,j] * L[j+1..i, j]. Each row segment is unit-stride
            // (row-major), so they go through the vectorising UnsafeOP.axpy ([NoAlias], the GEMM
            // pointer path) and run at LU speed (float ~2x double). Only the lower triangle is touched
            // (i >= column index), so no work is wasted on the symmetric upper half.
            //
            // The active column j is gathered into a contiguous buffer `lj` (one strided pass) so both
            // axpy operands are unit-stride. Results differ from the old left-looking form by rounding
            // only (a different, equally-valid summation order); A = L*Lᵀ to working precision.
            //
            // Above a size threshold this is further raised to LEVEL-3 (blocked, right-looking POTRF;
            // see docs/dev/level3-blocking-guide.md recipe B), mirroring LAPACK's DPOTRF: a CHOL_BLOCK-wide
            // diagonal block L11 is factored with the same rank-1 sweep above (narrowed to the panel's
            // own jb rows/cols — DPOTF2), the below-panel strip L21 is then solved for in one shot by
            // forward substitution against L11 (UnsafeOP.trsmLowerPanel — DTRSM), and finally the whole
            // trailing lower triangle is updated ONCE per panel with a triangular SYRK
            // (UnsafeOP.syrkLowerSub, A22 -= L21*L21ᵀ) instead of one rank-1 pass per column — trading
            // O(n) re-streams of the trailing matrix for O(n/CHOL_BLOCK), the memory-bandwidth-bound
            // part of the algorithm. (An earlier version computed L21 by extending the panel's rank-1
            // sweep to the below-panel rows with a narrowed column width; it was measurably WORSE at
            // mid-range n — same O(n^2) small-call count as the unblocked sweep, just doing less work
            // per call — which is why L21 is a dedicated forward-substitution pass instead.) Below the
            // threshold the panel/TRSM/SYRK bookkeeping isn't worth it, so the plain per-column sweep is
            // used unchanged.
            unsafe
            {
                double* lp = L.Data.Ptr;

                // Working matrix L := lower triangle of A, strict upper := 0. (L may alias A: the lower
                // copy is then a self-copy and zeroing the strict upper destroys A's upper half, which
                // matches the documented in-place behaviour. Only A's lower triangle is ever read.)
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++) L[i, j] = A[i, j];
                    for (int j = i + 1; j < n; j++) L[i, j] = (double)0;
                }

                var lj = new doubleN(n, Allocator.Temp, false);
                double* ljp = lj.Data.Ptr;

                // Panel width for the blocked (level-3) path. Method-local const — CHO is a
                // partial class shared by the float/double generated files, so a class-level const of
                // the same name would collide across them (CS0102; see QR_BLOCK).
                const int CHOL_BLOCK = 32;

                // Size gate: MEASURED crossover, not the naive 2*CHOL_BLOCK (see
                // docs/dev/level3-blocking-guide.md landmine "size gate" — LQ needed the same kind of
                // margin, LQ_BLOCK=64 but gate m>=512, i.e. 8x the block width). Benchmarked: n=128
                // (4 panels) is measurably SLOWER than the plain sweep — the panel/TRSM/SYRK
                // bookkeeping isn't amortised yet — while n=256 (8 panels) is the first size that wins
                // for both float and double. A shared float/double threshold uses the slower type's
                // crossover.
                const int CHOL_BLOCK_MIN_N = Consts.doubleCholBlockMinN;   // float/double split (see Consts); default 8*CHOL_BLOCK

                if (n < CHOL_BLOCK_MIN_N)
                {
                    // Small matrix: plain per-column right-looking sweep, unchanged.
                    for (int j = 0; j < n; j++)
                    {
                        // L[j,j] already holds A[j,j] - sum_{k<j} L[j,k]^2 (applied by earlier rank-1
                        // updates). Not positive-definite -> reject; !(d > 0) also catches NaN, before
                        // sqrt.
                        double d = L[j, j];
                        if (!(d > (double)0)) { lj.Dispose(); return new DirectSolveInfo { status = DirectSolveStatus.NotPositiveDefinite }; }

                        double Ljj = math.sqrt(d);
                        L[j, j] = Ljj;
                        double inv = (double)1 / Ljj;

                        // Scale column j below the diagonal and gather it contiguously into lj.
                        for (int i = j + 1; i < n; i++)
                        {
                            double v = L[i, j] * inv;
                            L[i, j] = v;
                            ljp[i] = v;
                        }

                        // Rank-1 update of the trailing lower triangle, one row-axpy per row.
                        for (int i = j + 1; i < n; i++)
                            UnsafeOP.axpy(lp + (long)i * n + (j + 1), ljp + (j + 1), -ljp[i], i - j);
                    }

                    lj.Dispose();
                    return new DirectSolveInfo { status = DirectSolveStatus.Success };
                }

                // ---- blocked (level-3) path ----
                // PT holds the transpose of the current panel's below-diagonal strip L21 (jb x ntrail,
                // contiguous), sized for the worst case (first panel, j0=0: jb=CHOL_BLOCK, ntrail<=n).
                var PT = new doubleN(CHOL_BLOCK * n, Allocator.Temp, false);
                double* ptp = PT.Data.Ptr;

                for (int j0 = 0; j0 < n; j0 += CHOL_BLOCK)
                {
                    int jb = math.min(CHOL_BLOCK, n - j0);
                    int panelEnd = j0 + jb;

                    // (1) factor the jb x jb diagonal block L11 (DPOTF2-style): EXACTLY the small-n
                    //     loop above, just with its row/col bound narrowed from n to panelEnd — this
                    //     factors ONLY the panel's own rows, so it costs O(jb^3), not O(jb*n^2). The
                    //     below-panel strip is handled separately by the TRSM step (2): it is NOT
                    //     touched here (unlike a naive column-by-column extension would do), which is
                    //     what keeps this cheap regardless of panel position.
                    for (int j = j0; j < panelEnd; j++)
                    {
                        double d = L[j, j];
                        if (!(d > (double)0)) { PT.Dispose(); lj.Dispose(); return new DirectSolveInfo { status = DirectSolveStatus.NotPositiveDefinite }; }

                        double Ljj = math.sqrt(d);
                        L[j, j] = Ljj;
                        double inv = (double)1 / Ljj;

                        for (int i = j + 1; i < panelEnd; i++)
                        {
                            double v = L[i, j] * inv;
                            L[i, j] = v;
                            ljp[i] = v;
                        }

                        for (int i = j + 1; i < panelEnd; i++)
                            UnsafeOP.axpy(lp + (long)i * n + (j + 1), ljp + (j + 1), -ljp[i], i - j);
                    }

                    int rStart = panelEnd;
                    if (rStart < n)
                    {
                        int ntrail = n - rStart;

                        // (2) TRSM: solve L11 * L21[i,:]ᵀ = A21[i,:]ᵀ for every below-panel row i
                        //     (forward substitution against the just-factored L11), writing L21 in
                        //     place into L[rStart:n, j0:j0+jb). A21[i,:] is exactly the CURRENT
                        //     L[i, j0:j0+jb) — untouched since the last panel's SYRK, because step (1)
                        //     above never wrote to rows >= panelEnd. ONE call for the whole panel
                        //     (UnsafeOP.trsmLowerPanel) instead of a rank-1 update per (row, column)
                        //     pair — the earlier per-column-per-row formulation was measured to keep
                        //     O(n^2) tiny NoInlining calls (same call count as the unblocked sweep,
                        //     just each doing less work), which ate the SYRK's savings at n up to ~512;
                        //     this collapses that to O(n/CHOL_BLOCK) calls, one per panel.
                        UnsafeOP.trsmLowerPanel(lp + (long)j0 * n + j0, n, lp + (long)rStart * n + j0, n, ntrail, jb);

                        // (3) SYRK trailing update: A22 -= L21*L21ᵀ, L21 = L[j0+jb:n, j0:j0+jb]. Touches
                        //     ONLY the lower triangle of the trailing block (see syrkLowerSub) — a full
                        //     rectangular update would double the flops and write past L's diagonal.
                        // PT[p*ntrail + kp] = L21[kp, p] = L[rStart+kp, j0+p].
                        for (int kp = 0; kp < ntrail; kp++)
                        {
                            int row = rStart + kp;
                            for (int p = 0; p < jb; p++)
                                ptp[p * ntrail + kp] = L[row, j0 + p];
                        }

                        UnsafeOP.syrkLowerSub(lp, n, rStart, j0, jb, ptp);
                    }
                }

                PT.Dispose();
                lj.Dispose();
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Solve A x = b for x given the Cholesky factor L (A = L * Lᵀ) from decomp. b is overwritten
        /// with x. Use this overload to solve for multiple right-hand sides without refactoring.
        /// Solves L y = b (forward substitution), then Lᵀ x = y (back substitution).
        /// Always reports DirectSolveStatus.Success — this assumes a valid factor from a decomp that
        /// returned Success; it does not re-verify it.
        ///
        /// PRECONDITION: L must be a valid factor from a decomp that returned Success. Passing an
        /// invalid or partially-computed L (e.g. from a decomposition that returned
        /// NotPositiveDefinite) divides by a zero/garbage diagonal and silently produces NaN/Inf —
        /// always check Solved.
        /// </summary>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo decompSolve(ref doubleMxN L, ref doubleN b_to_x) {
            if (!L.IsSquare)
                throw new ArgumentException("decompSolve: L must be square");

            if (b_to_x.N != L.M_Rows)
                throw new ArgumentException("decompSolve: b_to_x.N must equal L.M_Rows");

            // L y = b
            Blas.triLower(ref L, ref b_to_x);
            // Lᵀ x = y
            SolveUpperTriangularTransposed(ref L, ref b_to_x);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Cholesky factorization in place: A_to_L is factored using its own storage (L aliases A
        /// internally -- each entry is read before it is overwritten, so this is safe). On return,
        /// A_to_L holds the lower-triangular factor L (strict upper triangle zeroed); valid input to
        /// decompSolve. On NotPositiveDefinite the lower triangle is left partially overwritten --
        /// treat A_to_L as destroyed on failure.
        /// </summary>
        /// <param name="A_to_L">On entry A; on exit the lower-triangular factor L.</param>
        public static DirectSolveInfo decompInPlace(ref doubleMxN A_to_L) => decomp(in A_to_L, ref A_to_L);

        /// <summary>
        /// Factor-and-solve A x = b in one call (POSV): factors A in place (L aliases A's own storage)
        /// then solves for x. b is overwritten with x. Returns NotPositiveDefinite (forwarded from the
        /// factorization) WITHOUT solving if A is not positive-definite.
        /// A_to_L holds the lower-triangular factor L on return; valid input to decompSolve for
        /// solving additional right-hand sides without refactoring.
        /// </summary>
        /// <param name="A_to_L">On entry A; on exit the lower-triangular factor L.</param>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo solveInPlace(ref doubleMxN A_to_L, ref doubleN b_to_x) {
            // Validate everything decompInPlace/decompSolve would check BEFORE either runs, so a
            // caller error (e.g. a mis-sized b_to_x) cannot destroy A_to_L first.
            if (!A_to_L.IsSquare)
                throw new ArgumentException("solveInPlace: A_to_L needs to be square");

            if (b_to_x.N != A_to_L.M_Rows)
                throw new ArgumentException("solveInPlace: b_to_x.N must equal A_to_L.M_Rows");

            var info = decompInPlace(ref A_to_L);
            if (!info.Solved)
                return info;

            decompSolve(ref A_to_L, ref b_to_x);
            return info;
        }

        // Solve Lᵀ x = b for x in place, where L is lower-triangular (so Lᵀ is upper-triangular).
        // Reads L transposed: (Lᵀ)[r,c] = L[c,r]. Avoids materializing the transpose.
        static void SolveUpperTriangularTransposed(ref doubleMxN L, ref doubleN x) {
            int n = L.M_Rows;

            for (int r = n - 1; r >= 0; r--) {
                double sum = 0;

                for (int c = r + 1; c < n; c++)
                    sum += L[c, r] * x[c];

                x[r] = (x[r] - sum) / L[r, r];
            }
        }

        // ---- multi-RHS (TRSM) forms: solve A X = B for a whole matrix of right-hand sides ----

        // Lᵀ X = B (multi-RHS), reading L transposed as above; every scalar update becomes a
        // unit-stride axpy across B's k columns (see Blas triangular TRSM note).
        static unsafe void SolveUpperTriangularTransposed(ref doubleMxN L, ref doubleMxN X) {
            int n = L.M_Rows;
            int k = X.N_Cols;
            double* Xp = X.Data.Ptr;

            for (int r = n - 1; r >= 0; r--) {
                double* Xr = Xp + (long)r * k;

                for (int c = r + 1; c < n; c++)
                    UnsafeOP.axpy(Xr, Xp + (long)c * k, -L[c, r], k);   // (Lᵀ)[r,c] = L[c,r]

                double inv = (double)1 / L[r, r];
                for (int j = 0; j < k; j++)
                    Xr[j] *= inv;
            }
        }

        /// <summary>
        /// Solve A X = B for X (multi-RHS) given the Cholesky factor L (A = L * Lᵀ) from decomp; each
        /// right-hand side is a COLUMN of B_to_X (n x k, overwritten with X). Solves L Y = B (forward),
        /// then Lᵀ X = Y (back). See the vector decompSolve for the valid-factor precondition.
        /// </summary>
        /// <param name="B_to_X">On entry B (n rows x k cols); on exit the solution X.</param>
        public static DirectSolveInfo decompSolve(ref doubleMxN L, ref doubleMxN B_to_X) {
            if (!L.IsSquare)
                throw new ArgumentException("decompSolve: L must be square");

            if (B_to_X.M_Rows != L.M_Rows)
                throw new ArgumentException("decompSolve: B_to_X.M_Rows must equal L.M_Rows");

            Blas.triLower(ref L, ref B_to_X);
            SolveUpperTriangularTransposed(ref L, ref B_to_X);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Factor-and-solve A X = B in one call (POSV, multi-RHS): factors A in place (L aliases A's own
        /// storage) then solves for every column of B_to_X. Returns NotPositiveDefinite (forwarded from
        /// the factorization) WITHOUT solving if A is not positive-definite. A_to_L holds L on return.
        /// </summary>
        /// <param name="A_to_L">On entry A; on exit the lower-triangular factor L.</param>
        /// <param name="B_to_X">On entry B (n rows x k cols); on exit the solution X.</param>
        public static DirectSolveInfo solveInPlace(ref doubleMxN A_to_L, ref doubleMxN B_to_X) {
            if (!A_to_L.IsSquare)
                throw new ArgumentException("solveInPlace: A_to_L needs to be square");

            if (B_to_X.M_Rows != A_to_L.M_Rows)
                throw new ArgumentException("solveInPlace: B_to_X.M_Rows must equal A_to_L.M_Rows");

            var info = decompInPlace(ref A_to_L);
            if (!info.Solved)
                return info;

            decompSolve(ref A_to_L, ref B_to_X);
            return info;
        }
    }
}
