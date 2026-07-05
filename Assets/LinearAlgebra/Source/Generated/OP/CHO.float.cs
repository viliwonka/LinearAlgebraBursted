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
        /// L may alias A (in-place factorization): each A entry is read before it is overwritten, so
        /// passing the same matrix as both A and L is safe — but then A's strict upper triangle is
        /// destroyed (zeroed). On a NotPositiveDefinite return with L aliasing A, the lower triangle
        /// is left partially overwritten, so treat A as destroyed on failure. See decompInPlace
        /// (commit 2) for the documented in-place path.
        /// </summary>
        public static DirectSolveInfo decomp(in floatMxN A, ref floatMxN L) {
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
            // see docs/level3-blocking-guide.md recipe B), mirroring LAPACK's DPOTRF: a CHOL_BLOCK-wide
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
                float* lp = L.Data.Ptr;

                // Working matrix L := lower triangle of A, strict upper := 0. (L may alias A: the lower
                // copy is then a self-copy and zeroing the strict upper destroys A's upper half, which
                // matches the documented in-place behaviour. Only A's lower triangle is ever read.)
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++) L[i, j] = A[i, j];
                    for (int j = i + 1; j < n; j++) L[i, j] = (float)0;
                }

                var lj = new floatN(n, Allocator.Temp, false);
                float* ljp = lj.Data.Ptr;

                // Panel width for the blocked (level-3) path. Method-local const — Cholesky is a
                // partial class shared by the float/double generated files, so a class-level const of
                // the same name would collide across them (CS0102; see QR_BLOCK).
                const int CHOL_BLOCK = 32;

                // Size gate: MEASURED crossover, not the naive 2*CHOL_BLOCK (see
                // docs/level3-blocking-guide.md landmine "size gate" — LQ needed the same kind of
                // margin, LQ_BLOCK=64 but gate m>=512, i.e. 8x the block width). Benchmarked: n=128
                // (4 panels) is measurably SLOWER than the plain sweep — the panel/TRSM/SYRK
                // bookkeeping isn't amortised yet — while n=256 (8 panels) is the first size that wins
                // for both float and double. A shared float/double threshold uses the slower type's
                // crossover.
                const int CHOL_BLOCK_MIN_N = 8 * CHOL_BLOCK;

                if (n < CHOL_BLOCK_MIN_N)
                {
                    // Small matrix: plain per-column right-looking sweep, unchanged.
                    for (int j = 0; j < n; j++)
                    {
                        // L[j,j] already holds A[j,j] - sum_{k<j} L[j,k]^2 (applied by earlier rank-1
                        // updates). Not positive-definite -> reject; !(d > 0) also catches NaN, before
                        // sqrt.
                        float d = L[j, j];
                        if (!(d > (float)0)) { lj.Dispose(); return new DirectSolveInfo { status = DirectSolveStatus.NotPositiveDefinite }; }

                        float Ljj = math.sqrt(d);
                        L[j, j] = Ljj;
                        float inv = (float)1 / Ljj;

                        // Scale column j below the diagonal and gather it contiguously into lj.
                        for (int i = j + 1; i < n; i++)
                        {
                            float v = L[i, j] * inv;
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
                var PT = new floatN(CHOL_BLOCK * n, Allocator.Temp, false);
                float* ptp = PT.Data.Ptr;

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
                        float d = L[j, j];
                        if (!(d > (float)0)) { PT.Dispose(); lj.Dispose(); return new DirectSolveInfo { status = DirectSolveStatus.NotPositiveDefinite }; }

                        float Ljj = math.sqrt(d);
                        L[j, j] = Ljj;
                        float inv = (float)1 / Ljj;

                        for (int i = j + 1; i < panelEnd; i++)
                        {
                            float v = L[i, j] * inv;
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
        public static DirectSolveInfo decompSolve(ref floatMxN L, ref floatN b_to_x) {
            if (!L.IsSquare)
                throw new ArgumentException("decompSolve: L must be square");

            if (b_to_x.N != L.M_Rows)
                throw new ArgumentException("decompSolve: b_to_x.N must equal L.M_Rows");

            // L y = b
            Solvers.triLower(ref L, ref b_to_x);
            // Lᵀ x = y
            SolveUpperTriangularTransposed(ref L, ref b_to_x);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
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
