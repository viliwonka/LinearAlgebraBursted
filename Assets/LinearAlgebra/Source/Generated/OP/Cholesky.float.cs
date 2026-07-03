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
    public static partial class Cholesky {

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
        /// is left partially overwritten, so treat A as destroyed on failure.
        /// </summary>
        public static DirectSolveInfo choleskyDecomposition(in floatMxN A, ref floatMxN L) {
            if (!A.IsSquare)
                throw new ArgumentException("choleskyDecomposition: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("choleskyDecomposition: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("choleskyDecomposition: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (n == 0) return new DirectSolveInfo { status = DirectSolveStatus.Success };

            // RIGHT-LOOKING (outer-product) Cholesky. The left-looking form's hot loop is a dot
            // (reduction over already-computed columns), which stays effectively scalar under strict
            // FloatMode (loop-carried accumulator). This form instead, once column j is known,
            // immediately subtracts its rank-1 contribution from the trailing LOWER triangle as a set
            // of row-wise axpys: L[i, j+1..i] -= L[i,j] * L[j+1..i, j]. Each row segment is unit-stride
            // (row-major), so they go through the vectorising Unsafe_OP.axpy ([NoAlias], the GEMM
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
            // forward substitution against L11 (Unsafe_OP.trsmLowerPanel — DTRSM), and finally the whole
            // trailing lower triangle is updated ONCE per panel with a triangular SYRK
            // (Unsafe_OP.syrkLowerSub, A22 -= L21*L21ᵀ) instead of one rank-1 pass per column — trading
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
                            Unsafe_OP.axpy(lp + (long)i * n + (j + 1), ljp + (j + 1), -ljp[i], i - j);
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
                            Unsafe_OP.axpy(lp + (long)i * n + (j + 1), ljp + (j + 1), -ljp[i], i - j);
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
                        //     (Unsafe_OP.trsmLowerPanel) instead of a rank-1 update per (row, column)
                        //     pair — the earlier per-column-per-row formulation was measured to keep
                        //     O(n^2) tiny NoInlining calls (same call count as the unblocked sweep,
                        //     just each doing less work), which ate the SYRK's savings at n up to ~512;
                        //     this collapses that to O(n/CHOL_BLOCK) calls, one per panel.
                        Unsafe_OP.trsmLowerPanel(lp + (long)j0 * n + j0, n, lp + (long)rStart * n + j0, n, ntrail, jb);

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

                        Unsafe_OP.syrkLowerSub(lp, n, rStart, j0, jb, ptp);
                    }
                }

                PT.Dispose();
                lj.Dispose();
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Solve A x = b for x given the Cholesky factor L (A = L * Lᵀ) from choleskyDecomposition.
        /// b is overwritten with x. Use this overload to solve for multiple right-hand sides without
        /// refactoring. Solves L y = b (forward substitution), then Lᵀ x = y (back substitution).
        /// Always reports DirectSolveStatus.Success — this assumes a valid factor from a
        /// choleskyDecomposition that returned Success; it does not re-verify it.
        ///
        /// PRECONDITION: L must be a valid factor from a choleskyDecomposition that returned Success.
        /// Passing an invalid or partially-computed L (e.g. from a decomposition that returned
        /// NotPositiveDefinite) divides by a zero/garbage diagonal and silently produces NaN/Inf —
        /// always check Solved.
        /// </summary>
        public static DirectSolveInfo choleskySolve(ref floatMxN L, ref floatN b) {
            if (!L.IsSquare)
                throw new ArgumentException("choleskySolve: L must be square");

            if (b.N != L.M_Rows)
                throw new ArgumentException("choleskySolve: b.N must equal L.M_Rows");

            // L y = b
            Solvers.solveLowerTriangular(ref L, ref b);
            // Lᵀ x = y
            SolveUpperTriangularTransposed(ref L, ref b);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Factor SPD A = L * Lᵀ into caller-allocated L and solve A x = b in one call.
        /// b is overwritten with x. Returns NotPositiveDefinite (forwarded from the decomposition)
        /// without solving if A is not positive-definite.
        /// </summary>
        public static DirectSolveInfo choleskySolve(in floatMxN A, ref floatMxN L, ref floatN b) {
            var decompInfo = choleskyDecomposition(in A, ref L);
            if (!decompInfo.Solved)
                return decompInfo;

            return choleskySolve(ref L, ref b);
        }

        /// <summary>
        /// Pivoted (rank-revealing) Cholesky for a symmetric positive-SEMI-definite matrix:
        /// Pᵀ·A·P = L·Lᵀ, where the symmetric permutation P (a column/row Pivot) is chosen greedily,
        /// largest remaining diagonal first (LAPACK xPSTRF). Unlike the plain choleskyDecomposition —
        /// which hard-fails the instant it meets a non-positive pivot — this degrades gracefully on a
        /// rank-deficient PSD matrix: it stops when the largest remaining diagonal drops below the
        /// numerical-zero tolerance and reports that step count as the numerical rank. The returned L
        /// is lower-triangular n×n whose columns rank..n-1 are exactly zero, so the meaningful factor
        /// is the n×rank lower-trapezoidal block.
        ///
        /// A is read-only (only its lower triangle is referenced; the matrix is assumed symmetric).
        /// L (caller-allocated, square, same dimension as A) receives the factor. P (caller-allocated,
        /// size n) is Reset() internally and ends as the symmetric permutation.
        ///
        /// Returns a <see cref="RankRevealingInfo"/>: status Success (PSD, full rank n) or
        /// RankDeficient (PSD, rank &lt; n — still a usable factor, see rank) or Indefinite if A is
        /// not even positive-semi-definite — a Schur-complement diagonal goes significantly negative
        /// (below -n·eps·max|diag|). On Indefinite L/P/rank are undefined. Tolerance is
        /// scale-relative to the largest |diagonal| so it is scale-invariant.
        /// </summary>
        public static RankRevealingInfo choleskyDecompositionPivot(in floatMxN A, ref floatMxN L, ref Pivot P,
                                                       ref floatCholeskyPivot_WS ws) {
            if (!A.IsSquare)
                throw new ArgumentException("choleskyDecompositionPivot: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("choleskyDecompositionPivot: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("choleskyDecompositionPivot: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (P.N != n)
                throw new ArgumentException("choleskyDecompositionPivot: P.N must equal A dimension");
            RequireCholeskyPivotWorkspace(in ws, n, true, false);

            P.Reset();
            int rank = n;

            if (n == 0)
                return new RankRevealingInfo { status = DirectSolveStatus.Success, rank = rank };

            // zero L (only the lower-triangle columns 0..rank-1 get written below).
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    L[i, j] = 0;

            // Working symmetric matrix W (caller workspace) holds the UPPER triangle only: W[i,j], j>=i.
            // Factor as A = U^T U with U upper-triangular: the right-looking sweep broadcasts the freshly
            // computed factor ROW U[k, k..] (contiguous in row-major) and subtracts its rank-1 contribution
            // from each trailing row W[i, i..] (also contiguous) — a unit-stride axpy with NO symmetric
            // mirror (the lower triangle is never stored, so there is nothing to strided-copy; that mirror
            // was the cache cliff). The returned factor L is LOWER-triangular (P^T A P = L L^T, L = U^T);
            // each finished U row is scattered into L's column k (an O(n) strided write, « the O(n^3)
            // factor). Only A's lower triangle is read (A is assumed symmetric): W[i,j] := A[j,i], j>=i.
            var W = ws.W;
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                    W[i, j] = A[j, i];

            // scale-relative numerical-zero / indefinite tolerance. Scanned over all (upper-triangle)
            // entries, not just the diagonal: for a genuine PSD matrix the largest magnitude IS on the
            // diagonal, so this is identical there — but it keeps the tolerance meaningful for a non-PSD
            // input whose mass is off-diagonal (e.g. zero diagonal, nonzero off-diagonal), which must not
            // be silently accepted as a rank-0 PSD matrix.
            float absScale = 0;
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) {
                    float ad = math.abs(W[i, j]);
                    if (ad > absScale) absScale = ad;
                }
            float stopTol = (float)n * Consts.floatEpsilon * absScale;

            // Freshly-computed factor row U[k, k..n-1] gathered contiguously into urow so the rank-1 Schur
            // update is a set of unit-stride row-axpys (the vectorising Unsafe_OP.axpy path). One O(n) Temp
            // buffer (« the O(n^3) factor), matching the plain choleskyDecomposition's `lj`.
            var urow = new floatN(n, Allocator.Temp, false);
            unsafe {
                float* wp = W.Data.Ptr;
                float* urowp = urow.Data.Ptr;

                for (int k = 0; k < n; k++) {

                    // pick the largest remaining diagonal (the pivot); also track the smallest to detect
                    // indefiniteness (a PSD Schur complement keeps every diagonal >= 0).
                    int q = k;
                    float maxDiag = W[k, k];
                    float minDiag = W[k, k];
                    for (int j = k + 1; j < n; j++) {
                        float d = W[j, j];
                        if (d > maxDiag) { maxDiag = d; q = j; }
                        if (d < minDiag) minDiag = d;
                    }

                    // a clearly-negative remaining diagonal => not PSD.
                    if (minDiag < -stopTol) {
                        urow.Dispose();
                        rank = k;
                        return new RankRevealingInfo { status = DirectSolveStatus.Indefinite, rank = rank };
                    }

                    // largest remaining diagonal is numerically zero (NaN-safe). For a genuine PSD matrix
                    // a zero diagonal forces its whole row/column to zero, so the trailing block is now
                    // all-negligible and rank k is reached. But if any trailing entry is still
                    // significant the matrix is NOT PSD (e.g. [[0,1],[1,0]], eigenvalues +/-1) => reject
                    // as indefinite rather than silently returning a bogus low rank.
                    if (!(maxDiag > stopTol)) {
                        for (int i = k; i < n; i++)
                            for (int j = i; j < n; j++)
                                if (math.abs(W[i, j]) > stopTol) {
                                    urow.Dispose();
                                    rank = k;
                                    return new RankRevealingInfo { status = DirectSolveStatus.Indefinite, rank = rank };
                                }
                        rank = k;
                        break;
                    }

                    // symmetric pivot k <-> q (k < q) on the upper-triangle storage: swap the two
                    // diagonals, the column segments above row k, the row segments right of column q, and
                    // the transposed "between" segment; the cross entry W[k,q] maps to itself. Reads only
                    // upper entries (i<=j), so no lower triangle is needed.
                    if (q != k) {
                        { float t = W[k, k]; W[k, k] = W[q, q]; W[q, q] = t; }
                        for (int i = 0; i < k; i++)     { float t = W[i, k]; W[i, k] = W[i, q]; W[i, q] = t; }
                        for (int j = q + 1; j < n; j++) { float t = W[k, j]; W[k, j] = W[q, j]; W[q, j] = t; }
                        for (int m = k + 1; m < q; m++) { float t = W[k, m]; W[k, m] = W[m, q]; W[m, q] = t; }
                        Swap_OP.Rows(ref L, k, q, 0, k);    // permute the already-computed factor rows
                        P.Swap(k, q);
                    }

                    // factor row k: U[k,k] = sqrt(W[k,k]); U[k,j>k] = W[k,j] / U[k,k]. Gather U[k, k+1..]
                    // contiguously into urow and scatter it into L's column k (L = U^T, strided O(n) write).
                    float Ukk = math.sqrt(W[k, k]);
                    L[k, k] = Ukk;
                    for (int j = k + 1; j < n; j++) {
                        float u = W[k, j] / Ukk;
                        urowp[j] = u;
                        L[j, k] = u;
                    }

                    // rank-1 Schur update of the trailing UPPER triangle, one unit-stride row-axpy per
                    // row: W[i, i..n-1] -= urow[i] * urow[i..n-1].
                    for (int i = k + 1; i < n; i++)
                        Unsafe_OP.axpy(wp + (long)i * n + i, urowp + i, -urowp[i], n - i);
                }
            }

            urow.Dispose();
            return new RankRevealingInfo
            {
                status = (rank < n) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                rank = rank
            };
        }

        /// <summary>
        /// choleskyDecompositionPivot allocating its n x n symmetric working copy from Allocator.Temp.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static RankRevealingInfo choleskyDecompositionPivot(in floatMxN A, ref floatMxN L, ref Pivot P) {
            int n = A.IsSquare ? A.M_Rows : 0;
            var ws = new floatCholeskyPivot_WS
            {
                W = new floatMxN(n, n, Allocator.Temp),
                bt = default
            };
            var info = choleskyDecompositionPivot(in A, ref L, ref P, ref ws);
            ws.W.Dispose();
            return info;
        }

        /// <summary>
        /// Solve A x = b using a pivoted-Cholesky factor (Pᵀ·A·P = L·Lᵀ) from
        /// choleskyDecompositionPivot. b is overwritten with x (length n). L is passed by ref to feed
        /// the triangular solvers (it is not modified). Internally allocates small Allocator.Temp
        /// workspaces.
        ///
        /// - Full rank (rank == n): the exact solution, via permuted forward/back substitution.
        /// - Rank-deficient (rank &lt; n): the minimum-norm least-squares solution x = A⁺b. With the
        ///   rank factorization A = M·Mᵀ (M = P·L₁, the n×rank factor), the symmetric pseudoinverse is
        ///   A⁺ = M (MᵀM)⁻² Mᵀ; this forms the rank×rank SPD Gram matrix G = L₁ᵀL₁, factors it, and
        ///   applies G⁻¹ twice. If b ∈ range(A) this reproduces the exact solution; otherwise it
        ///   returns the least-squares solution of smallest norm.
        /// Always reports DirectSolveStatus.Success — this assumes a valid factor and the given
        /// rank are already known-good (from a choleskyDecompositionPivot that succeeded); it does
        /// not re-verify them.
        /// </summary>
        public static DirectSolveInfo choleskyPivotSolve(ref floatMxN L, in Pivot P, int rank, ref floatN b,
                                              ref floatCholeskyPivot_WS ws) {
            if (!L.IsSquare)
                throw new ArgumentException("choleskyPivotSolve: L must be square");

            int n = L.M_Rows;

            if (b.N != n)
                throw new ArgumentException("choleskyPivotSolve: b.N must equal L.M_Rows");

            if (P.N != n)
                throw new ArgumentException("choleskyPivotSolve: P.N must equal L.M_Rows");

            if (rank < 0 || rank > n)
                throw new ArgumentException("choleskyPivotSolve: rank must be in [0, n]");
            RequireCholeskyPivotWorkspace(in ws, n, false, true);

            // x = A⁺b = 0 for the zero matrix.
            if (rank == 0) {
                for (int i = 0; i < n; i++)
                    b[i] = 0;
                return new DirectSolveInfo { status = DirectSolveStatus.Success };
            }

            // gather b̃[i] = b[P[i]] (apply the symmetric permutation to the RHS) into the workspace.
            var bt = ws.bt;
            for (int i = 0; i < n; i++)
                bt[i] = b[P[i]];

            if (rank == n) {
                // full rank: L z = b̃, then Lᵀ x̃ = z, then scatter x[P[i]] = x̃[i].
                Solvers.solveLowerTriangular(ref L, ref bt);
                SolveUpperTriangularTransposed(ref L, ref bt);
                for (int i = 0; i < n; i++)
                    b[P[i]] = bt[i];
                return new DirectSolveInfo { status = DirectSolveStatus.Success };
            }

            // rank-deficient minimum-norm solution.
            int r = rank;

            // g = L₁ᵀ b̃   (L₁[i,k] = 0 for i < k, so the sum starts at i = k).
            var g = new floatN(r, Allocator.Temp);
            for (int k = 0; k < r; k++) {
                float s = 0;
                for (int i = k; i < n; i++)
                    s += L[i, k] * bt[i];
                g[k] = s;
            }

            // G = L₁ᵀ L₁  (r×r SPD, since L₁ has full column rank r).
            var G = new floatMxN(r, r, Allocator.Temp);
            for (int a = 0; a < r; a++)
                for (int c = 0; c <= a; c++) {
                    float s = 0;
                    for (int i = a; i < n; i++)   // i >= a >= c, and L[i,c] valid there
                        s += L[i, a] * L[i, c];
                    G[a, c] = s;
                    G[c, a] = s;
                }

            // z = G⁻² g : factor G once, apply G⁻¹ twice.
            var GL = new floatMxN(r, r, Allocator.Temp);
            if (!choleskyDecomposition(in G, ref GL)) {
                // G = L₁ᵀL₁ is SPD in exact arithmetic, but a borderline-revealed rank (a pivot just
                // above stopTol) can leave it numerically semidefinite, so the inner factorization
                // fails and the two solves below would divide by a ~0 pivot and emit NaN. Restore SPD
                // with a tiny Tikhonov ridge (negligible vs G's scale) and retry — Burst-friendly,
                // non-throwing, and far more useful than returning NaN.
                float gScale = 0;
                for (int a = 0; a < r; a++) {
                    float gd = math.abs(G[a, a]);
                    if (gd > gScale) gScale = gd;
                }
                float ridge = (float)r * Consts.floatEpsilon * gScale;
                for (int a = 0; a < r; a++)
                    G[a, a] += ridge;
                choleskyDecomposition(in G, ref GL);
            }
            choleskySolve(ref GL, ref g);   // g := G⁻¹ g
            choleskySolve(ref GL, ref g);   // g := G⁻² g_orig = z

            // x = M z : t = L₁ z (n-vector), then scatter x[P[i]] = t[i].
            for (int i = 0; i < n; i++) {
                float s = 0;
                int kmax = math.min(i + 1, r);   // L[i,k] = 0 for k > i
                for (int k = 0; k < kmax; k++)
                    s += L[i, k] * g[k];
                b[P[i]] = s;
            }

            GL.Dispose();
            G.Dispose();
            g.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// choleskyPivotSolve allocating its permuted-RHS scratch (length n) from Allocator.Temp.
        /// See the ref-workspace overload for semantics (the rank-deficient Gram buffers are per-call
        /// Temp in both forms). Always reports DirectSolveStatus.Success — see that overload.
        /// </summary>
        public static DirectSolveInfo choleskyPivotSolve(ref floatMxN L, in Pivot P, int rank, ref floatN b) {
            int n = L.IsSquare ? L.M_Rows : 0;
            var ws = new floatCholeskyPivot_WS
            {
                W = default,
                bt = new floatN(n, Allocator.Temp)
            };
            var info = choleskyPivotSolve(ref L, in P, rank, ref b, ref ws);
            ws.bt.Dispose();
            return info;
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve in one call: factors A (rank-revealing) and solves
        /// A x = b, b overwritten with x. Returns Indefinite (forwarded from the decomposition)
        /// WITHOUT solving if A is indefinite. For a rank-deficient (PSD) A this returns
        /// RankDeficient (with the detected rank) and the minimum-norm least-squares solution.
        /// </summary>
        public static RankRevealingInfo choleskyPivotSolve(in floatMxN A, ref floatMxN L, ref Pivot P, ref floatN b) {
            var decompInfo = choleskyDecompositionPivot(in A, ref L, ref P);
            if (!decompInfo.Solved)
                return decompInfo;

            choleskyPivotSolve(ref L, in P, decompInfo.rank, ref b);
            return decompInfo;
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve using a caller workspace (W for the factorization, bt for
        /// the solve). Returns Indefinite (forwarded from the decomposition) WITHOUT solving if A is
        /// indefinite. For a rank-deficient (PSD) A this returns RankDeficient (with the detected
        /// rank) and the minimum-norm least-squares solution.
        /// </summary>
        public static RankRevealingInfo choleskyPivotSolve(in floatMxN A, ref floatMxN L, ref Pivot P, ref floatN b,
                                              ref floatCholeskyPivot_WS ws) {
            var decompInfo = choleskyDecompositionPivot(in A, ref L, ref P, ref ws);
            if (!decompInfo.Solved)
                return decompInfo;

            choleskyPivotSolve(ref L, in P, decompInfo.rank, ref b, ref ws);
            return decompInfo;
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
