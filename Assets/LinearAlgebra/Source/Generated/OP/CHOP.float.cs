#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// Pivoted (rank-revealing) Cholesky for symmetric positive-SEMI-definite matrices. Split out of
    /// <see cref="CHO"/> (which stays the plain, full-rank-only factorization) because the
    /// pivoted family carries its own Pivot/rank/tolerance contract.
    /// </summary>
    public static partial class CHOP {

        /// <summary>
        /// Pivoted (rank-revealing) Cholesky for a symmetric positive-SEMI-definite matrix:
        /// Pᵀ·A·P = L·Lᵀ, where the symmetric permutation P (a column/row Pivot) is chosen greedily,
        /// largest remaining diagonal first (LAPACK xPSTRF). Unlike the plain CHO.decomp — which
        /// hard-fails the instant it meets a non-positive pivot — this degrades gracefully on a
        /// rank-deficient PSD matrix: it stops when the largest remaining diagonal drops below the
        /// numerical-zero tolerance and reports that step count as the numerical rank. The returned L
        /// is lower-triangular n×n whose columns rank..n-1 are exactly zero, so the meaningful factor
        /// is the n×rank lower-trapezoidal block.
        ///
        /// A is read-only (only its lower triangle is referenced; the matrix is assumed symmetric).
        /// L (caller-allocated, square, same dimension as A) receives the factor. P (caller-allocated,
        /// size n) is Reset() internally and ends as the symmetric permutation.
        ///
        /// Returns a <see cref="RankInfo"/>: status Success (PSD, full rank n) or
        /// RankDeficient (PSD, rank &lt; n — still a usable factor, see rank) or Indefinite if A is
        /// not even positive-semi-definite — a Schur-complement diagonal goes significantly negative
        /// (below -n·eps·max|diag|). On Indefinite L/P/rank are undefined. Tolerance is
        /// scale-relative to the largest |diagonal| so it is scale-invariant.
        /// </summary>
        public static RankInfo decomp(in floatMxN A, ref floatMxN L, ref Pivot P,
                                                       ref floatCHOPCache ws) {
            if (!A.IsSquare)
                throw new ArgumentException("decomp: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("decomp: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("decomp: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (P.N != n)
                throw new ArgumentException("decomp: P.N must equal A dimension");
            RequireCholeskyPivotWorkspace(in ws, n, true, false);

            P.Reset();
            int rank = n;

            if (n == 0)
                return new RankInfo { status = DirectSolveStatus.Success, rank = rank };

            // NOTE on orientation: U/"upper triangle" below refers ONLY to this method's internal
            // scratch (W) and its row-wise sweep — a computational device chosen for a unit-stride
            // inner loop. The PUBLIC output L is genuinely lower-triangular, same contract as plain
            // CHO.decomp's L (both consumed identically by triLower +
            // SolveUpperTriangularTransposed) — see the scatter into L's COLUMN k a few lines below.
            //
            // Working symmetric matrix W (caller workspace) holds the UPPER triangle only: W[i,j], j>=i.
            // Factor as A = U^T U with U upper-triangular: the right-looking sweep broadcasts the freshly
            // computed factor ROW U[k, k..] (contiguous in row-major) and subtracts its rank-1 contribution
            // from each trailing row W[i, i..] (also contiguous) — a unit-stride axpy with NO symmetric
            // mirror (the lower triangle is never stored, so there is nothing to strided-copy). The
            // returned factor L is LOWER-triangular (P^T A P = L L^T, L = U^T);
            // each finished U row is scattered into L's column k (an O(n) strided write, « the O(n^3)
            // factor). Only A's lower triangle is read (A is assumed symmetric): W[i,j] := A[j,i], j>=i.
            //
            // Read A into W BEFORE touching L: this ordering is what makes decompInPlace-style self-
            // aliasing (L passed as the SAME buffer as A, see CHOP.solveInPlace's destructive overload)
            // safe -- A is fully captured into W before L's own (independent) zero-fill below would
            // otherwise clobber it.
            var W = ws.W;
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++)
                    W[i, j] = A[j, i];

            // zero L (only the lower-triangle columns 0..rank-1 get written below). A is no longer
            // read past this point, so it is safe for L to alias A from here on.
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    L[i, j] = 0;

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

            // Panel width for the blocked (level-3) path below. Method-local const -- CHOP is a partial
            // class shared by the float/double generated files, so a class-level const of the same name
            // would collide across them (CS0102; see CHO's CHOL_BLOCK).
            const int CHOLP_BLOCK = 32;

            // Size gate: measured crossover, higher than plain CHO's gate since the panel phase here
            // is heavier (see the blocked path below).
            const int CHOLP_BLOCK_MIN_N = Consts.floatCholPivotBlockMinN;

            if (n < CHOLP_BLOCK_MIN_N) {
                // Small matrix: plain per-column right-looking sweep, unchanged.
                //
                // Freshly-computed factor row U[k, k..n-1] gathered contiguously into urow so the rank-1
                // Schur update is a set of unit-stride row-axpys (the vectorising UnsafeOP.axpy path).
                // One O(n) Temp buffer (« the O(n^3) factor), matching the plain CHO.decomp's `lj`.
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
                            return new RankInfo { status = DirectSolveStatus.Indefinite, rank = rank };
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
                                        return new RankInfo { status = DirectSolveStatus.Indefinite, rank = rank };
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
                            Swap.Rows(ref L, k, q, 0, k);    // permute the already-computed factor rows
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
                            UnsafeOP.axpy(wp + (long)i * n + i, urowp + i, -urowp[i], n - i);
                    }
                }

                urow.Dispose();
                return new RankInfo
                {
                    status = (rank < n) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                    rank = rank
                };
            }

            // ---- blocked (level-3) path — LAPACK-style right-looking PSTRF ----
            // Port of Lucas/Higham dpstrf.f (upper-triangular branch). Unlike CHO/LU, pivot selection
            // needs the largest remaining diagonal over the FULL trailing range at every column, not
            // just the panel, so the panel can't defer everything to one end-of-panel update:
            //   (a) CHEAP: `dot[i]` accumulates, per block, each finished panel row's contribution to
            //       row i's diagonal; `W[i,i] - dot[i]` gives the exact Schur-complement diagonal, used
            //       only for pivot search.
            //   (b) WINNER-ONLY: once a column is chosen, its full remaining row is corrected against
            //       every earlier-in-block finished row (unit-stride axpys) before use, since a pivot
            //       from deep in the trailing block was never touched by earlier updates.
            // The trailing block is updated once per panel via UnsafeOP.syrkUpperSub.
            //
            // Two deviations from a literal port: Ukk is read straight from the pivot search's maxDiag
            // (provably identical to re-deriving it, skips redundant work), and this port always
            // searches for a pivot rather than reusing LAPACK's precomputed first-column pivot. Also,
            // distinguishing rank-deficient from indefinite (this library's RankInfo, beyond LAPACK's
            // single INFO=1) requires W accurate before the off-diagonal scan, so the rare branch that
            // trips the tolerance check first flushes this block's pending columns [j0,k) via the same
            // syrkUpperSub kernel, scoped narrower.
            unsafe {
                float* wp = W.Data.Ptr;

                // dot[i]: see (a) above. Reset per block; only entries [j0,n) are read.
                var dotBuf = new floatN(n, Allocator.Temp, false);
                float* dotp = dotBuf.Data.Ptr;

                // QT: transposed gather of a jb x ntrail panel-rows-at-trailing-columns block into
                // ntrail x jb contiguous scratch (see UnsafeOP.syrkUpperSub). Sized for the worst case
                // (first panel, j0=0: jb=CHOLP_BLOCK, ntrail<=n); reused for both the normal panel-end
                // SYRK and the narrower rare-path flush.
                var QT = new floatN(CHOLP_BLOCK * n, Allocator.Temp, false);
                float* qtp = QT.Data.Ptr;

                for (int j0 = 0; j0 < n; j0 += CHOLP_BLOCK) {

                    int jb = math.min(CHOLP_BLOCK, n - j0);
                    int panelEnd = j0 + jb;

                    for (int i = j0; i < n; i++) dotp[i] = 0;

                    for (int k = j0; k < panelEnd; k++) {

                        // (a) cheap diagonal-only update + pivot search over the FULL remaining range.
                        if (k > j0) {
                            float* prevRow = wp + (long)(k - 1) * n;
                            for (int i = k; i < n; i++) dotp[i] += prevRow[i] * prevRow[i];
                        }

                        int q = k;
                        float maxDiag = W[k, k] - dotp[k];
                        float minDiag = maxDiag;
                        for (int i = k + 1; i < n; i++) {
                            float d = W[i, i] - dotp[i];
                            if (d > maxDiag) { maxDiag = d; q = i; }
                            if (d < minDiag) minDiag = d;
                        }

                        // a clearly-negative remaining diagonal => not PSD. Diagonal-only, so `dot`
                        // already gives an exact answer -- no flush needed (mirrors the unblocked path,
                        // which also skips the off-diagonal scan on this branch).
                        if (minDiag < -stopTol) {
                            QT.Dispose(); dotBuf.Dispose();
                            rank = k;
                            return new RankInfo { status = DirectSolveStatus.Indefinite, rank = rank };
                        }

                        // largest remaining diagonal is numerically zero (NaN-safe) -- same rank/indefinite
                        // split as the unblocked path, but W's off-diagonal entries in [k,n) are only
                        // accurate up to the LAST completed panel (this block's own columns [j0,k) are
                        // still deferred), so flush them first via the same SYRK kernel the panel-end
                        // update below uses, scoped to just the finished prefix.
                        if (!(maxDiag > stopTol)) {
                            int fjb = k - j0;
                            if (fjb > 0) {
                                int flen = n - k;
                                for (int p = 0; p < fjb; p++) {
                                    float* rowp = wp + (long)(j0 + p) * n + k;
                                    for (int ip = 0; ip < flen; ip++) qtp[(long)ip * fjb + p] = rowp[ip];
                                }
                                UnsafeOP.syrkUpperSub(wp, n, k, j0, fjb, qtp);
                            }

                            for (int i = k; i < n; i++)
                                for (int j = i; j < n; j++)
                                    if (math.abs(W[i, j]) > stopTol) {
                                        QT.Dispose(); dotBuf.Dispose();
                                        rank = k;
                                        return new RankInfo { status = DirectSolveStatus.Indefinite, rank = rank };
                                    }

                            // no early dispose here -- falls through to the shared post-loop Dispose +
                            // return below (breaking twice, via the `if (rank < n) break;` right after
                            // this k-loop, to also exit the j0 loop).
                            rank = k;
                            break;
                        }

                        // symmetric pivot k <-> q: same four-segment swap as the unblocked path (still
                        // full-width -- pivoting permutes the WHOLE matrix, not just the panel), plus the
                        // raw diagonal + dot bookkeeping the deferred scheme needs (unblocked doesn't,
                        // since its diagonal is always already exact). W[k,k]'s raw value is about to be
                        // overwritten by Ukk below regardless, so only the demoted (q-bound) side needs an
                        // explicit write.
                        if (q != k) {
                            float demoted = W[k, k];
                            for (int i = 0; i < k; i++)     { float t = W[i, k]; W[i, k] = W[i, q]; W[i, q] = t; }
                            for (int j = q + 1; j < n; j++) { float t = W[k, j]; W[k, j] = W[q, j]; W[q, j] = t; }
                            for (int m = k + 1; m < q; m++) { float t = W[k, m]; W[k, m] = W[m, q]; W[m, q] = t; }
                            W[q, q] = demoted;
                            { float t = dotp[k]; dotp[k] = dotp[q]; dotp[q] = t; }
                            Swap.Rows(ref L, k, q, 0, k);    // permute the already-computed factor rows
                            P.Swap(k, q);
                        }

                        float Ukk = math.sqrt(maxDiag);
                        L[k, k] = Ukk;

                        // (b) expensive, winner-only: bring row k's FULL remaining width up to date
                        // against every earlier-in-this-block finished row (unit-stride row axpys, not a
                        // strided dot -- see the section header), then scale.
                        if (k < n - 1) {
                            float* krow = wp + (long)k * n;
                            for (int c = j0; c < k; c++) {
                                float Wck = W[c, k];
                                UnsafeOP.axpy(krow + (k + 1), wp + (long)c * n + (k + 1), -Wck, n - (k + 1));
                            }

                            float inv = (float)1 / Ukk;
                            for (int j = k + 1; j < n; j++) {
                                float u = krow[j] * inv;
                                krow[j] = u;
                                L[j, k] = u;
                            }
                        }
                    }

                    if (rank < n) break;   // rank-deficient/indefinite exit inside the panel loop above

                    // level-3 SYRK trailing update: W[panelEnd:n,panelEnd:n) -= U12ᵀ·U12, U12 = this
                    // panel's finished rows [j0,panelEnd) restricted to the trailing columns -- ONE call
                    // per panel instead of one rank-1 update per (row,column) pair.
                    int rStart = panelEnd;
                    if (rStart < n) {
                        int ntrail = n - rStart;
                        for (int p = 0; p < jb; p++) {
                            float* rowp = wp + (long)(j0 + p) * n + rStart;
                            for (int ip = 0; ip < ntrail; ip++) qtp[(long)ip * jb + p] = rowp[ip];
                        }
                        UnsafeOP.syrkUpperSub(wp, n, rStart, j0, jb, qtp);
                    }
                }

                QT.Dispose();
                dotBuf.Dispose();
            }

            return new RankInfo
            {
                status = (rank < n) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                rank = rank
            };
        }

        /// <summary>
        /// decomp allocating its n x n symmetric working copy from Allocator.Temp.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static RankInfo decomp(in floatMxN A, ref floatMxN L, ref Pivot P) {
            // Replicate the ref-workspace overload's dimension checks BEFORE allocating ws.W, so a
            // caller error can't leak the Temp allocation (thrown mid-call, past the alloc but
            // before the Dispose below).
            if (!A.IsSquare)
                throw new ArgumentException("decomp: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("decomp: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("decomp: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (P.N != n)
                throw new ArgumentException("decomp: P.N must equal A dimension");

            var ws = new floatCHOPCache
            {
                W = new floatMxN(n, n, Allocator.Temp),
                bt = default
            };
            var info = decomp(in A, ref L, ref P, ref ws);
            ws.W.Dispose();
            return info;
        }

        /// <summary>
        /// Solve A x = b using a pivoted-Cholesky factor (Pᵀ·A·P = L·Lᵀ) from decomp. b is
        /// overwritten with x (length n). L is passed by ref to feed the triangular solvers (it is
        /// not modified). Internally allocates small Allocator.Temp workspaces.
        ///
        /// - Full rank (rank == n): the exact solution, via permuted forward/back substitution.
        /// - Rank-deficient (rank &lt; n): the minimum-norm least-squares solution x = A⁺b. With the
        ///   rank factorization A = M·Mᵀ (M = P·L₁, the n×rank factor), the symmetric pseudoinverse is
        ///   A⁺ = M (MᵀM)⁻² Mᵀ; this forms the rank×rank SPD Gram matrix G = L₁ᵀL₁, factors it, and
        ///   applies G⁻¹ twice. If b ∈ range(A) this reproduces the exact solution; otherwise it
        ///   returns the least-squares solution of smallest norm.
        /// Always reports DirectSolveStatus.Success — this assumes a valid factor and the given
        /// rank are already known-good (from a decomp that succeeded); it does not re-verify them.
        /// </summary>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo decompSolve(ref floatMxN L, in Pivot P, int rank, ref floatN b_to_x,
                                              ref floatCHOPCache ws) {
            if (!L.IsSquare)
                throw new ArgumentException("decompSolve: L must be square");

            int n = L.M_Rows;

            if (b_to_x.N != n)
                throw new ArgumentException("decompSolve: b_to_x.N must equal L.M_Rows");

            if (P.N != n)
                throw new ArgumentException("decompSolve: P.N must equal L.M_Rows");

            if (rank < 0 || rank > n)
                throw new ArgumentException("decompSolve: rank must be in [0, n]");
            RequireCholeskyPivotWorkspace(in ws, n, false, true);

            // x = A⁺b = 0 for the zero matrix.
            if (rank == 0) {
                for (int i = 0; i < n; i++)
                    b_to_x[i] = 0;
                return new DirectSolveInfo { status = DirectSolveStatus.Success };
            }

            // gather b̃[i] = b[P[i]] (apply the symmetric permutation to the RHS) into the workspace.
            var bt = ws.bt;
            for (int i = 0; i < n; i++)
                bt[i] = b_to_x[P[i]];

            if (rank == n) {
                // full rank: L z = b̃, then Lᵀ x̃ = z (the same two-step solve as CHO.decompSolve),
                // then scatter x[P[i]] = x̃[i].
                CHO.decompSolve(ref L, ref bt);
                for (int i = 0; i < n; i++)
                    b_to_x[P[i]] = bt[i];
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
            if (!CHO.decomp(in G, ref GL)) {
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
                CHO.decomp(in G, ref GL);
            }
            CHO.decompSolve(ref GL, ref g);   // g := G⁻¹ g
            CHO.decompSolve(ref GL, ref g);   // g := G⁻² g_orig = z

            // x = M z : t = L₁ z (n-vector), then scatter x[P[i]] = t[i].
            for (int i = 0; i < n; i++) {
                float s = 0;
                int kmax = math.min(i + 1, r);   // L[i,k] = 0 for k > i
                for (int k = 0; k < kmax; k++)
                    s += L[i, k] * g[k];
                b_to_x[P[i]] = s;
            }

            GL.Dispose();
            G.Dispose();
            g.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// decompSolve allocating its permuted-RHS scratch (length n) from Allocator.Temp.
        /// See the ref-workspace overload for semantics (the rank-deficient Gram buffers are per-call
        /// Temp in both forms). Always reports DirectSolveStatus.Success — see that overload.
        /// </summary>
        public static DirectSolveInfo decompSolve(ref floatMxN L, in Pivot P, int rank, ref floatN b_to_x) {
            // Replicate the ref-workspace overload's dimension checks BEFORE allocating ws.bt, so a
            // caller error can't leak the Temp allocation (thrown mid-call, past the alloc but
            // before the Dispose below).
            if (!L.IsSquare)
                throw new ArgumentException("decompSolve: L must be square");

            int n = L.M_Rows;

            if (b_to_x.N != n)
                throw new ArgumentException("decompSolve: b_to_x.N must equal L.M_Rows");

            if (P.N != n)
                throw new ArgumentException("decompSolve: P.N must equal L.M_Rows");

            if (rank < 0 || rank > n)
                throw new ArgumentException("decompSolve: rank must be in [0, n]");

            var ws = new floatCHOPCache
            {
                W = default,
                bt = new floatN(n, Allocator.Temp)
            };
            var info = decompSolve(ref L, in P, rank, ref b_to_x, ref ws);
            ws.bt.Dispose();
            return info;
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve in one call: factors A_to_L IN PLACE (rank-revealing,
        /// L aliases A's own storage -- see decomp's read-before-write ordering note) and solves
        /// A x = b, b overwritten with x. Returns Indefinite (forwarded from the decomposition)
        /// WITHOUT solving if A is indefinite. For a rank-deficient (PSD) A this returns
        /// RankDeficient (with the detected rank) and the minimum-norm least-squares solution.
        /// A_to_L holds the factorization on return; valid input to decompSolve.
        /// </summary>
        /// <param name="A_to_L">On entry A; on exit the lower-triangular factor L.</param>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static RankInfo solveInPlace(ref floatMxN A_to_L, ref Pivot P, ref floatN b_to_x) {
            // Validate everything decomp/decompSolve would check BEFORE decomp runs, so a caller
            // error (e.g. a mis-sized b_to_x) cannot destroy A_to_L first (L aliases A_to_L's own
            // storage inside decomp).
            if (!A_to_L.IsSquare)
                throw new ArgumentException("solveInPlace: A_to_L needs to be square");

            int n = A_to_L.M_Rows;

            if (P.N != n)
                throw new ArgumentException("solveInPlace: P.N must equal A_to_L dimension");

            if (b_to_x.N != n)
                throw new ArgumentException("solveInPlace: b_to_x.N must equal A_to_L.M_Rows");

            var decompInfo = decomp(in A_to_L, ref A_to_L, ref P);
            if (!decompInfo.Solved)
                return decompInfo;

            decompSolve(ref A_to_L, in P, decompInfo.rank, ref b_to_x);
            return decompInfo;
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve in place using a caller workspace (W for the
        /// factorization, bt for the solve). Same destructive contract as the non-workspace overload:
        /// A_to_L is factored using its own storage and holds the factorization on return. Returns
        /// Indefinite (forwarded from the decomposition) WITHOUT solving if A is indefinite. For a
        /// rank-deficient (PSD) A this returns RankDeficient (with the detected rank) and the
        /// minimum-norm least-squares solution.
        /// </summary>
        /// <param name="A_to_L">On entry A; on exit the lower-triangular factor L.</param>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static RankInfo solveInPlace(ref floatMxN A_to_L, ref Pivot P, ref floatN b_to_x,
                                              ref floatCHOPCache ws) {
            // Validate everything decomp/decompSolve would check — including the FULL workspace
            // (both W and bt) — BEFORE decomp runs, so a caller error (e.g. a mis-sized bt or
            // b_to_x) cannot destroy A_to_L first (L aliases A_to_L's own storage inside decomp).
            if (!A_to_L.IsSquare)
                throw new ArgumentException("solveInPlace: A_to_L needs to be square");

            int n = A_to_L.M_Rows;

            if (P.N != n)
                throw new ArgumentException("solveInPlace: P.N must equal A_to_L dimension");

            if (b_to_x.N != n)
                throw new ArgumentException("solveInPlace: b_to_x.N must equal A_to_L.M_Rows");

            RequireCholeskyPivotWorkspace(in ws, n, true, true);

            var decompInfo = decomp(in A_to_L, ref A_to_L, ref P, ref ws);
            if (!decompInfo.Solved)
                return decompInfo;

            decompSolve(ref A_to_L, in P, decompInfo.rank, ref b_to_x, ref ws);
            return decompInfo;
        }

        // ---- multi-RHS forms: solve A X = B for a whole matrix of right-hand sides ----
        // Each RHS is a COLUMN of B_to_X (n x k). Mirrors the vector decompSolve exactly, generalised
        // to k columns: the symmetric permutation is applied to B's ROWS, and every rank-deficient
        // Gram step (g, z, x = M z) becomes an axpy across the k columns. Allocates Allocator.Temp
        // scratch internally (the permuted-RHS block plus the r×r Gram buffers).

        /// <summary>
        /// Solve A X = B (multi-RHS) using a pivoted-Cholesky factor (Pᵀ·A·P = L·Lᵀ) from decomp;
        /// B_to_X (n x k) is overwritten with X. Full rank: exact solution via permuted forward/back
        /// substitution. Rank-deficient: the minimum-norm least-squares solution X = A⁺B, formed exactly
        /// as the vector overload (Gram matrix G = L₁ᵀL₁ factored once, G⁻¹ applied twice), per column.
        /// Always reports Success — assumes a valid factor + known-good rank.
        /// </summary>
        /// <param name="B_to_X">On entry B (n rows x k cols); on exit the solution X.</param>
        public static unsafe DirectSolveInfo decompSolve(ref floatMxN L, in Pivot P, int rank, ref floatMxN B_to_X) {
            if (!L.IsSquare)
                throw new ArgumentException("decompSolve: L must be square");

            int n = L.M_Rows;
            int k = B_to_X.N_Cols;

            if (B_to_X.M_Rows != n)
                throw new ArgumentException("decompSolve: B_to_X.M_Rows must equal L.M_Rows");

            if (P.N != n)
                throw new ArgumentException("decompSolve: P.N must equal L.M_Rows");

            if (rank < 0 || rank > n)
                throw new ArgumentException("decompSolve: rank must be in [0, n]");

            // X = A⁺B = 0 for the zero matrix.
            if (rank == 0) {
                float* Xz = B_to_X.Data.Ptr;
                for (long e = 0; e < (long)n * k; e++) Xz[e] = (float)0;
                return new DirectSolveInfo { status = DirectSolveStatus.Success };
            }

            // B̃[i,:] = B[P[i],:] (apply the symmetric permutation to the RHS rows).
            var Bt = new floatMxN(n, k, Allocator.Temp, false);
            {
                float* Btp = Bt.Data.Ptr;
                float* Xp = B_to_X.Data.Ptr;
                for (int i = 0; i < n; i++)
                {
                    float* dst = Btp + (long)i * k;
                    float* src = Xp + (long)P[i] * k;
                    for (int c = 0; c < k; c++) dst[c] = src[c];
                }
            }

            if (rank == n) {
                // full rank: L Z = B̃ then Lᵀ X̃ = Z (CHO multi-RHS), then scatter X[P[i],:] = X̃[i,:].
                CHO.decompSolve(ref L, ref Bt);
                float* Btp = Bt.Data.Ptr;
                float* Xp = B_to_X.Data.Ptr;
                for (int i = 0; i < n; i++)
                {
                    float* dst = Xp + (long)P[i] * k;
                    float* src = Btp + (long)i * k;
                    for (int c = 0; c < k; c++) dst[c] = src[c];
                }
                Bt.Dispose();
                return new DirectSolveInfo { status = DirectSolveStatus.Success };
            }

            // rank-deficient minimum-norm solution.
            int r = rank;

            // Grhs = L₁ᵀ B̃  (r x k; L₁[i,c]=0 for i<c so the sum starts at i=c).
            var Grhs = new floatMxN(r, k, Allocator.Temp, false);
            {
                float* Gp = Grhs.Data.Ptr;
                float* Btp = Bt.Data.Ptr;
                for (int c = 0; c < r; c++)
                {
                    float* Grow = Gp + (long)c * k;
                    for (int i = c; i < n; i++)
                        UnsafeOP.axpy(Grow, Btp + (long)i * k, L[i, c], k);
                }
            }

            // G = L₁ᵀ L₁  (r x r SPD).
            var G = new floatMxN(r, r, Allocator.Temp);
            for (int a = 0; a < r; a++)
                for (int c = 0; c <= a; c++) {
                    float s = 0;
                    for (int i = a; i < n; i++)
                        s += L[i, a] * L[i, c];
                    G[a, c] = s;
                    G[c, a] = s;
                }

            // Z = G⁻² Grhs : factor G once, apply G⁻¹ twice (both multi-RHS).
            var GL = new floatMxN(r, r, Allocator.Temp);
            if (!CHO.decomp(in G, ref GL)) {
                // borderline-revealed rank can leave G numerically semidefinite — tiny Tikhonov ridge
                // and retry (see the vector overload).
                float gScale = 0;
                for (int a = 0; a < r; a++) {
                    float gd = math.abs(G[a, a]);
                    if (gd > gScale) gScale = gd;
                }
                float ridge = (float)r * Consts.floatEpsilon * gScale;
                for (int a = 0; a < r; a++)
                    G[a, a] += ridge;
                CHO.decomp(in G, ref GL);
            }
            CHO.decompSolve(ref GL, ref Grhs);   // Grhs := G⁻¹ Grhs
            CHO.decompSolve(ref GL, ref Grhs);   // Grhs := G⁻² Grhs = Z

            // X = M Z : t[i,:] = L₁ Z rows, then scatter X[P[i],:] = t[i,:] (L[i,c]=0 for c>i).
            {
                float* Zp = Grhs.Data.Ptr;
                float* Xp = B_to_X.Data.Ptr;
                for (int i = 0; i < n; i++)
                {
                    float* dst = Xp + (long)P[i] * k;
                    for (int c = 0; c < k; c++) dst[c] = (float)0;
                    int kmax = math.min(i + 1, r);
                    for (int c = 0; c < kmax; c++)
                        UnsafeOP.axpy(dst, Zp + (long)c * k, L[i, c], k);
                }
            }

            GL.Dispose();
            G.Dispose();
            Grhs.Dispose();
            Bt.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve in place (multi-RHS): factors A_to_L IN PLACE
        /// (rank-revealing) and solves A X = B for every column of B_to_X. Returns Indefinite WITHOUT
        /// solving if A is indefinite; RankDeficient (with rank) + minimum-norm solution if PSD but
        /// rank-deficient. A_to_L holds the factorization on return.
        /// </summary>
        /// <param name="A_to_L">On entry A; on exit the lower-triangular factor L.</param>
        /// <param name="B_to_X">On entry B (n rows x k cols); on exit the solution X.</param>
        public static RankInfo solveInPlace(ref floatMxN A_to_L, ref Pivot P, ref floatMxN B_to_X) {
            if (!A_to_L.IsSquare)
                throw new ArgumentException("solveInPlace: A_to_L needs to be square");

            int n = A_to_L.M_Rows;

            if (P.N != n)
                throw new ArgumentException("solveInPlace: P.N must equal A_to_L dimension");

            if (B_to_X.M_Rows != n)
                throw new ArgumentException("solveInPlace: B_to_X.M_Rows must equal A_to_L.M_Rows");

            var decompInfo = decomp(in A_to_L, ref A_to_L, ref P);
            if (!decompInfo.Solved)
                return decompInfo;

            decompSolve(ref A_to_L, in P, decompInfo.rank, ref B_to_X);
            return decompInfo;
        }
    }
}
