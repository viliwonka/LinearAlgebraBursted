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

            // zero L (only the lower-triangle columns 0..rank-1 get written below).
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    L[i, j] = 0;

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
            // update is a set of unit-stride row-axpys (the vectorising UnsafeOP.axpy path). One O(n) Temp
            // buffer (« the O(n^3) factor), matching the plain CHO.decomp's `lj`.
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

        /// <summary>
        /// decomp allocating its n x n symmetric working copy from Allocator.Temp.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static RankInfo decomp(in floatMxN A, ref floatMxN L, ref Pivot P) {
            int n = A.IsSquare ? A.M_Rows : 0;
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
            int n = L.IsSquare ? L.M_Rows : 0;
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
        /// Pivoted-Cholesky factor-and-solve in one call: factors A (rank-revealing) and solves
        /// A x = b, b overwritten with x. Returns Indefinite (forwarded from the decomposition)
        /// WITHOUT solving if A is indefinite. For a rank-deficient (PSD) A this returns
        /// RankDeficient (with the detected rank) and the minimum-norm least-squares solution.
        ///
        /// TRANSITIONAL (commit 1): despite the solveInPlace name, A is still preserved (`in`) —
        /// this rename lands ahead of the commit-2 behavior change that makes it genuinely
        /// destructive (A_to_L). Kept here only for the mechanical rename; do not rely on the name
        /// implying destructive semantics yet.
        /// </summary>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static RankInfo solveInPlace(in floatMxN A, ref floatMxN L, ref Pivot P, ref floatN b_to_x) {
            var decompInfo = decomp(in A, ref L, ref P);
            if (!decompInfo.Solved)
                return decompInfo;

            decompSolve(ref L, in P, decompInfo.rank, ref b_to_x);
            return decompInfo;
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve using a caller workspace (W for the factorization, bt for
        /// the solve). Returns Indefinite (forwarded from the decomposition) WITHOUT solving if A is
        /// indefinite. For a rank-deficient (PSD) A this returns RankDeficient (with the detected
        /// rank) and the minimum-norm least-squares solution.
        ///
        /// TRANSITIONAL (commit 1): see the non-workspace overload's note — A is still preserved
        /// (`in`) despite the solveInPlace name; commit 2 makes this genuinely destructive.
        /// </summary>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static RankInfo solveInPlace(in floatMxN A, ref floatMxN L, ref Pivot P, ref floatN b_to_x,
                                              ref floatCHOPCache ws) {
            var decompInfo = decomp(in A, ref L, ref P, ref ws);
            if (!decompInfo.Solved)
                return decompInfo;

            decompSolve(ref L, in P, decompInfo.rank, ref b_to_x, ref ws);
            return decompInfo;
        }
    }
}
