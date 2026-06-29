#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
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
        public static bool choleskyDecomposition(in doubleMxN A, ref doubleMxN L) {
            if (!A.IsSquare)
                throw new ArgumentException("choleskyDecomposition: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("choleskyDecomposition: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("choleskyDecomposition: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (n == 0) return true;

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

                for (int j = 0; j < n; j++)
                {
                    // L[j,j] already holds A[j,j] - sum_{k<j} L[j,k]^2 (applied by earlier rank-1
                    // updates). Not positive-definite -> reject; !(d > 0) also catches NaN, before sqrt.
                    double d = L[j, j];
                    if (!(d > (double)0)) { lj.Dispose(); return false; }

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
        public static void choleskySolve(ref doubleMxN L, ref doubleN b) {
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
        public static bool choleskySolve(in doubleMxN A, ref doubleMxN L, ref doubleN b) {
            if (!choleskyDecomposition(in A, ref L))
                return false;

            choleskySolve(ref L, ref b);
            return true;
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
        /// Returns true (PSD, any rank 0..n) or FALSE if A is indefinite — a Schur-complement diagonal
        /// goes significantly negative (below -n·eps·max|diag|). On a false return L/P/rank are
        /// undefined. Tolerance is scale-relative to the largest |diagonal| so it is scale-invariant.
        /// </summary>
        public static bool choleskyDecompositionPivot(in doubleMxN A, ref doubleMxN L, ref Pivot P, out int rank,
                                                       ref doubleCholeskyPivotWorkspace ws) {
            if (!A.IsSquare)
                throw new ArgumentException("choleskyDecompositionPivot: A needs to be square");

            if (!L.IsSquare)
                throw new ArgumentException("choleskyDecompositionPivot: L needs to be square");

            if (A.M_Rows != L.M_Rows)
                throw new ArgumentException("choleskyDecompositionPivot: A and L need to have the same dimensions");

            int n = A.M_Rows;

            if (P.N != n)
                throw new ArgumentException("choleskyDecompositionPivot: P.N must equal A dimension");
            RequireCholeskyPivotWorkspace(in ws, n, true, false, "choleskyDecompositionPivot");

            P.Reset();
            rank = n;

            if (n == 0)
                return true;

            // zero L (only the lower-triangle columns 0..rank-1 get written below).
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    L[i, j] = 0;

            // Working full symmetric matrix W (caller workspace): pivoting needs a destroyable symmetric
            // copy, and keeping BOTH triangles lets each symmetric pivot be a plain row-swap + column-swap.
            var W = ws.W;
            for (int i = 0; i < n; i++)
                for (int j = 0; j <= i; j++) {
                    double v = A[i, j];
                    W[i, j] = v;
                    W[j, i] = v;
                }

            // scale-relative numerical-zero / indefinite tolerance. Scanned over ALL entries (not
            // just the diagonal): for a genuine PSD matrix the largest magnitude IS on the diagonal,
            // so this is identical there — but it keeps the tolerance meaningful for a non-PSD input
            // whose mass is off-diagonal (e.g. zero diagonal, nonzero off-diagonal), which must not
            // be silently accepted as a rank-0 PSD matrix.
            double absScale = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double ad = math.abs(W[i, j]);
                    if (ad > absScale) absScale = ad;
                }
            double stopTol = (double)n * Consts.doubleEpsilon * absScale;

            for (int k = 0; k < n; k++) {

                // pick the largest remaining diagonal (the pivot); also track the smallest to detect
                // indefiniteness (a PSD Schur complement keeps every diagonal >= 0).
                int q = k;
                double maxDiag = W[k, k];
                double minDiag = W[k, k];
                for (int j = k + 1; j < n; j++) {
                    double d = W[j, j];
                    if (d > maxDiag) { maxDiag = d; q = j; }
                    if (d < minDiag) minDiag = d;
                }

                // a clearly-negative remaining diagonal => not PSD.
                if (minDiag < -stopTol) {
                    rank = k;
                    return false;
                }

                // largest remaining diagonal is numerically zero (NaN-safe). For a genuine PSD matrix
                // a zero diagonal forces its whole row/column to zero, so the trailing block is now
                // all-negligible and rank k is reached. But if any trailing entry is still
                // significant the matrix is NOT PSD (e.g. [[0,1],[1,0]], eigenvalues +/-1) => reject
                // as indefinite rather than silently returning a bogus low rank.
                if (!(maxDiag > stopTol)) {
                    for (int i = k; i < n; i++)
                        for (int j = k; j < n; j++)
                            if (math.abs(W[i, j]) > stopTol) {
                                rank = k;
                                return false;
                            }
                    rank = k;
                    break;
                }

                // symmetric pivot: bring column/row q to position k.
                if (q != k) {
                    SwapOP.Rows(ref W, k, q);          // full row swap
                    SwapOP.Columns(ref W, k, q);       // + full column swap => symmetric
                    SwapOP.Rows(ref L, k, q, 0, k);    // permute the already-computed factor rows
                    P.Swap(k, q);
                }

                // factor column k.
                double Lkk = math.sqrt(W[k, k]);
                L[k, k] = Lkk;
                for (int i = k + 1; i < n; i++)
                    L[i, k] = W[i, k] / Lkk;

                // rank-1 Schur update of the trailing block, kept symmetric for the next pivot.
                for (int i = k + 1; i < n; i++) {
                    double Lik = L[i, k];
                    for (int j = k + 1; j <= i; j++) {
                        W[i, j] -= Lik * L[j, k];
                        W[j, i] = W[i, j];
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// choleskyDecompositionPivot allocating its n x n symmetric working copy from Allocator.Temp.
        /// See the ref-workspace overload for semantics.
        /// </summary>
        public static bool choleskyDecompositionPivot(in doubleMxN A, ref doubleMxN L, ref Pivot P, out int rank) {
            int n = A.IsSquare ? A.M_Rows : 0;
            var ws = new doubleCholeskyPivotWorkspace
            {
                W = new doubleMxN(n, n, Allocator.Temp),
                bt = default
            };
            bool ok = choleskyDecompositionPivot(in A, ref L, ref P, out rank, ref ws);
            ws.W.Dispose();
            return ok;
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
        /// </summary>
        public static void choleskyPivotSolve(ref doubleMxN L, in Pivot P, int rank, ref doubleN b,
                                              ref doubleCholeskyPivotWorkspace ws) {
            if (!L.IsSquare)
                throw new ArgumentException("choleskyPivotSolve: L must be square");

            int n = L.M_Rows;

            if (b.N != n)
                throw new ArgumentException("choleskyPivotSolve: b.N must equal L.M_Rows");

            if (P.N != n)
                throw new ArgumentException("choleskyPivotSolve: P.N must equal L.M_Rows");

            if (rank < 0 || rank > n)
                throw new ArgumentException("choleskyPivotSolve: rank must be in [0, n]");
            RequireCholeskyPivotWorkspace(in ws, n, false, true, "choleskyPivotSolve");

            // x = A⁺b = 0 for the zero matrix.
            if (rank == 0) {
                for (int i = 0; i < n; i++)
                    b[i] = 0;
                return;
            }

            // gather b̃[i] = b[P[i]] (apply the symmetric permutation to the RHS) into the workspace.
            var bt = ws.bt;
            for (int i = 0; i < n; i++)
                bt[i] = b[P[i]];

            if (rank == n) {
                // full rank: L z = b̃, then Lᵀ x̃ = z, then scatter x[P[i]] = x̃[i].
                Solvers.SolveLowerTriangular(ref L, ref bt);
                SolveUpperTriangularTransposed(ref L, ref bt);
                for (int i = 0; i < n; i++)
                    b[P[i]] = bt[i];
                return;
            }

            // rank-deficient minimum-norm solution.
            int r = rank;

            // g = L₁ᵀ b̃   (L₁[i,k] = 0 for i < k, so the sum starts at i = k).
            var g = new doubleN(r, Allocator.Temp);
            for (int k = 0; k < r; k++) {
                double s = 0;
                for (int i = k; i < n; i++)
                    s += L[i, k] * bt[i];
                g[k] = s;
            }

            // G = L₁ᵀ L₁  (r×r SPD, since L₁ has full column rank r).
            var G = new doubleMxN(r, r, Allocator.Temp);
            for (int a = 0; a < r; a++)
                for (int c = 0; c <= a; c++) {
                    double s = 0;
                    for (int i = a; i < n; i++)   // i >= a >= c, and L[i,c] valid there
                        s += L[i, a] * L[i, c];
                    G[a, c] = s;
                    G[c, a] = s;
                }

            // z = G⁻² g : factor G once, apply G⁻¹ twice.
            var GL = new doubleMxN(r, r, Allocator.Temp);
            if (!choleskyDecomposition(in G, ref GL)) {
                // G = L₁ᵀL₁ is SPD in exact arithmetic, but a borderline-revealed rank (a pivot just
                // above stopTol) can leave it numerically semidefinite, so the inner factorization
                // fails and the two solves below would divide by a ~0 pivot and emit NaN. Restore SPD
                // with a tiny Tikhonov ridge (negligible vs G's scale) and retry — Burst-friendly,
                // non-throwing, and far more useful than returning NaN.
                double gScale = 0;
                for (int a = 0; a < r; a++) {
                    double gd = math.abs(G[a, a]);
                    if (gd > gScale) gScale = gd;
                }
                double ridge = (double)r * Consts.doubleEpsilon * gScale;
                for (int a = 0; a < r; a++)
                    G[a, a] += ridge;
                choleskyDecomposition(in G, ref GL);
            }
            choleskySolve(ref GL, ref g);   // g := G⁻¹ g
            choleskySolve(ref GL, ref g);   // g := G⁻² g_orig = z

            // x = M z : t = L₁ z (n-vector), then scatter x[P[i]] = t[i].
            for (int i = 0; i < n; i++) {
                double s = 0;
                int kmax = math.min(i + 1, r);   // L[i,k] = 0 for k > i
                for (int k = 0; k < kmax; k++)
                    s += L[i, k] * g[k];
                b[P[i]] = s;
            }

            GL.Dispose();
            G.Dispose();
            g.Dispose();
        }

        /// <summary>
        /// choleskyPivotSolve allocating its permuted-RHS scratch (length n) from Allocator.Temp.
        /// See the ref-workspace overload for semantics (the rank-deficient Gram buffers are per-call
        /// Temp in both forms).
        /// </summary>
        public static void choleskyPivotSolve(ref doubleMxN L, in Pivot P, int rank, ref doubleN b) {
            int n = L.IsSquare ? L.M_Rows : 0;
            var ws = new doubleCholeskyPivotWorkspace
            {
                W = default,
                bt = new doubleN(n, Allocator.Temp)
            };
            choleskyPivotSolve(ref L, in P, rank, ref b, ref ws);
            ws.bt.Dispose();
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve in one call: factors A (rank-revealing) and solves
        /// A x = b, b overwritten with x. Returns false WITHOUT solving if A is indefinite. For a
        /// rank-deficient (PSD) A this returns the minimum-norm least-squares solution.
        /// </summary>
        public static bool choleskyPivotSolve(in doubleMxN A, ref doubleMxN L, ref Pivot P, ref doubleN b) {
            if (!choleskyDecompositionPivot(in A, ref L, ref P, out int rank))
                return false;

            choleskyPivotSolve(ref L, in P, rank, ref b);
            return true;
        }

        /// <summary>
        /// Pivoted-Cholesky factor-and-solve using a caller workspace (W for the factorization, bt for
        /// the solve). Returns false WITHOUT solving if A is indefinite. For a rank-deficient (PSD) A
        /// this returns the minimum-norm least-squares solution.
        /// </summary>
        public static bool choleskyPivotSolve(in doubleMxN A, ref doubleMxN L, ref Pivot P, ref doubleN b,
                                              ref doubleCholeskyPivotWorkspace ws) {
            if (!choleskyDecompositionPivot(in A, ref L, ref P, out int rank, ref ws))
                return false;

            choleskyPivotSolve(ref L, in P, rank, ref b, ref ws);
            return true;
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
    }
}
