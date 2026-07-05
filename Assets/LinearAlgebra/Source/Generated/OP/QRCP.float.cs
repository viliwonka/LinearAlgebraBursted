#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    /// <summary>
    /// Column-pivoted (rank-revealing) QR — Businger-Golub. Split out of <see cref="QR"/> because
    /// the pivoted family carries its own Pivot/rank contract; shares QR's private Householder
    /// kernels (genHouseholder / applyReflectorRight, marked internal for this reason).
    /// </summary>
    /// <remarks>
    /// NO floatQRCache here (deliberate — see docs/spec-solver-api-rework.md OQ-7): the pivot
    /// decision at each step depends on trailing column norms recomputed AFTER the previous step's
    /// reflector is applied, so panels of columns can't be factored ahead of their pivot choice —
    /// this kernel is inherently level-2 (unblocked) and will never use the compact-WY buffers
    /// (Vpanel/Tbuf/Wbuf/tcolBuf/VfullBuf) that make up most of QR's cache. Sharing that cache here
    /// would leave those five fields permanently dead for every QRCP call. QRCP's own scratch need
    /// (u, plus an internal w/colNorm2 pair the decompInPlace/solveInPlace overloads still allocate
    /// from Allocator.Temp) stays as-is; widening it to a dedicated zero-alloc workspace is a
    /// separate, smaller future change, not part of this commit.
    /// </remarks>
    public static partial class QRCP {

        // Column-pivoted (rank-revealing) QR — Businger–Golub. Factorizes A·P = Q·R, where the
        // column permutation P is chosen greedily so the pivot at each step is the trailing column
        // of largest 2-norm. This forces the magnitudes of the R diagonal to be non-increasing
        // (|R[0,0]| >= |R[1,1]| >= ... >= |R[n-1,n-1]|), so trailing near-zero diagonal entries
        // reveal the numerical rank — the stable choice for rank-deficient least squares where the
        // plain (un-pivoted) QR.decompInPlace requires full column rank.
        //
        //   A_to_Q in:  A (m x n, m >= n)              out: orthogonal Q (m x n)
        //   R      out: upper triangular R (n x n)
        //   P      out: column Pivot, size n. Reset internally. Result column j is original column P[j];
        //           equivalently A[:, P[j]] == (Q*R)[:, j].
        //   u      scratch Householder vector, length EXACTLY A_to_Q.M_Rows.
        //
        // Partial column norms are recomputed exactly at each step (rows d..m-1) rather than
        // downdated. That is the same O(n^2 m) order as the reflector sweep itself, and it sidesteps
        // the catastrophic-cancellation failure mode of norm downdating (LAPACK xGEQPF needs a
        // recompute guard precisely because the cheap downdate loses all accuracy near rank
        // deficiency) — for the modest matrices this library targets, exact recompute is both
        // simpler and unconditionally robust.
        // Always reports DirectSolveStatus.Success — this factorization has no failure mode; it does
        // NOT itself compute an integer rank (see solveInPlace for the rank-revealing consumer).
        /// <remarks>R must not alias A_to_Q (unchecked) — A_to_Q's upper triangle is read to build R
        /// while A_to_Q is simultaneously being overwritten with the Q reconstruction.</remarks>
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref floatMxN A_to_Q, ref floatMxN R, ref Pivot P, ref floatN u)
        {
            if (A_to_Q.M_Rows < A_to_Q.N_Cols)
                throw new ArgumentException("QRCP.decompInPlace: Matrix must be square or tall (M_Rows >= N_Cols)");

            if (u.N != A_to_Q.M_Rows)
                throw new ArgumentException("QRCP.decompInPlace: scratch vector u.N must equal A_to_Q.M_Rows");

            if (P.N != A_to_Q.N_Cols)
                throw new ArgumentException("QRCP.decompInPlace: pivot P.N must equal A_to_Q.N_Cols");

            if (R.M_Rows != A_to_Q.N_Cols || R.N_Cols != A_to_Q.N_Cols)
                throw new ArgumentException("QRCP.decompInPlace: R must be N_Cols x N_Cols");

            P.Reset();

            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            // Reflector-apply accumulator (length n) + per-column squared-norm buffer for pivoting.
            // Allocated once per call (O(n) « O(n³)); this path has no zero-alloc w contract, unlike
            // QR.decompInPlace's 4-arg overload.
            var w = new floatN(n, Allocator.Temp, false);
            var colNorm2 = new floatN(n, Allocator.Temp, false);

            // scale-relative zero-column threshold (see QR.genHouseholder); LInf(Q) == max |entry|.
            float zeroThreshold = Consts.floatZeroThreshold * Norms.LInf(in A_to_Q);

            for (int d = 0; d < n; d++)
            {
                // --- column pivot: among trailing columns d..n-1, pick the one whose partial 2-norm
                //     over rows d..m-1 is largest (recomputed exactly), and bring it to position d. ---

                // Squared 2-norms of all trailing columns built in ONE row-major sweep (unit-stride,
                // vectorised addSquares) rather than n separate down-a-column reductions — the same
                // restructuring as the reflector apply. Recomputed exactly each step (not downdated).
                // Bitwise identical to the per-column form: each colNorm2[c] sums rows d..m-1 in the
                // same ascending order.
                unsafe
                {
                    float* qp = A_to_Q.Data.Ptr;
                    float* cn = colNorm2.Data.Ptr;
                    int L = n - d;
                    UnsafeUtility.MemClear(cn + d, (long)L * UnsafeUtility.SizeOf<float>());
                    for (int r = d; r < m; r++)
                        UnsafeOP.addSquares(cn + d, qp + (long)r * n + d, L);
                }

                float diagNorm2 = colNorm2[d];
                int pivotCol = d;
                float maxNorm2 = diagNorm2;
                for (int c = d + 1; c < n; c++)
                {
                    if (colNorm2[c] > maxNorm2)
                    {
                        maxNorm2 = colNorm2[c];
                        pivotCol = c;
                    }
                }

                // Only pivot when the best column beats the incumbent by more than the accumulated
                // rounding noise of the norm sums (~ #terms * eps). This leaves numerically-tied
                // columns in place — notably the Kahan matrix, whose columns all have norm exactly 1
                // and which is provably invariant under column pivoting; a bare `>` would let a
                // ~1 ulp difference induce a spurious (and non-reproducible) permutation.
                float pivotRelTol = (float)(8 * m) * Consts.floatEpsilon;
                if (pivotCol != d && maxNorm2 > diagNorm2 * ((float)1 + pivotRelTol))
                {
                    // Full-column swap (all rows): rows < d hold finished R entries that must travel
                    // with the column; rows >= d hold the live sub-matrix. Stored Householder vectors
                    // of earlier steps live in columns < d and are untouched (both indices are >= d).
                    Swap.Columns(ref A_to_Q, d, pivotCol);
                    P.Swap(d, pivotCol);
                }

                QR.genHouseholder(ref A_to_Q, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix (vectorised, see QR.applyReflectorRight).
                QR.applyReflectorRight(ref A_to_Q, ref u, ref w, d);

                // R[d,d] and the stored Householder vector — see QR.decompInPlace (same pattern).
                R[d, d] = A_to_Q[d, d];

                for (int i = d; i < m; i++)
                    A_to_Q[i, d] = u[i];
            }

            // Copy the upper triangular part of Q into R
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                    R[r, c] = 0;
                else if (c > r)
                    R[r, c] = A_to_Q[r, c];
            }

            // Reconstruct Q from the Householder vectors stored in its columns (identical to the
            // un-pivoted QR.decompInPlace: pivoting only reordered the columns, not this step).
            for (int r = 0; r < m; r++)
                for (int c = r; c < n; c++)
                    if (c > r)
                        A_to_Q[r, c] = 0;

            for (int d = n - 1; d >= 0; d--)
            {
                for (int i = d; i < m; i++)
                {
                    u[i] = A_to_Q[i, d];
                    A_to_Q[i, d] = i == d ? 1 : 0;
                }

                // Apply the reflector to the trailing columns (vectorised, see QR.applyReflectorRight).
                QR.applyReflectorRight(ref A_to_Q, ref u, ref w, d);
            }

            colNorm2.Dispose();
            w.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates.
        // The caller still owns P (its size carries the column count and it is reset internally).
        /// <remarks>R must not alias A_to_Q (unchecked) — see the 4-arg overload.</remarks>
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref floatMxN A_to_Q, ref floatMxN R, ref Pivot P)
        {
            var u = new floatN(A_to_Q.M_Rows, Allocator.Temp, false);
            var info = decompInPlace(ref A_to_Q, ref R, ref P, ref u);
            u.Dispose();
            return info;
        }

        // ---- decomp: A-preserving variants (copy A into Q, then delegate to decompInPlace) ----

        /// <summary>
        /// Column-pivoted QR preserving A: A is copied into Q (one memcpy), then factored via
        /// decompInPlace. Q (caller-allocated, same dimensions as A) receives the orthogonal factor.
        /// Always reports DirectSolveStatus.Success — see decompInPlace.
        /// </summary>
        /// <remarks>R must not alias A_to_Q/Q (unchecked) — see decompInPlace. If R, P, or u is the
        /// wrong size, this throws AFTER Q has already been overwritten with a copy of A (still
        /// un-factored); A itself is always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in floatMxN A, ref floatMxN Q, ref floatMxN R, ref Pivot P, ref floatN u)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QRCP.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R, ref P, ref u);
        }

        /// <summary>
        /// decomp allocating its scratch vector u (Allocator.Temp). See the 5-arg overload for
        /// semantics.
        /// </summary>
        /// <remarks>R must not alias A_to_Q/Q (unchecked). If R or P is the wrong size, this throws
        /// AFTER Q has already been overwritten with a copy of A (still un-factored); A itself is
        /// always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in floatMxN A, ref floatMxN Q, ref floatMxN R, ref Pivot P)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QRCP.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R, ref P);
        }

        /// <summary>
        /// QRCP-based rank-safe least-squares: basic (truncated) solution. Solves A x ≈ b (m >= n)
        /// for a possibly rank-deficient A using column-pivoted QR (Businger-Golub, A·P = Q·R) to
        /// expose the numerical rank r: the R diagonal is non-increasing, so r = count of leading
        /// entries with |R[i,i]| &gt; tol, where tol = relTol * |R[0,0]| and relTol defaults to
        /// max(m,n) * Consts.floatZeroThreshold (matching SVD.pinvSolve / MatrixMetrics.rank). A
        /// negative relTol is an "auto" sentinel that selects that same default.
        ///
        /// Only the leading r×r block of R is back-substituted (divide-safe by construction: every
        /// used R diagonal exceeds tol); the remaining (n-r) free variables are set to zero in the
        /// permuted ordering, then P is un-applied to recover x. This is the BASIC (truncated)
        /// solution: it minimizes the residual ‖Ax - b‖ but is NOT the minimum-norm solution — use
        /// SVD.pinvSolve for that. When A has full column rank (r == n) the result is identical to
        /// ordinary QR least-squares.
        ///
        /// NO-COPY: factors A_to_Q's own buffer directly (no memcpy, no separate Q scratch param) —
        /// A_to_Q holds the usable orthogonal factor (alongside R and P) on return. b is preserved
        /// (read only via dot, never written).
        /// </summary>
        /// <param name="A_to_Q">On entry A (m x n, m >= n); on exit the orthogonal factor Q.</param>
        /// <param name="b">Right-hand side, length m. Preserved (read-only). Must not alias x.</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true. Solution, length n.</param>
        /// <param name="R">Scratch: n x n (receives upper-triangular factor; consumed).</param>
        /// <param name="P">Scratch: column Pivot of size n (reset internally).</param>
        /// <param name="u">Scratch: length EXACTLY m (Householder workspace; first n entries are
        /// repurposed for the un-permute scatter after the decomposition).</param>
        /// <param name="relTol">Rank threshold ratio; tol = relTol * |R[0,0]|. Negative = auto default.</param>
        /// <returns>Status Success (r == n, full rank) or RankDeficient (r &lt; n, still a usable
        /// truncated least-squares solution); rank = detected r. See
        /// <see cref="RankInfo.Solved"/>.</returns>
        /// <remarks>R must not alias A_to_Q (unchecked) — see decompInPlace.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref floatMxN A_to_Q, ref floatN b, ref floatN x,
                                           ref floatMxN R, ref Pivot P,
                                           ref floatN u, float relTol)
        {
            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            if (m < n)
                throw new ArgumentException("QRCP.solveInPlace: A_to_Q must be square or tall (M_Rows >= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("QRCP.solveInPlace: b.N must equal A_to_Q.M_Rows");
            if (x.N != n)
                throw new ArgumentException("QRCP.solveInPlace: x.N must equal A_to_Q.N_Cols");
            if (R.M_Rows != n || R.N_Cols != n)
                throw new ArgumentException("QRCP.solveInPlace: R must be N_Cols x N_Cols");
            if (P.N != n)
                throw new ArgumentException("QRCP.solveInPlace: P.N must equal A_to_Q.N_Cols");
            if (u.N != m)
                throw new ArgumentException("QRCP.solveInPlace: u.N must equal A_to_Q.M_Rows");

            // Negative relTol is an "auto" sentinel: use the library-standard rank threshold
            // (same default as SVD.pinvSolve / MatrixMetrics.rank). This also makes the threshold
            // divide-safe (tol >= 0), so a stray negative can never inflate rank into a divide-by-tiny.
            if (relTol < (float)0)
                relTol = (float)(math.max(m, n)) * Consts.floatZeroThreshold;

            // Degenerate: zero-column system.
            if (n == 0) return new RankInfo { status = DirectSolveStatus.Success, rank = 0 };

            // Step 1: QRCP — A·P = Q·R, factored directly into A_to_Q's own buffer (no copy).
            // P is reset and built inside this call.
            decompInPlace(ref A_to_Q, ref R, ref P, ref u);

            // Step 2: determine numerical rank r from R's non-increasing diagonal.
            // tol = relTol * |R[0,0]|. When R[0,0] == 0 tol == 0, and |R[0,0]| > 0 is false
            // → rank stays 0. NaN in R[0,0] → tol = NaN → all comparisons false → rank = 0.
            float tol = relTol * math.abs(R[0, 0]);
            int rank = 0;
            for (int i = 0; i < n; i++)
            {
                if (math.abs(R[i, i]) > tol)
                    rank++;
                else
                    break;
            }

            // Step 3: zero matrix (rank == 0) → x = 0, done.
            if (rank == 0)
            {
                for (int j = 0; j < n; j++)
                    x[j] = (float)0;
                return new RankInfo { status = DirectSolveStatus.RankDeficient, rank = 0 };
            }

            int r = rank;

            // Step 4: form c = Qᵀ b into x.
            // dot(in b, in A_to_Q, ref x) computes x[j] = Σ_i A_to_Q[i,j]·b[i] = (Qᵀb)[j].
            // dot zeroes x via MemClear before accumulating, so x needs no prior initialisation.
            // Guard: x must not alias b (enforced inside dot by pointer comparison).
            Blas.dot(in b, in A_to_Q, ref x);

            // Step 5: back-solve the leading r×r block of R in place.
            // x holds c = Qᵀb; overwrite x[0..r-1] with the triangular solution.
            // Every R[i,i] for i < r satisfies |R[i,i]| > tol, so no divide-by-zero.
            for (int i = r - 1; i >= 0; i--)
            {
                float sum = (float)0;
                for (int j = i + 1; j < r; j++)
                    sum += R[i, j] * x[j];
                x[i] = (x[i] - sum) / R[i, i];
            }
            // Zero the free variables (columns beyond the numerical rank).
            for (int j = r; j < n; j++)
                x[j] = (float)0;

            // Step 6: un-permute — scatter from permuted ordering back to original column ordering.
            // QRCP gives A·P = Q·R where P[j] = original column index promoted to position j.
            // The permuted solution z (in x) satisfies: x_final[P[j]] = z[j].
            // Borrow u[0..n-1] as scatter scratch (u is no longer needed after Step 1).
            for (int j = 0; j < n; j++)
                u[j] = x[j];
            for (int j = 0; j < n; j++)
                x[P[j]] = u[j];

            return new RankInfo
            {
                status = (r < n) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                rank = r
            };
        }

        // Default-tolerance overload: passes the auto sentinel (relTol < 0), so the primitive
        // uses max(m,n) * Consts.floatZeroThreshold (consistent with SVD.pinvSolve / MatrixMetrics.rank).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref floatMxN A_to_Q, ref floatN b, ref floatN x,
                                           ref floatMxN R, ref Pivot P,
                                           ref floatN u)
        {
            return solveInPlace(ref A_to_Q, ref b, ref x, ref R, ref P, ref u, (float)(-1));
        }

        /// <summary>
        /// Allocating convenience wrapper: allocates R (n×n), P (n-Pivot) and u (m) from
        /// Allocator.Temp and delegates to the zero-alloc primitive. Use the primitive in hot loops
        /// to avoid repeated Temp allocs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref floatMxN A_to_Q, ref floatN b, ref floatN x,
                                           float relTol)
        {
            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            // Replicate the primitive's own dimension checks (see the 7-arg overload above) BEFORE
            // allocating R/P/u, so a caller error can't leak the Temp allocations (thrown mid-call,
            // past the allocs but before the Dispose calls below).
            if (m < n)
                throw new ArgumentException("QRCP.solveInPlace: A_to_Q must be square or tall (M_Rows >= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("QRCP.solveInPlace: b.N must equal A_to_Q.M_Rows");
            if (x.N != n)
                throw new ArgumentException("QRCP.solveInPlace: x.N must equal A_to_Q.N_Cols");

            var R = new floatMxN(n, n, Allocator.Temp, false);
            var P = new Pivot(n, Allocator.Temp);
            var u = new floatN(m, Allocator.Temp, false);
            var info = solveInPlace(ref A_to_Q, ref b, ref x, ref R, ref P, ref u, relTol);
            u.Dispose();
            P.Dispose();
            R.Dispose();
            return info;
        }

        /// <summary>
        /// Allocating convenience wrapper with default tolerance (max(m,n) * Consts.floatZeroThreshold,
        /// matching SVD.pinvSolve / MatrixMetrics.rank).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref floatMxN A_to_Q, ref floatN b, ref floatN x)
        {
            return solveInPlace(ref A_to_Q, ref b, ref x, (float)(-1));
        }
    }
}
