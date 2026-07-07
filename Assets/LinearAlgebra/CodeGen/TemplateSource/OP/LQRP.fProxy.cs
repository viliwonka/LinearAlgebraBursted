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
    /// Row-pivoted (rank-revealing) LQ — the transpose-dual of <see cref="QRCP"/>. Factorizes
    /// P·A = L·Q, where the ROW permutation P is chosen greedily so the pivot at each step is the
    /// trailing row of largest 2-norm. This forces the magnitudes of the L diagonal to be
    /// non-increasing (|L[0,0]| ≥ |L[1,1]| ≥ … ≥ |L[m-1,m-1]|), so trailing near-zero diagonal
    /// entries reveal the numerical rank — the stable choice for rank-deficient <em>underdetermined</em>
    /// (wide, m ≤ n) least squares, where the plain (un-pivoted) <see cref="LQ"/>.minNormSolve requires
    /// full ROW rank.
    /// </summary>
    /// <remarks>
    /// Built natively on LQ's row-Householder kernels (genHouseholderRow / applyHouseholderRight /
    /// applyQtFromReflectors, marked internal for this reason), NOT by transposing A and calling QRCP —
    /// same no-transpose choice LQ itself made over transpose-to-QR (see docs/dev/perf-vectorization-lessons.md).
    /// An UNBLOCKED per-reflector core, but with the partial ROW norms DOWNDATED (LAPACK
    /// dgeqp3/dlaqps-style, guarded, transposed to rows — see lqrpKernel) rather than recomputed
    /// exactly at every step: pivot selection needs the current row NORMS, not the current row DATA,
    /// so tracking them incrementally removes the second O(m²n) pass the original exact-recompute cut
    /// spent re-summing candidate norms. QRCP's LEVEL-3 machinery — the blocked dlaqps panel core with
    /// its deferred F-matrix trailing update — is deliberately NOT mirrored here yet: it only earns its
    /// bookkeeping at large sizes, and the primary consumer (rank-deficient IK Jacobians) is small
    /// (task DOF × joint DOF). Growing into a blocked core later mirrors exactly how QRCP itself
    /// evolved (unblocked+downdated first, then blocked).
    ///
    /// Two rank-safe solves, exactly mirroring <see cref="QRCP"/> on the tall side. solveInPlace gives
    /// the BASIC solution (dependent rows dropped, w[r..] = 0). For a rank-deficient A that is the
    /// minimum-norm solution ONLY when b is CONSISTENT (b ∈ range A): then the dropped equations are
    /// automatically satisfied. For an INCONSISTENT (genuine least-squares) rank-deficient b it is NOT
    /// minimum-norm — the below-diagonal block L21 couples the independent variables into the dropped
    /// equations, the transpose-dual of QRCP's R12 coupling (NOTE: it is L21 that couples, NOT a
    /// top-right block — L's top-right IS zero, but that is not where the coupling lives; the trailing
    /// rows of L keep their full norm, only the trailing DIAGONAL is small). minNormSolveInPlace closes
    /// that gap: it returns the pseudoinverse solution x = A⁺b (= SVD.pinvSolve) at direct cost, by
    /// least-squares-solving the coupled m×r block K = [L11; L21] instead of just the top L11. So both
    /// classes need a COD completion for the inconsistent rank-deficient case (there is no free lunch on
    /// the wide side after all). At full row rank both solves coincide with LQ.minNormSolve.
    /// </remarks>
    public static partial class LQRP {

        // Sum of squares of a matrix row over columns [colStart, n) — row-major, so the segment is
        // contiguous (unit stride). Four independent accumulators for ILP, mirroring LQ.dot4's rationale
        // (a single running-sum reduction can't be auto-vectorised under strict FloatMode). Feeds the
        // one-time norm initialisation and the guard-triggered exact re-sum; pivot selection itself
        // reads the incrementally DOWNDATED vn1 rather than calling this per candidate.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe fProxy rowSumSq(ref fProxyMxN W, int row, int colStart, int n)
        {
            fProxy* p = W.Data.Ptr + (long)row * n + colStart;
            int len = n - colStart;

            fProxy s0 = (fProxy)0, s1 = (fProxy)0, s2 = (fProxy)0, s3 = (fProxy)0;
            int i = 0;
            for (; i + 4 <= len; i += 4)
            {
                s0 += p[i] * p[i];
                s1 += p[i + 1] * p[i + 1];
                s2 += p[i + 2] * p[i + 2];
                s3 += p[i + 3] * p[i + 3];
            }
            fProxy s = (s0 + s1) + (s2 + s3);
            for (; i < len; i++)
                s += p[i] * p[i];
            return s;
        }

        // ---- shared factorization kernel (row-pivoted, unblocked, DOWNDATED norms) ----

        // Reduces the working copy W (m x n, already holding a copy of A — or A itself, in the
        // destructive solve) to L (m x m lower-triangular) via m row-Householder reflectors WITH row
        // pivoting recorded in P, then optionally reconstructs Q (m x n, orthonormal rows) from those
        // reflectors. Structurally LQ.lqKernel plus a pivot step; the transpose-dual of QRCP's
        // (column-pivoted) unblocked decompInPlaceCore, norm-downdating included.
        //
        // Per step d (d = 0..m-1):
        //   pivot: among trailing ROWS [d, m), pick the one whose tracked partial norm (vn1 — downdated
        //     or exact) over columns [d, n) is largest, swap it to position d (full-row swap — rows < d
        //     hold finished L entries in columns < d that must travel with the row; stored reflectors of
        //     earlier steps live in rows < d and are untouched, both indices >= d), and record the swap
        //     in P (vn1/vn2 swap with it). The tie guard leaves numerically-tied rows in place (see
        //     QRCP's decompInPlaceCore for the same reasoning transposed).
        //   reduce: build the reflector from row d (columns [d, n)), apply it from the right to
        //     W[d:, d:], capture L[d,d], then stash the reflector into W[d, d..n-1] for reuse.
        //   downdate: a row-Householder is orthogonal over columns [d, n), so it preserves each trailing
        //     row's norm over that window exactly; advancing the window to [d+1, n) for the NEXT pivot
        //     therefore subtracts only the column-d entry W[i,d] (== the finished L[i,d]):
        //         ‖row_i‖(cols d+1..) = sqrt( vn1[i]² − W[i,d]² ).
        //     Guarded exactly as LAPACK dgeqp3/dlaqps (transposed): vn1[i] tracks the current estimate,
        //     vn2[i] the norm at the last EXACT re-sum; once the cumulative decay collapses below
        //     sqrt(eps) an exact re-sum of the trailing rows forces both back in sync. Re-summing a
        //     single row is a CONTIGUOUS (unit-stride) segment here, so — unlike QRCP's strided
        //     per-column case — the guard trip just re-sums every trailing row via rowSumSq.
        // |L[d,d]| equals the pivot's trailing row norm, so pivoting makes |L[0,0]| >= |L[1,1]| >= ...,
        // which is what exposes the numerical rank to solveInPlace. (The reflector — hence L[d,d] — is
        // built from the EXACT row data; only the pivot CHOICE reads the downdated vn1.)
        //
        // reconstructQ == false (the solve's factor-only path): stop after the forward sweep and L
        // extraction, leaving the reflectors in W's rows for applyQtFromReflectors; Q may be default.
        // The row permutation is NOT applied to any RHS here — solveInPlace gathers P·b afterwards from
        // the finished P, keeping this kernel shared byte-for-byte between decomp and solve.
        static unsafe void lqrpKernel(ref fProxyMxN W, ref fProxyMxN L, ref fProxyMxN Q, ref Pivot P, ref fProxyN v,
                                      fProxy zeroThreshold, bool reconstructQ)
        {
            int m = W.M_Rows;
            int n = W.N_Cols;

            P.Reset();

            // Downdating scratch (vn1/vn2, length m — one partial norm per ROW). Allocated Temp inside
            // the kernel (O(m) « O(m²n)), mirroring how QRCP.decompInPlaceCore allocates its `w`: keeps
            // every public overload's signature and the LQRP cache unchanged, and both the allocating
            // and cache decomp paths run this identical kernel (so their results stay bit-identical).
            // vn1[i] tracks row i's current partial 2-norm over the active trailing columns; vn2[i] the
            // norm at its last EXACT (re-summed) value.
            var vn1 = new fProxyN(m, Allocator.Temp, false);
            var vn2 = new fProxyN(m, Allocator.Temp, false);
            fProxy* vn1p = vn1.Data.Ptr;
            fProxy* vn2p = vn2.Data.Ptr;

            // Initial partial norms == exact full ROW norms (cols [0, n)); vn2 == vn1 (no decay yet).
            for (int i = 0; i < m; i++)
            {
                fProxy nrm = math.sqrt(rowSumSq(ref W, i, 0, n));
                vn1p[i] = nrm;
                vn2p[i] = nrm;
            }

            // Relative tie tolerance for the (now UNSQUARED) row-norm pivot compare — dual of QRCP's
            // (8*m)*eps expressed for unsquared norms (sqrt(1+·)); the row-norm reduction runs over up
            // to n columns, so the length bound is n here. Hoisted (n is fixed for the whole call).
            fProxy pivotRelTol = (fProxy)(8 * n) * Consts.fProxyEpsilon;
            fProxy pivotRelTolRoot = math.sqrt((fProxy)1 + pivotRelTol);

            for (int d = 0; d < m; d++)
            {
                // --- row pivot: largest tracked partial norm (vn1) among trailing rows [d, m) ---
                fProxy diagNorm1 = vn1p[d];
                int pivotRow = d;
                fProxy maxNorm1 = diagNorm1;
                for (int i = d + 1; i < m; i++)
                {
                    if (vn1p[i] > maxNorm1)
                    {
                        maxNorm1 = vn1p[i];
                        pivotRow = i;
                    }
                }

                // Only pivot when the best row beats the incumbent past the accumulated rounding noise —
                // leaves numerically-tied rows in place (bare '>' would let ~1 ulp induce a spurious,
                // non-reproducible permutation). Unsquared compare (see pivotRelTolRoot above).
                if (pivotRow != d && maxNorm1 > diagNorm1 * pivotRelTolRoot)
                {
                    Swap.Rows(ref W, d, pivotRow);   // full row: carries finished L columns < d too
                    P.Swap(d, pivotRow);
                    fProxy tv1 = vn1p[d]; vn1p[d] = vn1p[pivotRow]; vn1p[pivotRow] = tv1;
                    fProxy tv2 = vn2p[d]; vn2p[d] = vn2p[pivotRow]; vn2p[pivotRow] = tv2;
                }

                LQ.genHouseholderRow(ref W, ref v, d, d, zeroThreshold);
                LQ.applyHouseholderRight(ref W, ref v, d, d);

                L[d, d] = W[d, d];
                for (int c = d; c < n; c++)
                    W[d, c] = v[c];

                // --- downdate trailing row norms for the NEXT step (guarded; see the method doc). The
                //     column-d entry W[i,d] is the finished L[i,d] — read it BEFORE it is later swapped
                //     around as part of full rows (it stays put here; only future steps swap it). ---
                bool anyExact = false;
                for (int i = d + 1; i < m; i++)
                {
                    fProxy v1 = vn1p[i];
                    if (v1 <= (fProxy)0)
                        continue; // already (exactly) zero -- stays zero, no decay possible

                    fProxy v2 = vn2p[i];
                    if (v2 <= (fProxy)0)
                    {
                        anyExact = true; // defensive; see QRCP.decompInPlaceCore
                        continue;
                    }

                    fProxy ratio = math.abs(W[i, d]) / v1;
                    // (1+ratio)*(1-ratio) instead of 1-ratio² — same value, but avoids cancellation
                    // when ratio ≈ 1 (matches LAPACK's own formulation).
                    fProxy temp = math.max((fProxy)0, ((fProxy)1 + ratio) * ((fProxy)1 - ratio));
                    fProxy decaySinceExact = v1 / v2;
                    fProxy temp2 = temp * decaySinceExact * decaySinceExact;

                    if (temp2 <= Consts.fProxySqrtEps)
                        anyExact = true; // cumulative decay too large to trust — re-sum below
                    else
                        vn1p[i] = v1 * math.sqrt(temp);
                }

                if (anyExact)
                {
                    // Exact re-sum of every trailing row [d+1, m) over cols [d+1, n) — contiguous per
                    // row (unit stride), so cheaper than QRCP's strided per-column re-sum. Re-sum ALL
                    // trailing rows (not just the tripped ones): one batched sweep, strictly more
                    // accurate for the rows that didn't strictly need it.
                    for (int i = d + 1; i < m; i++)
                    {
                        fProxy nrm = math.sqrt(rowSumSq(ref W, i, d + 1, n));
                        vn1p[i] = nrm;
                        vn2p[i] = nrm;
                    }
                }
            }

            vn2.Dispose();
            vn1.Dispose();

            // L extraction (below-diagonal from W's untouched lower triangle; above-diagonal zero).
            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < r; c++)
                    L[r, c] = W[r, c];
                for (int c = r + 1; c < m; c++)
                    L[r, c] = (fProxy)0;
            }

            if (!reconstructQ)
                return;

            // Reconstruct Q from the stored reflectors (backward pass, identical to LQ.lqKernel): seed
            // Q = [I_m | 0], apply H_{m-1} … H_0 from the right. Row pivoting only reordered W's rows,
            // so the reflectors — and hence Q's rows — already sit in the permuted order (P·A = L·Q).
            unsafe { UnsafeUtility.MemClear(Q.Data.Ptr, (long)m * n * UnsafeUtility.SizeOf<fProxy>()); }
            for (int i = 0; i < m; i++)
                Q[i, i] = (fProxy)1;

            for (int d = m - 1; d >= 0; d--)
            {
                for (int c = d; c < n; c++)
                    v[c] = W[d, c];
                LQ.applyHouseholderRight(ref Q, ref v, d, d);
            }
        }

        // ---- decomposition: A-preserving, produces L, Q, P with P·A = L·Q ----

        /// <summary>
        /// Row-pivoted (rank-revealing) LQ of A (m × n, m ≤ n): P·A = L·Q, where P is a row permutation
        /// (see <paramref name="P"/>), L is m × m lower-triangular with non-increasing |diagonal|
        /// (revealing the numerical rank), and Q is m × n with orthonormal rows (Q Qᵀ = I_m). A is not
        /// modified. Allocates Allocator.Temp scratch internally.
        /// Always reports DirectSolveStatus.Success — the factorization has no failure mode (read the
        /// numerical rank off L's diagonal, or use solveInPlace which reports it as <see cref="RankInfo"/>).
        /// </summary>
        /// <param name="A">Input m × n matrix (m ≤ n). Not modified.</param>
        /// <param name="L">Output m × m lower-triangular factor (caller-allocated, m × m).</param>
        /// <param name="Q">Output m × n row-orthonormal factor (caller-allocated, m × n).</param>
        /// <param name="P">Output row permutation of size m (reset internally): (P·A)[j, :] == A[P[j], :],
        /// equivalently A[P[j], :] == (L·Q)[j, :].</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in fProxyMxN A, ref fProxyMxN L, ref fProxyMxN Q, ref Pivot P)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQRP.decomp: A must be wide or square (M_Rows <= N_Cols)");
            if (L.M_Rows != m || L.N_Cols != m)
                throw new ArgumentException("LQRP.decomp: L must be m x m");
            if (Q.M_Rows != m || Q.N_Cols != n)
                throw new ArgumentException("LQRP.decomp: Q must be m x n");
            if (P.N != m)
                throw new ArgumentException("LQRP.decomp: P.N must equal A.M_Rows");

            if (m == 0 || n == 0)
                return new DirectSolveInfo { status = DirectSolveStatus.Success };

            var W = new fProxyMxN(m, n, Allocator.Temp, false);
            var v = new fProxyN(n, Allocator.Temp, false);

            W.Data.CopyFrom(A.Data);
            fProxy zeroThreshold = Consts.fProxyZeroThreshold * Norms.LInf(in A);

            lqrpKernel(ref W, ref L, ref Q, ref P, ref v, zeroThreshold, reconstructQ: true);

            v.Dispose();
            W.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// decomp using a reusable workspace (Arena.fProxyLQRPCache(m, n)) — zero-alloc.
        /// Semantics identical to the allocating overload; see that one for full documentation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in fProxyMxN A, ref fProxyMxN L, ref fProxyMxN Q, ref Pivot P, ref fProxyLQRPCache ws)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQRP.decomp: A must be wide or square (M_Rows <= N_Cols)");
            if (L.M_Rows != m || L.N_Cols != m)
                throw new ArgumentException("LQRP.decomp: L must be m x m");
            if (Q.M_Rows != m || Q.N_Cols != n)
                throw new ArgumentException("LQRP.decomp: Q must be m x n");
            if (P.N != m)
                throw new ArgumentException("LQRP.decomp: P.N must equal A.M_Rows");
            RequireLQRPWorkspace(in ws, m, n);

            if (m == 0 || n == 0)
                return new DirectSolveInfo { status = DirectSolveStatus.Success };

            var W = ws.W;
            var v = ws.v;

            W.Data.CopyFrom(A.Data);
            fProxy zeroThreshold = Consts.fProxyZeroThreshold * Norms.LInf(in A);

            lqrpKernel(ref W, ref L, ref Q, ref P, ref v, zeroThreshold, reconstructQ: true);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // ---- rank-safe basic solve for underdetermined systems ----

        /// <summary>
        /// LQRP-based rank-safe solve of the underdetermined system A x = b (m ≤ n): BASIC (truncated)
        /// solution for a possibly rank-deficient A (minimum-norm when b is consistent — see the class
        /// remarks; for an inconsistent rank-deficient LS use minNormSolveInPlace). Row-pivoted LQ
        /// (P·A = L·Q) exposes the numerical
        /// row rank r: the L diagonal is non-increasing, so r = count of leading entries with
        /// |L[i,i]| &gt; tol, where tol = relTol · |L[0,0]| and relTol defaults to
        /// max(m,n) · Consts.fProxyZeroThreshold (matching SVD.pinvSolve / MatrixMetrics.rank). A
        /// negative relTol is an "auto" sentinel that selects that same default.
        ///
        /// The reduced system L w = P·b is forward-solved on its leading r×r block (divide-safe by
        /// construction: every used L diagonal exceeds tol); the remaining (m-r) dependent equations are
        /// dropped (w[r..] = 0), then x = Qᵀ w. This is the BASIC solution: it satisfies the r independent
        /// equations. When b is CONSISTENT it is also the minimum-norm solution (the dropped equations are
        /// automatically satisfied); when b is INCONSISTENT (a genuine least-squares problem) it is NOT —
        /// the below-diagonal L21 block couples the leading variables into the dropped equations, so use
        /// minNormSolveInPlace (COD) for x = A⁺b there. At full row rank (r == m) the result is identical
        /// to LQ.minNormSolve (which IS min-norm).
        ///
        /// DESTRUCTIVE FAST PATH (like QR/QRCP.solveInPlace): factors A's own buffer in place (no memcpy,
        /// no separate Q) and applies Qᵀ straight from the stored reflectors, never forming Q. On return
        /// A is DESTROYED (stored reflectors + L's sub-diagonal, contents undefined); b is NOT modified.
        /// Need the factors? Use LQRP.decomp instead, which preserves A and reconstructs Q.
        /// </summary>
        /// <param name="A">On entry A (m × n, m ≤ n); DESTROYED on exit (NOT the factor — use decomp for L/Q).</param>
        /// <param name="b">Right-hand side, length m. Read-only (not modified). Must not alias x.</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true. Solution, length n.</param>
        /// <param name="L">Scratch: m × m (receives the lower-triangular factor; consumed).</param>
        /// <param name="P">Scratch: row Pivot of size m (reset internally).</param>
        /// <param name="v">Scratch: length EXACTLY n (Householder + reduced-RHS workspace).</param>
        /// <param name="relTol">Rank threshold ratio; tol = relTol · |L[0,0]|. Negative = auto default.</param>
        /// <returns>Status Success (r == m, full row rank) or RankDeficient (r &lt; m, still a usable
        /// basic solution); rank = detected r. See <see cref="RankInfo.Solved"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                            ref fProxyMxN L, ref Pivot P, ref fProxyN v, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQRP.solveInPlace: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQRP.solveInPlace: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQRP.solveInPlace: x.N must equal A.N_Cols");
            if (L.M_Rows != m || L.N_Cols != m)
                throw new ArgumentException("LQRP.solveInPlace: L must be m x m");
            if (P.N != m)
                throw new ArgumentException("LQRP.solveInPlace: P.N must equal A.M_Rows");
            if (v.N != n)
                throw new ArgumentException("LQRP.solveInPlace: v.N must equal A.N_Cols");

            // Negative relTol → library-standard rank threshold (same default as SVD.pinvSolve /
            // MatrixMetrics.rank). Also keeps tol >= 0, so a stray negative can't inflate rank.
            if (relTol < (fProxy)0)
                relTol = (fProxy)(math.max(m, n)) * Consts.fProxyZeroThreshold;

            if (n == 0)
                return new RankInfo { status = DirectSolveStatus.Success, rank = 0 };

            fProxy zeroThreshold = Consts.fProxyZeroThreshold * Norms.LInf(in A);

            // Factor A in place (destroyed): reflectors in A's rows + L (separate), row pivot in P; no Q.
            var Qnull = default(fProxyMxN);
            lqrpKernel(ref A, ref L, ref Qnull, ref P, ref v, zeroThreshold, reconstructQ: false);

            // --- rank from L's non-increasing |diagonal| (tol = relTol·|L[0,0]|; NaN/zero L[0,0] → 0) ---
            fProxy tol = relTol * math.abs(L[0, 0]);
            int rank = 0;
            for (int i = 0; i < m; i++)
            {
                if (math.abs(L[i, i]) > tol)
                    rank++;
                else
                    break;
            }

            if (rank == 0)
            {
                for (int j = 0; j < n; j++)
                    x[j] = (fProxy)0;
                return new RankInfo { status = DirectSolveStatus.RankDeficient, rank = 0 };
            }

            int r = rank;

            // --- reduced RHS c = P·b, gathered into v[0..m-1] (v is free scratch after factorization;
            //     length n >= m). c[j] = b[P[j]]. ---
            for (int j = 0; j < m; j++)
                v[j] = b[P[j]];

            // --- basic forward-solve of L w = c on the leading r×r block (in place in v[0..m-1]);
            //     dependent rows dropped (w[r..m-1] = 0). Every used L[i,i] exceeds tol (divide-safe). ---
            for (int i = 0; i < r; i++)
            {
                fProxy sum = (fProxy)0;
                for (int j = 0; j < i; j++)
                    sum += L[i, j] * v[j];
                v[i] = (v[i] - sum) / L[i, i];
            }
            for (int i = r; i < m; i++)
                v[i] = (fProxy)0;

            // --- x = Qᵀ w, applied straight from A's stored reflectors (no dense Q). Column ordering is
            //     untouched (only ROWS were permuted), so x needs no un-permute. ---
            LQ.applyQtFromReflectors(ref A, ref v, ref x);

            return new RankInfo
            {
                status = (r < m) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                rank = r
            };
        }

        /// <summary>
        /// solveInPlace with the default rank tolerance (max(m,n) · Consts.fProxyZeroThreshold,
        /// matching SVD.pinvSolve / MatrixMetrics.rank). See the primitive for full semantics.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                            ref fProxyMxN L, ref Pivot P, ref fProxyN v)
        {
            return solveInPlace(ref A, ref b, ref x, ref L, ref P, ref v, (fProxy)(-1));
        }

        /// <summary>
        /// Allocating convenience wrapper: allocates L (m×m), P (m-Pivot) and v (n) from Allocator.Temp
        /// and delegates to the zero-alloc primitive. DESTROYS A (see the primitive). Use the primitive
        /// in hot loops to avoid repeated Temp allocs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // Replicate the primitive's dimension checks BEFORE allocating, so a caller error can't leak
            // the Temp allocations (validate-before-alloc).
            if (m > n)
                throw new ArgumentException("LQRP.solveInPlace: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQRP.solveInPlace: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQRP.solveInPlace: x.N must equal A.N_Cols");

            var L = new fProxyMxN(m, m, Allocator.Temp, false);
            var P = new Pivot(m, Allocator.Temp);
            var v = new fProxyN(n, Allocator.Temp, false);
            var info = solveInPlace(ref A, ref b, ref x, ref L, ref P, ref v, relTol);
            v.Dispose();
            P.Dispose();
            L.Dispose();
            return info;
        }

        /// <summary>
        /// Allocating convenience wrapper with default tolerance (max(m,n) · Consts.fProxyZeroThreshold).
        /// DESTROYS A.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x)
        {
            return solveInPlace(ref A, ref b, ref x, (fProxy)(-1));
        }

        // ---- COD (complete orthogonal decomposition): MINIMUM-NORM rank-safe least squares ----
        //
        // solveInPlace above gives the BASIC solution (dependent rows dropped, w[r..] = 0). For a
        // rank-deficient wide A that is the minimum-norm solution ONLY when b is CONSISTENT; for an
        // inconsistent (genuine least-squares) b it is NOT — the below-diagonal block L21 couples the
        // independent variables into the dependent (dropped) equations, exactly the transpose-dual of
        // QRCP's R12 coupling. minNormSolveInPlace closes that gap and returns x = A⁺b (the pseudoinverse
        // / minimum-norm least-squares solution — the SAME answer SVD.pinvSolve gives) at
        // direct-factorization cost, mirroring QRCP.minNormSolveInPlace on the tall side.
        //
        // Derivation. P·A = L·Q with L = [L11 0; L21 L22] (L11 r×r lower-tri, full rank; the trailing
        // DIAGONAL below tol, so L22 ≈ 0 — but L21 is NOT small). Writing g = Q·x ∈ ℝ^m and c = P·b, the
        // residual is ‖L·g − c‖² = ‖L11·g1 − c1‖² + ‖L21·g1 + L22·g2 − c2‖². With L22 ≈ 0 the free block
        // g2 cannot reduce the residual, so it is fixed to 0 for minimum ‖x‖ (= ‖g‖, Q having orthonormal
        // rows); g1 minimizes ‖K·g1 − c‖ over BOTH row blocks at once, where K = [L11; L21] is the first
        // r columns of L (m×r, full COLUMN rank r). The basic solve instead takes g1 = L11⁻¹·c1 from the
        // top block only — which coincides with the least-squares g1 iff c is consistent (then L21·g1 = c2
        // automatically). So the whole solve is:
        //   1. factor P·A = L·Q in place (reflectors left in A; Q never formed), read rank r off L.
        //   2. c = P·b; g1 = argmin ‖K·g1 − c‖ via ordinary QR least-squares on K = L[:, 0..r).
        //   3. x = Qᵀ·[g1 ; 0] straight from A's stored reflectors (dependent block zeroed = min-norm).
        // Reuses QR.solveInPlace (the tall LS solve) and LQ.applyQtFromReflectors — no new kernel.

        // COD completion (r >= 1). On entry A holds the row-reflectors from lqrpKernel(reconstructQ:false),
        // L the lower-triangular factor, P the row pivot, r the detected rank. Fills x with the
        // minimum-norm least-squares solution. b is read-only (gathered into c = P·b). Allocates one set
        // of Allocator.Temp scratch (K/c/g1/w); the QR-LS on the m×r block is O(mr²), negligible against
        // the O(m²n) LQRP factorization that produced L.
        static unsafe void minNormCodFinish(ref fProxyMxN A, in fProxyN b, ref fProxyN x,
                                            ref fProxyMxN L, in Pivot P, int r)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // K = first r columns of L (m×r). K is full column rank r (L11's diagonal all exceed tol), so
            // the QR least-squares below is well-posed. c = P·b (reduced RHS, length m).
            var K = new fProxyMxN(m, r, Allocator.Temp, false);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < r; j++)
                    K[i, j] = L[i, j];
            var c = new fProxyN(m, Allocator.Temp, false);
            for (int i = 0; i < m; i++)
                c[i] = b[P[i]];

            // g1 = argmin ‖K·g1 − c‖ (ordinary QR least-squares). QR.solveInPlace DESTROYS K and c —
            // both are private Temp copies.
            var g1 = new fProxyN(r, Allocator.Temp, false);
            QR.solveInPlace(ref K, ref c, ref g1);

            // w = [g1 ; 0] (length n; applyQtFromReflectors reads w[0..m) = [g1 ; 0_{m-r}] — the dependent
            // block zeroed for the minimum-norm choice). x = Qᵀ·w straight from A's reflectors (no dense
            // Q). Columns were never permuted (only rows), so x needs no un-permute.
            var w = new fProxyN(n, Allocator.Temp, false);
            for (int i = 0; i < r; i++) w[i] = g1[i];
            for (int i = r; i < n; i++) w[i] = (fProxy)0;
            LQ.applyQtFromReflectors(ref A, ref w, ref x);

            w.Dispose();
            g1.Dispose();
            c.Dispose();
            K.Dispose();
        }

        /// <summary>
        /// LQRP-based MINIMUM-NORM rank-safe least-squares (complete orthogonal decomposition). Solves
        /// the underdetermined A x ≈ b (m ≤ n) for a possibly rank-deficient A and returns the
        /// pseudoinverse solution x = A⁺b — the minimum-2-norm vector among all least-squares minimizers,
        /// the SAME result SVD.pinvSolve gives, but at direct-factorization cost (one row-pivoted LQ plus
        /// one small QR on the m×r rank-revealed block). For a CONSISTENT b this coincides with the basic
        /// <see cref="solveInPlace(ref fProxyMxN, ref fProxyN, ref fProxyN, ref fProxyMxN, ref Pivot, ref fProxyN, fProxy)"/>;
        /// the two differ only for an INCONSISTENT (genuine least-squares) rank-deficient system, where
        /// the basic solution is not minimum-norm (see the class remarks / the L21 coupling). At full row
        /// rank (r == m) both coincide with LQ.minNormSolve.
        ///
        /// DESTRUCTIVE FAST PATH (like solveInPlace): factors A's own buffer in place (reflectors, no Q).
        /// On return A is DESTROYED (stored reflectors + L's sub-diagonal); b is NOT modified. Numerical
        /// rank r is read from L's non-increasing diagonal (tol = relTol·|L[0,0]|; negative relTol
        /// auto-selects max(m,n)·Consts.fProxyZeroThreshold, matching SVD.pinvSolve / MatrixMetrics.rank).
        /// </summary>
        /// <param name="A">On entry A (m × n, m ≤ n); DESTROYED on exit (reflectors, NOT a factor — use decomp for L/Q).</param>
        /// <param name="b">Right-hand side, length m. Read-only (not modified). Must not alias x.</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true. Minimum-norm solution, length n.</param>
        /// <param name="L">Scratch: m × m (receives the lower-triangular factor; consumed).</param>
        /// <param name="P">Scratch: row Pivot of size m (reset internally).</param>
        /// <param name="v">Scratch: length EXACTLY n (Householder workspace).</param>
        /// <param name="relTol">Rank threshold ratio; tol = relTol·|L[0,0]|. Negative = auto default.</param>
        /// <returns>Status Success (r == m) or RankDeficient (r &lt; m, still a usable minimum-norm
        /// solution); rank = detected r. See <see cref="RankInfo.Solved"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo minNormSolveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                                   ref fProxyMxN L, ref Pivot P, ref fProxyN v, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQRP.minNormSolveInPlace: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQRP.minNormSolveInPlace: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQRP.minNormSolveInPlace: x.N must equal A.N_Cols");
            if (L.M_Rows != m || L.N_Cols != m)
                throw new ArgumentException("LQRP.minNormSolveInPlace: L must be m x m");
            if (P.N != m)
                throw new ArgumentException("LQRP.minNormSolveInPlace: P.N must equal A.M_Rows");
            if (v.N != n)
                throw new ArgumentException("LQRP.minNormSolveInPlace: v.N must equal A.N_Cols");

            if (relTol < (fProxy)0)
                relTol = (fProxy)(math.max(m, n)) * Consts.fProxyZeroThreshold;

            if (n == 0)
                return new RankInfo { status = DirectSolveStatus.Success, rank = 0 };

            fProxy zeroThreshold = Consts.fProxyZeroThreshold * Norms.LInf(in A);

            // Factor A in place (destroyed): reflectors in A's rows + L (separate), row pivot in P; no Q.
            var Qnull = default(fProxyMxN);
            lqrpKernel(ref A, ref L, ref Qnull, ref P, ref v, zeroThreshold, reconstructQ: false);

            // Rank from L's non-increasing |diagonal| (tol = relTol·|L[0,0]|; NaN/zero L[0,0] → 0).
            fProxy tol = relTol * math.abs(L[0, 0]);
            int r = 0;
            for (int i = 0; i < m; i++)
            {
                if (math.abs(L[i, i]) > tol) r++;
                else break;
            }

            if (r == 0)
            {
                for (int j = 0; j < n; j++) x[j] = (fProxy)0;
                return new RankInfo { status = DirectSolveStatus.RankDeficient, rank = 0 };
            }

            unsafe { minNormCodFinish(ref A, in b, ref x, ref L, in P, r); }

            return new RankInfo
            {
                status = (r < m) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                rank = r
            };
        }

        /// <summary>minNormSolveInPlace with the default rank tolerance (max(m,n)·Consts.fProxyZeroThreshold,
        /// matching SVD.pinvSolve / MatrixMetrics.rank). See the primitive for full semantics.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo minNormSolveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x,
                                                   ref fProxyMxN L, ref Pivot P, ref fProxyN v)
        {
            return minNormSolveInPlace(ref A, ref b, ref x, ref L, ref P, ref v, (fProxy)(-1));
        }

        /// <summary>
        /// Allocating convenience wrapper: allocates L (m×m), P (m-Pivot) and v (n) from Allocator.Temp
        /// and delegates to the zero-alloc primitive. DESTROYS A (b is preserved). Use the primitive in
        /// hot loops to avoid repeated Temp allocs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo minNormSolveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x, fProxy relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // Replicate the primitive's dimension checks BEFORE allocating (validate-before-alloc).
            if (m > n)
                throw new ArgumentException("LQRP.minNormSolveInPlace: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQRP.minNormSolveInPlace: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQRP.minNormSolveInPlace: x.N must equal A.N_Cols");

            var L = new fProxyMxN(m, m, Allocator.Temp, false);
            var P = new Pivot(m, Allocator.Temp);
            var v = new fProxyN(n, Allocator.Temp, false);
            var info = minNormSolveInPlace(ref A, ref b, ref x, ref L, ref P, ref v, relTol);
            v.Dispose();
            P.Dispose();
            L.Dispose();
            return info;
        }

        /// <summary>Allocating convenience wrapper with default tolerance (max(m,n)·Consts.fProxyZeroThreshold).
        /// DESTROYS A (b preserved).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo minNormSolveInPlace(ref fProxyMxN A, ref fProxyN b, ref fProxyN x)
        {
            return minNormSolveInPlace(ref A, ref b, ref x, (fProxy)(-1));
        }
    }
}
