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
    /// Partial column norms are DOWNDATED (LAPACK dgeqp3/dlaqps-style, unsquared, guarded) rather
    /// than recomputed exactly at every step — see <see cref="floatQRCPCache"/> and
    /// decompInPlaceCore. That downdating is what unlocks a level-3 path: pivot selection needs the
    /// current column NORMS, not the current column DATA, so once N_Cols >= 2*QRCP_BLOCK the
    /// factorization runs the LAPACK dlaqps-style partially-blocked panel core
    /// (decompInPlaceBlockedCore) — a whole panel of reflectors is factored against a deferred
    /// F-matrix and its trailing update flushed once as a rank-kb GEMM, and Q is reconstructed by the
    /// same blocked-WY kernel QR uses (QR.reconstructQBlocked). Below that gate the unblocked
    /// per-reflector core runs (decompCoreDispatch chooses). <see cref="floatQRCPCache"/> still
    /// carries only the two n-sized downdating vectors (vn1, vn2); the blocked core's larger working
    /// buffers (F, the flush GEMM scratch, and the reconstruction WY buffers) are Allocator.Temp
    /// allocated per call inside decompInPlaceBlockedCore — one set per factorization, negligible
    /// against its O(n²m) work — rather than folded into the cache. Promoting them into the cache for
    /// a fully zero-alloc blocked path (as QR's cache does) is a candidate follow-up, not part of this.
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
        // Partial column norms are DOWNDATED rather than recomputed exactly at every step — see
        // decompInPlaceCore for the guarded LAPACK dgeqp3/dlaqps algorithm (norm values feed only the
        // pivot CHOICE; the reflector math itself is untouched).
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

            // Downdating scratch (vn1, vn2 — see decompInPlaceCore): 2n Allocator.Temp, allocated
            // AFTER every validation above so a caller error can't leak them (validate-before-alloc).
            var vn1 = new floatN(A_to_Q.N_Cols, Allocator.Temp, false);
            var vn2 = new floatN(A_to_Q.N_Cols, Allocator.Temp, false);
            var info = decompCoreDispatch(ref A_to_Q, ref R, ref P, ref u, ref vn1, ref vn2);
            vn2.Dispose();
            vn1.Dispose();
            return info;
        }

        /// <summary>
        /// Zero-alloc cache overload: routes through the SAME downdating core as every other overload
        /// (see decompInPlaceCore), using caller-owned <see cref="floatQRCPCache"/> scratch (vn1/vn2)
        /// instead of a per-call Allocator.Temp pair — bit-identical results, see Arena.floatQRCPCache.
        /// u is still caller-provided separately (not folded into the cache — see the class remarks).
        /// </summary>
        /// <remarks>R must not alias A_to_Q (unchecked) — see the 4-arg overload.</remarks>
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref floatMxN A_to_Q, ref floatMxN R, ref Pivot P, ref floatN u, ref floatQRCPCache cache)
        {
            if (A_to_Q.M_Rows < A_to_Q.N_Cols)
                throw new ArgumentException("QRCP.decompInPlace: Matrix must be square or tall (M_Rows >= N_Cols)");

            if (u.N != A_to_Q.M_Rows)
                throw new ArgumentException("QRCP.decompInPlace: scratch vector u.N must equal A_to_Q.M_Rows");

            if (P.N != A_to_Q.N_Cols)
                throw new ArgumentException("QRCP.decompInPlace: pivot P.N must equal A_to_Q.N_Cols");

            if (R.M_Rows != A_to_Q.N_Cols || R.N_Cols != A_to_Q.N_Cols)
                throw new ArgumentException("QRCP.decompInPlace: R must be N_Cols x N_Cols");

            RequireQRCPWorkspace(in cache, A_to_Q.N_Cols);

            return decompCoreDispatch(ref A_to_Q, ref R, ref P, ref u, ref cache.vn1, ref cache.vn2);
        }

        // ---- shared core: every decompInPlace/decomp/solveInPlace overload routes through this ----

        // Guarded LAPACK dgeqp3/dlaqps-style norm downdating (transcribed unsquared, per
        // docs/spec-qrcp-downdate.md). Householder reflectors preserve a column's norm over rows
        // d..m-1 exactly (orthogonal transform restricted to that row range), so the norm over rows
        // d+1..m-1 differs from the PREVIOUS step's tracked norm only by the row-d entry the
        // reflector apply just wrote:
        //     ‖col_j‖²(d+1..) = vn1[j]² (rows d.., BEFORE this step) − A_to_Q[d,j]² (AFTER this step).
        // Per column, vn1 tracks the current estimate and vn2 the norm at the last EXACT computation;
        // once the cumulative decay since that last exact value collapses below tol3z = sqrt(eps) —
        // checked via vn2, not just this step's own ratio, since gradual decay over many benign-looking
        // steps is the failure LAPACK's guard exists for — an exact re-sum forces both back in sync.
        // tol3z is Consts.floatSqrtEps directly: Consts.cs already defines it as the precise,
        // type-correct sqrt(Consts.floatEpsilon) (see its own comment), and every other caller in
        // this codebase (Eigen/LOBPCG/Solvers/SVD.LowRank) references it the same way rather than
        // recomputing math.sqrt(Consts.floatEpsilon) at runtime.
        //
        // Guard-triggered exact re-sum recomputes EVERY trailing column (not just the one that
        // failed) via the same row-major addSquares sweep as the one-time init below, writing
        // straight into vn1 (no separate colNorm2 buffer is needed — the old exact-recompute-every-
        // step buffer is fully retired). This is a deliberate widening from LAPACK's own per-column
        // selective recompute: this codebase is row-major, so a single column's exact norm is a
        // strided reduction (the same shape the ORIGINAL always-exact QRCP avoided by summing all
        // trailing columns per row instead of one column at a time) — reusing that same batched sweep
        // when ANY column trips the guard is simpler, no more expensive (the sweep touches every
        // trailing column per row regardless of how many needed it), and strictly more accurate for
        // the columns that didn't strictly need re-summing.
        static DirectSolveInfo decompInPlaceCore(ref floatMxN A_to_Q, ref floatMxN R, ref Pivot P, ref floatN u,
                                                  ref floatN vn1, ref floatN vn2)
        {
            P.Reset();

            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            // Reflector-apply accumulator (length n) — see QR.applyReflectorRight. Allocated once per
            // call (O(n) « O(n²m)); this is unchanged from before this change (see the class remarks).
            var w = new floatN(n, Allocator.Temp, false);

            // scale-relative zero-column threshold (see QR.genHouseholder); LInf(Q) == max |entry|.
            float zeroThreshold = Consts.floatZeroThreshold * Norms.LInf(in A_to_Q);

            // Initial partial norms == exact FULL column norms (rows 0..m-1), computed once in a
            // single row-major sweep (unit-stride, vectorised addSquares — same restructuring as the
            // reflector apply). vn2 starts equal to vn1: no decay has happened yet.
            unsafe
            {
                float* qp = A_to_Q.Data.Ptr;
                float* vp = vn1.Data.Ptr;
                UnsafeUtility.MemClear(vp, (long)n * UnsafeUtility.SizeOf<float>());
                for (int r = 0; r < m; r++)
                    UnsafeOP.addSquares(vp, qp + (long)r * n, n);
            }
            for (int j = 0; j < n; j++)
            {
                float nrm = math.sqrt(vn1[j]);
                vn1[j] = nrm;
                vn2[j] = nrm;
            }

            // Same relative tie tolerance the exact-recompute kernel used, expressed for UNSQUARED
            // norms: the old test compared squared norms via maxNorm2 > diagNorm2*(1+pivotRelTol);
            // sqrt(1+pivotRelTol) is the equivalent unsquared-domain multiplier, so pivot selection
            // is unchanged whenever vn1 holds exact norms (true at d=0, and after any guard-triggered
            // re-sum) and separation-preserving otherwise — see docs/spec-qrcp-downdate.md OQ-D1.
            // m is fixed for the whole call, so this is hoisted out of the per-step loop.
            float pivotRelTol = (float)(8 * m) * Consts.floatEpsilon;
            float pivotRelTolRoot = math.sqrt((float)1 + pivotRelTol);

            for (int d = 0; d < n; d++)
            {
                // --- column pivot: among trailing columns d..n-1, pick the one whose tracked partial
                //     norm (vn1 — downdated or exact, see below) is largest. ---
                float diagNorm1 = vn1[d];
                int pivotCol = d;
                float maxNorm1 = diagNorm1;
                for (int c = d + 1; c < n; c++)
                {
                    if (vn1[c] > maxNorm1)
                    {
                        maxNorm1 = vn1[c];
                        pivotCol = c;
                    }
                }

                // Only pivot when the best column beats the incumbent by more than the accumulated
                // rounding noise of the norm tracking. This leaves numerically-tied columns in place —
                // notably the Kahan matrix, whose columns all have norm exactly 1 and which is provably
                // invariant under column pivoting; a bare `>` would let a ~1 ulp difference induce a
                // spurious (and non-reproducible) permutation.
                if (pivotCol != d && maxNorm1 > diagNorm1 * pivotRelTolRoot)
                {
                    // Full-column swap (all rows): rows < d hold finished R entries that must travel
                    // with the column; rows >= d hold the live sub-matrix. Stored Householder vectors
                    // of earlier steps live in columns < d and are untouched (both indices are >= d).
                    Swap.Columns(ref A_to_Q, d, pivotCol);
                    P.Swap(d, pivotCol);

                    float tv1 = vn1[d]; vn1[d] = vn1[pivotCol]; vn1[pivotCol] = tv1;
                    float tv2 = vn2[d]; vn2[d] = vn2[pivotCol]; vn2[pivotCol] = tv2;
                }

                QR.genHouseholder(ref A_to_Q, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix (vectorised, see QR.applyReflectorRight).
                QR.applyReflectorRight(ref A_to_Q, ref u, ref w, d);

                // R[d,d] and the stored Householder vector — see QR.decompInPlace (same pattern).
                R[d, d] = A_to_Q[d, d];

                for (int i = d; i < m; i++)
                    A_to_Q[i, d] = u[i];

                // --- downdate trailing norms for the NEXT step (guarded, see the method doc above) ---
                bool anyExact = false;
                for (int j = d + 1; j < n; j++)
                {
                    float v1 = vn1[j];
                    if (v1 <= (float)0)
                        continue; // already (exactly) zero -- stays zero, no decay possible

                    float v2 = vn2[j];
                    if (v2 <= (float)0)
                    {
                        // Defensive only: v2 tracks v1's own history (both set together at init and at
                        // every exact re-sum), so v1 > 0 with v2 <= 0 should not occur; treat it as an
                        // immediate guard trip rather than divide by a non-positive v2.
                        anyExact = true;
                        continue;
                    }

                    float ratio = math.abs(A_to_Q[d, j]) / v1;
                    // (1+ratio)*(1-ratio) instead of 1-ratio*ratio: the SAME value algebraically, but
                    // avoids cancellation when ratio is close to 1 — (1-ratio) is small and exact,
                    // (1+ratio) is close to 2 — matching LAPACK's own formulation exactly.
                    float temp = math.max((float)0, ((float)1 + ratio) * ((float)1 - ratio));
                    float decaySinceExact = v1 / v2;
                    float temp2 = temp * decaySinceExact * decaySinceExact;

                    if (temp2 <= Consts.floatSqrtEps)
                        anyExact = true; // cumulative decay too large to trust — re-sum below
                    else
                        vn1[j] = v1 * math.sqrt(temp);
                }

                if (anyExact)
                {
                    int L = n - (d + 1);
                    if (L > 0)
                    {
                        unsafe
                        {
                            float* qp = A_to_Q.Data.Ptr;
                            float* vp = vn1.Data.Ptr;
                            UnsafeUtility.MemClear(vp + (d + 1), (long)L * UnsafeUtility.SizeOf<float>());
                            for (int r = d + 1; r < m; r++)
                                UnsafeOP.addSquares(vp + (d + 1), qp + (long)r * n + (d + 1), L);
                        }
                        for (int j = d + 1; j < n; j++)
                        {
                            float nrm = math.sqrt(vn1[j]);
                            vn1[j] = nrm;
                            vn2[j] = nrm;
                        }
                    }
                }
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

            w.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // ---- level-3 dispatch: pick the blocked (dlaqps-style) core once the problem is wide enough ----

        // Every decomposition path (decompInPlace / decomp / solveInPlace, plain- and cache-scratch)
        // routes through here. Mirrors QR's size gate: the partially-blocked panel core only earns its
        // extra bookkeeping (an F matrix, a per-panel GEMM flush, blocked Q reconstruction) once
        // N_Cols >= 2*QRCP_BLOCK; below that the original per-reflector unblocked core (decompInPlaceCore)
        // is already fast and has no panel overhead. Both cores take the SAME vn1/vn2 downdating scratch.
        static DirectSolveInfo decompCoreDispatch(ref floatMxN A_to_Q, ref floatMxN R, ref Pivot P, ref floatN u,
                                                  ref floatN vn1, ref floatN vn2)
        {
            // See decompInPlaceBlockedCore for why this is a method-local const, not a class field.
            const int QRCP_BLOCK = 32;

            if (A_to_Q.N_Cols >= 2 * QRCP_BLOCK)
                return decompInPlaceBlockedCore(ref A_to_Q, ref R, ref P, ref u, ref vn1, ref vn2);

            return decompInPlaceCore(ref A_to_Q, ref R, ref P, ref u, ref vn1, ref vn2);
        }

        // ---- blocked (LAPACK dgeqp3/dlaqps-style) partially-blocked panel core ----
        //
        // Raises QRCP from level-2 (each reflector's trailing update applied immediately as two
        // memory-bound passes) to level-3: the trailing update of a whole panel of up to QRCP_BLOCK
        // reflectors is DEFERRED and applied once, as a single rank-kb GEMM (UnsafeOP.wySubVW), per
        // panel. The enabler is norm DOWNDATING (see decompInPlaceCore): pivot selection needs the
        // current column norms, not the current column DATA, so vn1 lets us choose pivots while the
        // trailing matrix stays stale between flushes. Full derivation + range table:
        // docs/spec-qrcp-blocked.md. The reflectors are stored exactly as QR's (τ≡1 Householder
        // vectors in the lower triangle), so Q is reconstructed by the SAME blocked-WY kernel QR uses
        // (QR.reconstructQBlocked) — only pivoting differs, and that is confined to the factorization.
        //
        // F is the (width x kb) accumulator with the invariant A_true = A_stale − V·Fᵀ over the panel's
        // not-yet-flushed rows; V is the panel's stored reflectors. Per panel step k (panel-local, the
        // pivot lands on global column/row d = rk = p0+k):
        //   1. pivot by max vn1 over trailing columns; the full-column swap in A carries each column's
        //      already-written R prefix with it (R is extracted from A's upper triangle at the end),
        //      so no separate R swap is needed — only vn1/vn2/P and the k filled F rows are swapped.
        //   2. bring ONLY the pivot column up to date wrt the k prior reflectors (A[:,d] −= V·F[k,·]ᵀ).
        //   3. generate the Householder reflector; 4. take R[d,d] from it and store the reflector.
        //   5. ONE combined pass acc = uᵀ·A over the panel width: acc's reflector-column entries are
        //      the compact-WY aux (uₖᵀuᵢ), its trailing entries are the direct term of F's new column.
        //   6. F's new column = direct − F·aux (correction).  7. bring row rk of the trailing part up
        //      to date (it becomes R and feeds the norm downdate).  8. downdate vn1 with the same
        //      guarded formula as the unblocked core BUT, because the trailing matrix is stale
        //      mid-panel, a tripped column is MARKED (its vn1 left stale) rather than re-summed on the
        //      spot; the first trip cuts the panel short (kb = k+1) — dlaqps returns KB for this reason.
        // Panel end: one GEMM flush of the deferred trailing update, THEN an exact re-sum of the marked
        // columns over the now-updated trailing matrix.
        static unsafe DirectSolveInfo decompInPlaceBlockedCore(ref floatMxN A_to_Q, ref floatMxN R, ref Pivot P,
                                                               ref floatN u, ref floatN vn1, ref floatN vn2)
        {
            // Factorization panel width. A method-local const (QRCP is a partial class shared by the
            // float/double generated files, so a class-level const of this name would collide, CS0102).
            // 32 measured the sweep optimum (docs/spec-qrcp-blocked.md OQ-B1: 16 and 64 both lost ~2-10%
            // at 2048x512 float) — same width QR settled on; the pivoted core's heavier per-step level-2
            // work doesn't shift the optimum the way the spec speculated it might.
            const int QRCP_BLOCK = 32;

            // Q reconstruction runs QR's shared blocked-WY kernel, which is hardwired to a 32-wide
            // block (QR_BLOCK) — so its scratch (Vpanel/Tbuf/Wbuf/tcolBuf) is sized for the LARGER of
            // the two widths, keeping the factorization NB free to differ from reconstruction's fixed 32
            // (at NB=32 they coincide and rb==QRCP_BLOCK; the max() just guards a future NB change).
            const int RECON_BLOCK = 32;
            int rb = QRCP_BLOCK > RECON_BLOCK ? QRCP_BLOCK : RECON_BLOCK;

            P.Reset();

            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            // Deferred-update scratch (Allocator.Temp — one set per call, negligible against the
            // O(n²m) factorization). F is row-major (F[jl,k] at Fp[jl*QRCP_BLOCK+k]): row jl is a
            // trailing A-column, contiguous over the panel step k, which makes the step-2/5/7 dots and
            // the flush transpose cache-friendly. acc is the uᵀA accumulator (also carries the WY aux).
            // mark flags guard-tripped columns (float 0/1 — new floatN(.., false) zero-inits).
            var F = new floatN(n * QRCP_BLOCK, Allocator.Temp, true);
            var acc = new floatN(n, Allocator.Temp, true);
            var mark = new floatN(n, Allocator.Temp, false);
            // Blocked-WY reconstruction buffers — sized for max(QRCP_BLOCK, RECON_BLOCK) so they satisfy
            // both the flush GEMM (<= QRCP_BLOCK wide) and QR.reconstructQBlocked (RECON_BLOCK wide).
            var Vpanel = new floatN(m * rb, Allocator.Temp, true);
            var Tbuf = new floatN(rb * rb, Allocator.Temp, true);
            var Wbuf = new floatN(rb * n, Allocator.Temp, true);
            var tcolBuf = new floatN(rb, Allocator.Temp, true);
            var VfullBuf = new floatN(m * n, Allocator.Temp, true);

            // scale-relative zero-column threshold (see QR.genHouseholder); computed once on the
            // original A, exactly as the unblocked core does.
            float zeroThreshold = Consts.floatZeroThreshold * Norms.LInf(in A_to_Q);

            float* Ap = A_to_Q.Data.Ptr;
            float* Fp = F.Data.Ptr;
            float* accp = acc.Data.Ptr;
            float* markp = mark.Data.Ptr;
            float* up = u.Data.Ptr;
            float* vn1p = vn1.Data.Ptr;
            float* vn2p = vn2.Data.Ptr;

            // Initial partial norms == exact full column norms (single row-major addSquares sweep);
            // vn2 == vn1 (no decay yet). Identical to the unblocked core's init.
            {
                float* vp = vn1p;
                UnsafeUtility.MemClear(vp, (long)n * UnsafeUtility.SizeOf<float>());
                for (int r = 0; r < m; r++)
                    UnsafeOP.addSquares(vp, Ap + (long)r * n, n);
            }
            for (int j = 0; j < n; j++)
            {
                float nrm = math.sqrt(vn1p[j]);
                vn1p[j] = nrm;
                vn2p[j] = nrm;
            }

            // Same relative tie tolerance as the unblocked core (see decompInPlaceCore).
            float pivotRelTol = (float)(8 * m) * Consts.floatEpsilon;
            float pivotRelTolRoot = math.sqrt((float)1 + pivotRelTol);

            int p0 = 0;
            while (p0 < n)
            {
                int pb = math.min(QRCP_BLOCK, n - p0);
                int width = n - p0;
                int kb = pb;                              // shortened to k+1 if a column trips (below)

                for (int k = 0; k < pb; k++)
                {
                    int d = p0 + k;
                    int rk = d;

                    // --- 1. pivot: largest tracked partial norm among trailing columns [d, n) ---
                    float diagNorm1 = vn1p[d];
                    int pivotCol = d;
                    float maxNorm1 = diagNorm1;
                    for (int c = d + 1; c < n; c++)
                        if (vn1p[c] > maxNorm1) { maxNorm1 = vn1p[c]; pivotCol = c; }

                    // Tie guard (see decompInPlaceCore): only pivot past accumulated rounding noise.
                    if (pivotCol != d && maxNorm1 > diagNorm1 * pivotRelTolRoot)
                    {
                        Swap.Columns(ref A_to_Q, d, pivotCol);   // full column: carries the R prefix too
                        P.Swap(d, pivotCol);
                        float tv1 = vn1p[d]; vn1p[d] = vn1p[pivotCol]; vn1p[pivotCol] = tv1;
                        float tv2 = vn2p[d]; vn2p[d] = vn2p[pivotCol]; vn2p[pivotCol] = tv2;
                        // Swap the k already-filled F rows (columns 0..k-1) for the two A-columns —
                        // panel-local rows k (the pivot slot) and pivotCol-p0.
                        int jlp = pivotCol - p0;
                        float* Frk = Fp + (long)k * QRCP_BLOCK;
                        float* Frp = Fp + (long)jlp * QRCP_BLOCK;
                        for (int i = 0; i < k; i++) { float tf = Frk[i]; Frk[i] = Frp[i]; Frp[i] = tf; }
                    }

                    // --- 2. bring the pivot column up to date: A[r,d] −= Σ_{i<k} u_i[r]·F[k,i] ---
                    //     u_i is stored in A[:, p0+i]; F row k is contiguous. Rows [rk, m) only (rows
                    //     above rk are finished R entries, already made true by prior row updates).
                    if (k > 0)
                    {
                        float* Frk = Fp + (long)k * QRCP_BLOCK;
                        for (int r = rk; r < m; r++)
                        {
                            float* Aseg = Ap + (long)r * n + p0;   // A[r, p0 + i]
                            float s = (float)0;
                            for (int i = 0; i < k; i++)
                                s += Aseg[i] * Frk[i];
                            Aseg[k] -= s;                            // Aseg[k] == A[r, d]
                        }
                    }

                    // --- 3. Householder reflector for the (now up-to-date) column d ---
                    QR.genHouseholder(ref A_to_Q, ref u, d, zeroThreshold);

                    // --- 4. R[d,d] = reflector applied to its own column, then store the reflector ---
                    //     beta = uᵀ·col_d (ascending rows) ⇒ R[d,d] = A[d,d] − u[d]·beta, matching what
                    //     applyReflectorRight would write. col_d still holds the original data here.
                    float beta = (float)0;
                    for (int r = d; r < m; r++)
                        beta += up[r] * Ap[(long)r * n + d];
                    R[d, d] = A_to_Q[d, d] - up[d] * beta;
                    for (int r = d; r < m; r++)
                        Ap[(long)r * n + d] = up[r];

                    // --- 5. one combined pass acc[jl] = Σ_{r=rk}^{m-1} u[r]·A[r, p0+jl], jl in [0,width) ---
                    //     Reflector columns (jl<k, now holding u_i) give acc[jl] = uₖᵀuᵢ (the compact-WY
                    //     aux); trailing columns (jl>k, still original data) give F's direct term. One
                    //     unit-stride axpy per row — the read-only GEMV pass over the stale trailing
                    //     matrix that replaces the unblocked kernel's two memory-bound passes.
                    UnsafeUtility.MemClear(accp, (long)width * UnsafeUtility.SizeOf<float>());
                    for (int r = rk; r < m; r++)
                        UnsafeOP.axpy(accp, Ap + (long)r * n + p0, up[r], width);

                    // --- 6. new F column: F[jl,k] = acc[jl] − Σ_{i<k} F[jl,i]·acc[i]  (trailing jl only) ---
                    for (int jl = k + 1; jl < width; jl++)
                    {
                        float* Fjl = Fp + (long)jl * QRCP_BLOCK;
                        float s = accp[jl];
                        for (int i = 0; i < k; i++)
                            s -= Fjl[i] * accp[i];
                        Fjl[k] = s;
                    }

                    // --- 7. bring row rk of the trailing part up to date (becomes R; feeds the downdate) ---
                    //     A[rk, p0+jl] −= Σ_{i=0}^{k} u_i[rk]·F[jl,i]  — 0..k INCLUSIVE of the reflector
                    //     just generated (its contribution to this row arrives via F's new column).
                    {
                        float* Ark = Ap + (long)rk * n + p0;       // A[rk, p0 + i]
                        for (int jl = k + 1; jl < width; jl++)
                        {
                            float* Fjl = Fp + (long)jl * QRCP_BLOCK;
                            float s = (float)0;
                            for (int i = 0; i <= k; i++)
                                s += Ark[i] * Fjl[i];
                            Ark[jl] -= s;                            // Ark[jl] == A[rk, p0+jl]
                        }
                    }

                    // --- 8. downdate trailing norms (guarded). Trip ⇒ MARK (leave vn1 stale) + cut panel.
                    //     Unlike the unblocked core we must NOT re-sum on the spot: the trailing matrix
                    //     is stale below row rk mid-panel, so an immediate re-sum would be wrong. LAPACK
                    //     defers to panel end; we mark and, on the first trip, cut the panel short. ---
                    bool tripped = false;
                    {
                        float* Ark = Ap + (long)rk * n;
                        for (int jl = k + 1; jl < width; jl++)
                        {
                            int c = p0 + jl;
                            float v1 = vn1p[c];
                            if (v1 <= (float)0)
                                continue;

                            float v2 = vn2p[c];
                            if (v2 <= (float)0)
                            {
                                markp[c] = (float)1; tripped = true; continue;   // defensive; see unblocked core
                            }

                            float ratio = math.abs(Ark[c]) / v1;
                            float temp = math.max((float)0, ((float)1 + ratio) * ((float)1 - ratio));
                            float dse = v1 / v2;
                            float temp2 = temp * dse * dse;

                            if (temp2 <= Consts.floatSqrtEps) { markp[c] = (float)1; tripped = true; }
                            else vn1p[c] = v1 * math.sqrt(temp);
                        }
                    }

                    if (tripped) { kb = k + 1; break; }
                }

                // --- panel flush: one rank-kb GEMM applies the whole panel's deferred trailing update.
                //     A[rows cStart.., cols cStart..) −= V·Fᵀ, with V = the kb stored reflectors and
                //     W := Fᵀ (kb x cw) built by transposing F's flush rows. Row rk_last is already
                //     current from step 7, so the flush starts one row below it (cStart = p0+kb). ---
                int cStart = p0 + kb;
                int cw = n - cStart;
                int frows = m - cStart;
                if (cw > 0 && frows > 0)
                {
                    float* Wp = Wbuf.Data.Ptr;
                    for (int i = 0; i < kb; i++)
                    {
                        float* Wi = Wp + (long)i * cw;
                        for (int jl2 = 0; jl2 < cw; jl2++)
                            Wi[jl2] = Fp[(long)(kb + jl2) * QRCP_BLOCK + i];   // Wᵀ: W[i,jl'] = F[kb+jl', i]
                    }
                    float* Vp = Ap + (long)cStart * n + p0;        // V: rows [cStart,m), cols [p0, p0+kb)
                    float* Cp = Ap + (long)cStart * n + cStart;    // C: rows [cStart,m), cols [cStart,n)
                    UnsafeOP.wySubVW(Vp, n, Cp, n, frows, kb, cw, Wp);
                }

                // --- re-sum marked columns over the now-updated trailing matrix (rows [cStart, m)) ---
                if (cw > 0)
                {
                    for (int c = cStart; c < n; c++)
                    {
                        if (markp[c] == (float)0)
                            continue;
                        float s = (float)0;
                        for (int r = cStart; r < m; r++)
                        {
                            float a = Ap[(long)r * n + c];
                            s += a * a;
                        }
                        float nrm = math.sqrt(s);
                        vn1p[c] = nrm;
                        vn2p[c] = nrm;
                        markp[c] = (float)0;
                    }
                }

                p0 += kb;                                  // advance by kb (not pb) so cut columns re-enter
            }

            // R off-diagonal from A's upper triangle (each R[d,d] was written per step above).
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r) R[r, c] = 0;
                else if (c > r) R[r, c] = A_to_Q[r, c];
            }

            // Reconstruct Q from the stored reflectors via the shared blocked-WY kernel.
            QR.reconstructQBlocked(ref A_to_Q, ref Vpanel, ref Tbuf, ref Wbuf, ref tcolBuf, ref VfullBuf);

            VfullBuf.Dispose();
            tcolBuf.Dispose();
            Wbuf.Dispose();
            Tbuf.Dispose();
            Vpanel.Dispose();
            mark.Dispose();
            acc.Dispose();
            F.Dispose();

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
        /// decomp routed through caller-owned <see cref="floatQRCPCache"/> scratch (vn1/vn2) — see
        /// decompInPlace's cache overload. A is copied into Q (one memcpy), then factored in place.
        /// </summary>
        /// <remarks>R must not alias A_to_Q/Q (unchecked) — see decompInPlace. If R, P, u, or the
        /// cache is mis-sized, this throws AFTER Q has already been overwritten with a copy of A
        /// (still un-factored); A itself is always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in floatMxN A, ref floatMxN Q, ref floatMxN R, ref Pivot P, ref floatN u, ref floatQRCPCache cache)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QRCP.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R, ref P, ref u, ref cache);
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

            return solveInPlaceFinish(ref A_to_Q, ref b, ref x, ref R, ref P, ref u, relTol);
        }

        /// <summary>
        /// solveInPlace routed through caller-owned <see cref="floatQRCPCache"/> scratch (vn1/vn2)
        /// for the internal decomposition step — see decompInPlace's cache overload. Same semantics
        /// as the 7-arg primitive above.
        /// </summary>
        /// <param name="A_to_Q">On entry A (m x n, m >= n); on exit the orthogonal factor Q.</param>
        /// <param name="b">Right-hand side, length m. Preserved (read-only). Must not alias x.</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true. Solution, length n.</param>
        /// <param name="R">Scratch: n x n (receives upper-triangular factor; consumed).</param>
        /// <param name="P">Scratch: column Pivot of size n (reset internally).</param>
        /// <param name="u">Scratch: length EXACTLY m.</param>
        /// <param name="cache">Caller-owned vn1/vn2 scratch — see Arena.floatQRCPCache.</param>
        /// <param name="relTol">Rank threshold ratio; tol = relTol * |R[0,0]|. Negative = auto default.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref floatMxN A_to_Q, ref floatN b, ref floatN x,
                                           ref floatMxN R, ref Pivot P,
                                           ref floatN u, ref floatQRCPCache cache, float relTol)
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

            if (relTol < (float)0)
                relTol = (float)(math.max(m, n)) * Consts.floatZeroThreshold;

            if (n == 0) return new RankInfo { status = DirectSolveStatus.Success, rank = 0 };

            RequireQRCPWorkspace(in cache, n);

            decompCoreDispatch(ref A_to_Q, ref R, ref P, ref u, ref cache.vn1, ref cache.vn2);

            return solveInPlaceFinish(ref A_to_Q, ref b, ref x, ref R, ref P, ref u, relTol);
        }

        // Default-tolerance cache overload: passes the auto sentinel (relTol < 0) — see the 7-arg
        // primitive's default-tolerance overload below.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankInfo solveInPlace(ref floatMxN A_to_Q, ref floatN b, ref floatN x,
                                           ref floatMxN R, ref Pivot P,
                                           ref floatN u, ref floatQRCPCache cache)
        {
            return solveInPlace(ref A_to_Q, ref b, ref x, ref R, ref P, ref u, ref cache, (float)(-1));
        }

        // Shared tail: rank detection from R's diagonal + back-substitution + un-permute. Factored
        // out so the plain-scratch and cache-scratch overloads above (which differ only in HOW the
        // decomposition step is reached) share one copy of this logic — see decompInPlaceCore for the
        // analogous split on the decomposition side.
        static RankInfo solveInPlaceFinish(ref floatMxN A_to_Q, ref floatN b, ref floatN x,
                                            ref floatMxN R, ref Pivot P, ref floatN u, float relTol)
        {
            int n = A_to_Q.N_Cols;

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
