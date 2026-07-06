#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class QR {

        // ---- Scratch contract across decomp / decompInPlace / solveInPlace ----
        //   (no scratch param)      — allocating convenience: Allocator.Temp scratch, BLOCKED
        //                              (level-3 / compact-WY) once N_Cols >= 2*QR_BLOCK.
        //   ref u[, ref w]          — caller-provided scratch, LEVEL-2 (unblocked) kernel: the
        //                              minimal zero-alloc path, cheapest for small N.
        //   ref doubleQRCache cache — caller-provided scratch, BLOCKED (same kernel, same gate as
        //                              the allocating overload — bit-identical results); zero-alloc
        //                              AND fastest for repeated large-N use. solveInPlace's cache
        //                              overload only ever touches cache.u/cache.w — its fused kernel
        //                              never forms Q, so the blocked-WY buffers are unused there.

        // internal (not private): shared with QRCP.decompInPlace/solveInPlace, which live in a
        // separate class after the QR/QRCP split but reuse the same Householder kernels.
        internal static double sign(double x) {
            return x < 0 ? -1 : 1;
        }

        // zeroThreshold is the ABSOLUTE column-norm below which a column is treated as zero. Callers
        // pass a SCALE-RELATIVE value (Consts.doubleZeroThreshold * matrix magnitude) so QR is
        // scale-invariant — a fixed absolute constant mis-classifies every column of a uniformly
        // tiny-magnitude matrix as a zero column and silently produces a garbage decomposition.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void genHouseholder(ref doubleMxN Q, ref doubleN u, int k, double zeroThreshold) {

            for (int r = k; r < u.N; r++)
                u[r] = Q[r, k];

            double xNorm = Norms.L2Range(u, k, u.N);

            if (math.abs(xNorm) > zeroThreshold) {

                for (int r = k; r < u.N; r++)
                    u[r] = u[r] / xNorm;

                u[k] = u[k] + sign(u[k]);

                var div = math.sqrt(math.abs(u[k]));
                for (int r = k; r < u.N; r++) {
                    u[r] = u[r] / div;
                }
            }
            else {

                u[k] = math.SQRT2;
            }
        }

        // Apply a Householder reflector to the trailing submatrix in place, restricted to columns
        // [d, colEnd):
        //     Q[d:, d:colEnd] -= u · (uᵀ · Q[d:, d:colEnd]).
        // Two contiguous-memory passes through the vectorising UnsafeOP.axpy ([NoAlias]) — the same
        // raw-pointer path GEMM uses, so Burst SIMD-vectorises the inner work (float runs ~2x double).
        // The previous formulation looped over rows r (stride N_Cols when indexing Q[r, c]), which
        // Burst cannot vectorise — it vectorises loops, and the unit-stride axis here is the columns,
        // not r. Walking each row left-to-right instead lets axpy run at GEMM speed.
        //
        // w is scratch of length >= (colEnd - d); only w[0..L) is used. Bitwise identical to the
        // prior per-column scalar form: pass 1 accumulates each w[i] over rows r = d..M-1 in the SAME
        // ascending order, and pass 2's (-u[r])·w[i] added to Q[r,c] equals Q[r,c] - u[r]·w[i]
        // exactly in IEEE (negation and sign-symmetric multiply are exact).
        //
        // colEnd lets the blocked (compact-WY) factorization restrict the per-column reflector apply
        // to just its own panel (cols [d, p0+pb)) instead of the whole trailing matrix — the panel's
        // remaining columns [p0+pb, N) are updated once per PANEL as a block GEMM instead of once per
        // COLUMN; see qrDecompositionBlockedCore.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void applyReflectorRightCols(ref doubleMxN Q, ref doubleN u, ref doubleN w, int d, int colEnd)
        {
            int M = Q.M_Rows;
            int N = Q.N_Cols;
            int L = colEnd - d;                     // width of the (restricted) trailing column block
            if (L <= 0)
                return;

            double* qp = Q.Data.Ptr;
            double* up = u.Data.Ptr;
            double* wp = w.Data.Ptr;

            // pass 1: w[0..L) = Σ_{r=d}^{M-1} u[r] · Q[r, d..colEnd)   (row segments are unit-stride)
            UnsafeUtility.MemClear(wp, (long)L * UnsafeUtility.SizeOf<double>());
            for (int r = d; r < M; r++)
                UnsafeOP.axpy(wp, qp + (long)r * N + d, up[r], L);

            // pass 2: Q[r, d..colEnd) += (-u[r]) · w[0..L)  ==  Q[r, d..colEnd) -= u[r] · w
            for (int r = d; r < M; r++)
                UnsafeOP.axpy(qp + (long)r * N + d, wp, -up[r], L);
        }

        // Un-restricted form: applies to the full trailing block [d, N_Cols). Used by every path
        // that has not been raised to the blocked (compact-WY) factorization — the zero-alloc
        // decompInPlace overload, QRCP.decompInPlace, solveInPlace, and Q-reconstruction.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void applyReflectorRight(ref doubleMxN Q, ref doubleN u, ref doubleN w, int d)
        {
            applyReflectorRightCols(ref Q, ref u, ref w, d, Q.N_Cols);
        }

        // Caller-provided scratch overload — LEVEL-2 (unblocked) zero-alloc tier: u is a workspace
        // vector of length EXACTLY Q.M_Rows; w is a workspace vector of length >= Q.N_Cols (the
        // reflector-apply accumulator). Hoist both out of a hot loop to skip the per-call
        // Allocator.Temp allocs. This is the minimal-scratch path (cheapest for small N); for a
        // zero-alloc path that ALSO gains the level-3 blocked kernel at large N, use the
        // ref doubleQRCache overload instead. See the scratch-contract note at the top of this class.
        // Always reports DirectSolveStatus.Success — this factorization has no failure mode (a
        // zero-norm column is handled via the sign-convention fallback in genHouseholder, not
        // rejected).
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref doubleMxN A_to_Q, ref doubleMxN R, ref doubleN u, ref doubleN w)
        {
            if (A_to_Q.M_Rows < A_to_Q.N_Cols)
                throw new ArgumentException("QR.decompInPlace: Matrix R must be square or tall (more or equal rows than cols)");

            if (u.N != A_to_Q.M_Rows)
                throw new ArgumentException("QR.decompInPlace: scratch vector u.N must equal A_to_Q.M_Rows");

            if (w.N < A_to_Q.N_Cols)
                throw new ArgumentException("QR.decompInPlace: scratch vector w.N must be at least A_to_Q.N_Cols");

            int qrSteps = A_to_Q.N_Cols;

            // scale-relative zero-column threshold (see genHouseholder); LInf(Q) == max |entry|.
            double zeroThreshold = Consts.doubleZeroThreshold * Norms.LInf(in A_to_Q);

            for (int d = 0; d < qrSteps; d++)
            {
                genHouseholder(ref A_to_Q, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix: Q[d:, d:] -= u·(uᵀ·Q[d:, d:]).
                // Vectorised, zero-alloc (w is caller scratch). See applyReflectorRight.
                applyReflectorRight(ref A_to_Q, ref u, ref w, d);

                R[d, d] = A_to_Q[d, d];

                // copy v into Q below diagonal, will be used to reconstruct Q
                for (int i = d; i < A_to_Q.M_Rows; i++)
                {
                    A_to_Q[i, d] = u[i];
                }
            }
            // Copy the upper triangular part of Q into R
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                {
                    R[r, c] = 0;
                }
                else if (c > r)
                {
                    R[r, c] = A_to_Q[r, c];
                }
            }

            // Reconstruct Q from vectors stored inside Q columns

            // Initialize upper part of Q to identity matrix, including diagonals
            for (int r = 0; r < A_to_Q.M_Rows; r++)
            {
                for (int c = r; c < A_to_Q.N_Cols; c++)
                {
                    if (c > r)
                    {
                        A_to_Q[r, c] = 0;
                    }
                }
            }

            // Apply Householder transformations in reverse order
            // Reconstruct the Householder vector v from the original Q
            for (int d = A_to_Q.N_Cols - 1; d >= 0; d--)
            {
                // includes diagonal elements
                for (int i = d; i < A_to_Q.M_Rows; i++)
                {
                    u[i] = A_to_Q[i, d];
                    A_to_Q[i, d] = i == d? 1 : 0;
                }

                // Apply the reflector to the trailing columns: Q[d:, d:] -= u·(uᵀ·Q[d:, d:]).
                // Same vectorised, zero-alloc helper as the factorization apply above.
                applyReflectorRight(ref A_to_Q, ref u, ref w, d);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Blocked (level-3 / compact-WY, GEMM trailing-update) factorization core. τ≡1 convention
        // throughout (see file-header notes on genHouseholder / applyReflectorRight): each
        // H_i = I - u_i u_iᵀ, so the compact-WY T has T[i,i] = 1 (not LAPACK's τ-scaled diagonal).
        //
        // Panels of QR_BLOCK columns are factored with the existing rank-1 sweep (cheap — pb is
        // small), but their combined effect on the REST of the matrix is applied once per panel as
        // two GEMM-shaped passes (UnsafeOP.wyVtC / wySubVW, unit-stride inner loop) instead of pb
        // separate rank-1 (applyReflectorRight) passes — the memory-traffic-bound part of the
        // algorithm. Reconstruction of Q is similarly batched, applying panels right-to-left.
        //
        // Direction matters and is easy to get backwards (see spec landmines):
        //   factorization applies (I - V T Vᵀ)ᵀ = I - V Tᵀ Vᵀ   → wyTriTransMul (Tᵀ)
        //   reconstruction applies  I - V T Vᵀ  (un-transposed)  → wyTriMul (T)
        //
        // Scratch (all caller-provided, sized by the decompInPlace(Q,R) allocating wrapper):
        //   u       length M_Rows        — Householder vector (per-column panel factor step).
        //   w       length N_Cols        — per-column reflector-apply accumulator (panel-local).
        //   Vpanel  length M_Rows*QR_BLOCK — clean contiguous panel, reused for factor+reconstruct;
        //           accessed with leading dimension == the CURRENT pb (<= QR_BLOCK), not QR_BLOCK.
        //   Tbuf    length QR_BLOCK*QR_BLOCK — compact-WY T, reused per panel.
        //   Wbuf    length QR_BLOCK*N_Cols — block-apply GEMM scratch. Worst case is reconstruction's
        //           LAST-processed panel (p0 = 0), whose column range [p0, n) is the full N_Cols width.
        //   tcolBuf length QR_BLOCK      — formT scratch.
        //   VfullBuf length M_Rows*N_Cols — clean masked copy of the stored reflectors, needed
        //           because reconstruction overwrites Q in place while still reading V.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void qrDecompositionBlockedCore(ref doubleMxN Q, ref doubleMxN R,
            ref doubleN u, ref doubleN w,
            ref doubleN Vpanel, ref doubleN Tbuf, ref doubleN Wbuf, ref doubleN tcolBuf, ref doubleN VfullBuf)
        {
            // Panel width for the blocked (level-3 / compact-WY) factorization path. 32 columns
            // keeps the panel (and its T factor) tiny relative to cache while still amortising the
            // trailing-update GEMM over enough columns to reach GEMM-shaped throughput. A method-
            // local const (not a class field) — QR is a partial class shared by the float/double
            // generated files, so a class-level const of the same name would collide (CS0102).
            const int QR_BLOCK = 32;

            int m = Q.M_Rows;
            int n = Q.N_Cols;

            // scale-relative zero-column threshold (see genHouseholder); LInf(Q) == max |entry|.
            double zeroThreshold = Consts.doubleZeroThreshold * Norms.LInf(in Q);

            double* Qp = Q.Data.Ptr;
            double* Vp = Vpanel.Data.Ptr;
            double* T = Tbuf.Data.Ptr;
            double* Wmat = Wbuf.Data.Ptr;
            double* tcol = tcolBuf.Data.Ptr;

            // ---- factorization: panels left to right ----
            for (int p0 = 0; p0 < n; p0 += QR_BLOCK)
            {
                int pb = math.min(QR_BLOCK, n - p0);

                // (1) factor panel columns d in [p0, p0+pb); reflector apply restricted to the
                //     panel's OWN remaining columns [d, p0+pb) — cols beyond the panel are updated
                //     once below as a single block GEMM instead of once per column.
                for (int d = p0; d < p0 + pb; d++)
                {
                    genHouseholder(ref Q, ref u, d, zeroThreshold);
                    applyReflectorRightCols(ref Q, ref u, ref w, d, p0 + pb);

                    R[d, d] = Q[d, d];

                    for (int i = d; i < m; i++)
                        Q[i, d] = u[i];
                }

                // (2) gather the clean panel V: local row t (global row p0+t), local col i (global
                //     col p0+i); masked to zero above each reflector's own diagonal (t < i).
                int rows = m - p0;
                for (int t = 0; t < rows; t++)
                {
                    int r = p0 + t;
                    double* Vrow = Vp + (long)t * pb;
                    for (int i = 0; i < pb; i++)
                        Vrow[i] = (t >= i) ? Qp[(long)r * n + (p0 + i)] : (double)0;
                }

                // (3) form the pb x pb compact-WY T (τ≡1) from the panel.
                UnsafeOP.formT(Vp, pb, rows, pb, T, tcol, Wmat);

                // (4) trailing block update on cols [p0+pb, n): C -= V*(Tᵀ*(Vᵀ*C)). One untiled
                //     GEMM call per panel — UnsafeOP.wyVtC/wySubVW already reach full GEMM
                //     throughput (~70 GFLOP/s, matched matMatDot) at this width without tiling;
                //     column-tiling was tried and measured SLOWER (added MemClear/call overhead
                //     for no cache-locality benefit), so it is deliberately not done here.
                int cStart = p0 + pb;
                int cw = n - cStart;
                if (cw > 0)
                {
                    double* Cp = Qp + (long)p0 * n + cStart;
                    UnsafeUtility.MemClear(Wmat, (long)pb * cw * UnsafeUtility.SizeOf<double>());
                    UnsafeOP.wyVtC(Vp, pb, Cp, n, rows, pb, cw, Wmat);
                    UnsafeOP.wyTriTransMul(T, pb, Wmat, cw);      // Tᵀ — factorization direction
                    UnsafeOP.wySubVW(Vp, pb, Cp, n, rows, pb, cw, Wmat);
                }
            }

            // Copy the upper triangular part of Q into R (unchanged from the unblocked path).
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                    R[r, c] = 0;
                else if (c > r)
                    R[r, c] = Q[r, c];
            }

            // ---- reconstruct Q from the stored reflectors, panels right to left ----
            reconstructQBlocked(ref Q, ref Vpanel, ref Tbuf, ref Wbuf, ref tcolBuf, ref VfullBuf);
        }

        // Reconstruct the orthogonal factor Q from Householder reflectors stored in Q's lower triangle
        // (the τ≡1 convention, H_i = I - u_i u_iᵀ), using the compact-WY blocked kernel — panels right
        // to left. Split out of qrDecompositionBlockedCore so QRCP's blocked factorization can reuse
        // the SAME reconstruction (its reflectors are stored identically): only pivoting differs, which
        // is entirely on the factorization side. Reads Q's lower triangle (the stored reflectors) and
        // overwrites Q in place with the reconstructed orthogonal factor.
        //
        // Scratch (same buffers, same sizes, as qrDecompositionBlockedCore's reconstruction phase):
        //   Vpanel  m*QR_BLOCK, Tbuf QR_BLOCK*QR_BLOCK, Wbuf QR_BLOCK*N_Cols, tcolBuf QR_BLOCK,
        //   VfullBuf m*N_Cols (the clean masked reflector snapshot).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void reconstructQBlocked(ref doubleMxN Q,
            ref doubleN Vpanel, ref doubleN Tbuf, ref doubleN Wbuf, ref doubleN tcolBuf, ref doubleN VfullBuf)
        {
            // See qrDecompositionBlockedCore for why this is a method-local const, not a class field.
            const int QR_BLOCK = 32;

            int m = Q.M_Rows;
            int n = Q.N_Cols;

            double* Qp = Q.Data.Ptr;
            double* Vp = Vpanel.Data.Ptr;
            double* T = Tbuf.Data.Ptr;
            double* Wmat = Wbuf.Data.Ptr;
            double* tcol = tcolBuf.Data.Ptr;

            // Snapshot the stored reflectors into a clean, masked (r >= c ? Q[r,c] : 0) full copy
            // BEFORE Q is overwritten below — reconstruction both reads V and writes Q in place.
            double* Vfull = VfullBuf.Data.Ptr;
            for (int r = 0; r < m; r++)
            {
                double* qrow = Qp + (long)r * n;
                double* vrow = Vfull + (long)r * n;
                for (int c = 0; c < n; c++)
                    vrow[c] = (r >= c) ? qrow[c] : (double)0;
            }

            // Seed Q = [I_n; 0] (m x n).
            UnsafeUtility.MemClear(Qp, (long)m * n * UnsafeUtility.SizeOf<double>());
            for (int i = 0; i < n; i++)
                Q[i, i] = 1;

            // Largest multiple of QR_BLOCK that is < n (the last, possibly-narrower, panel).
            int lastP0 = ((n - 1) / QR_BLOCK) * QR_BLOCK;
            for (int p0 = lastP0; p0 >= 0; p0 -= QR_BLOCK)
            {
                int pb = math.min(QR_BLOCK, n - p0);
                int rows = m - p0;

                // Gather Vpanel from the clean snapshot. Vfull is already masked (r >= c ? .. : 0),
                // so no extra masking is needed here — copying directly reproduces it.
                for (int t = 0; t < rows; t++)
                {
                    int r = p0 + t;
                    double* Vrow = Vp + (long)t * pb;
                    double* Vfrow = Vfull + (long)r * n;
                    for (int i = 0; i < pb; i++)
                        Vrow[i] = Vfrow[p0 + i];
                }

                UnsafeOP.formT(Vp, pb, rows, pb, T, tcol, Wmat);

                // Apply the block to columns [p0, n) of Q, rows [p0, m): Q -= V*(T*(Vᵀ*Q)).
                // NOT columns [0, n): columns < p0 are PROVABLY still their original seeded unit
                // vectors at this point (every reflector processed so far — panel-starts >= p0 —
                // has V nonzero only for rows >= p0 > c for any column c < p0, and the seeded
                // column c is nonzero only at row c < p0, so Vᵀ·(column c) is always exactly 0;
                // the block reflector is a no-op on it). This is the SAME invariant the unblocked
                // reconstruction already exploited via applyReflectorRight's cols-[d,N) restriction
                // — skipping columns < p0 here roughly HALVES reconstruction's work (was redoing
                // full-width n every panel).
                int cw = n - p0;
                double* Cp = Qp + (long)p0 * n + p0;
                UnsafeUtility.MemClear(Wmat, (long)pb * cw * UnsafeUtility.SizeOf<double>());
                UnsafeOP.wyVtC(Vp, pb, Cp, n, rows, pb, cw, Wmat);
                UnsafeOP.wyTriMul(T, pb, Wmat, cw);               // T — reconstruction direction
                UnsafeOP.wySubVW(Vp, pb, Cp, n, rows, pb, cw, Wmat);
            }
        }

        // Back-compat workspace overload: takes only the u scratch (length Q.M_Rows) and allocates
        // the small w accumulator (length Q.N_Cols) from Allocator.Temp. Behaviour is identical to
        // the 4-arg primitive; use that one to be fully zero-alloc in a hot loop.
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref doubleMxN A_to_Q, ref doubleMxN R, ref doubleN u)
        {
            var w = new doubleN(A_to_Q.N_Cols, Allocator.Temp, false);
            var info = decompInPlace(ref A_to_Q, ref R, ref u, ref w);
            w.Dispose();
            return info;
        }

        // Allocating wrapper: allocates scratch (Allocator.Temp) and delegates. This is the fast
        // path — it routes to the BLOCKED (level-3 / compact-WY) factorization core once N_Cols is
        // large enough to amortise the extra panel bookkeeping (>= 2*QR_BLOCK columns); smaller
        // matrices fall back to the plain rank-1 sweep, which has no panel/GEMM overhead and is
        // already fast enough at that size. The zero-alloc overloads (ref u / ref u, w) are NOT
        // blocked — they keep the original zero-alloc contract; only this allocating convenience
        // wrapper (used by, e.g., the benchmark and most call sites that don't hoist scratch) gets
        // the speedup.
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref doubleMxN A_to_Q, ref doubleMxN R)
        {
            // See qrDecompositionBlockedCore for why this is a method-local const, not a class field.
            const int QR_BLOCK = 32;

            if (A_to_Q.M_Rows < A_to_Q.N_Cols)
                throw new ArgumentException("QR.decompInPlace: Matrix R must be square or tall (more or equal rows than cols)");

            if (A_to_Q.N_Cols < Consts.doubleQrBlockMinN)   // float/double split (see Consts); default 2*QR_BLOCK
            {
                var uSmall = new doubleN(A_to_Q.M_Rows, Allocator.Temp, false);
                var wSmall = new doubleN(A_to_Q.N_Cols, Allocator.Temp, false);
                var infoSmall = decompInPlace(ref A_to_Q, ref R, ref uSmall, ref wSmall);
                wSmall.Dispose();
                uSmall.Dispose();
                return infoSmall;
            }

            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            var u = new doubleN(m, Allocator.Temp, true);
            var w = new doubleN(n, Allocator.Temp, true);
            var Vpanel = new doubleN(m * QR_BLOCK, Allocator.Temp, true);
            var Tbuf = new doubleN(QR_BLOCK * QR_BLOCK, Allocator.Temp, true);
            var Wbuf = new doubleN(QR_BLOCK * n, Allocator.Temp, true);
            var tcolBuf = new doubleN(QR_BLOCK, Allocator.Temp, true);
            var VfullBuf = new doubleN(m * n, Allocator.Temp, true);

            qrDecompositionBlockedCore(ref A_to_Q, ref R, ref u, ref w, ref Vpanel, ref Tbuf, ref Wbuf, ref tcolBuf, ref VfullBuf);

            VfullBuf.Dispose();
            tcolBuf.Dispose();
            Wbuf.Dispose();
            Tbuf.Dispose();
            Vpanel.Dispose();
            w.Dispose();
            u.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Cache overload: zero-alloc (caller-owned doubleQRCache, see Arena.doubleQRCache) AND
        // BLOCKED once N_Cols >= 2*QR_BLOCK — the same gate, and the same qrDecompositionBlockedCore
        // call, as the fully-allocating overload above, so results are bit-identical to it; only the
        // scratch's allocation source differs (arena-owned buffers vs Allocator.Temp). Below the gate,
        // falls back to the unblocked ref-u,w kernel using cache.u/cache.w — again bit-identical to
        // the allocating overload's own small-N fallback.
        /// <param name="A_to_Q">On entry A; on exit the orthogonal factor Q.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decompInPlace(ref doubleMxN A_to_Q, ref doubleMxN R, ref doubleQRCache cache)
        {
            // See qrDecompositionBlockedCore for why this is a method-local const, not a class field.
            const int QR_BLOCK = 32;

            if (A_to_Q.M_Rows < A_to_Q.N_Cols)
                throw new ArgumentException("QR.decompInPlace: Matrix R must be square or tall (more or equal rows than cols)");

            bool blocked = A_to_Q.N_Cols >= Consts.doubleQrBlockMinN;   // float/double split (see Consts)
            RequireQRWorkspace(in cache, A_to_Q.M_Rows, A_to_Q.N_Cols, blocked);

            if (!blocked)
                return decompInPlace(ref A_to_Q, ref R, ref cache.u, ref cache.w);

            qrDecompositionBlockedCore(ref A_to_Q, ref R, ref cache.u, ref cache.w,
                ref cache.Vpanel, ref cache.Tbuf, ref cache.Wbuf, ref cache.tcolBuf, ref cache.VfullBuf);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // ---- decomp: A-preserving variants (copy A into Q, then delegate to decompInPlace) ----

        /// <summary>
        /// QR decomposition preserving A: A is copied into Q (one memcpy), then factored via
        /// decompInPlace. Q (caller-allocated, same dimensions as A) receives the orthogonal factor;
        /// R (N_Cols x N_Cols) the upper-triangular factor. Zero-alloc primitive (caller-provided u, w
        /// scratch) — see decompInPlace for the scratch-size contract.
        /// Always reports DirectSolveStatus.Success — see decompInPlace.
        /// </summary>
        /// <remarks>If R, u, or w is the wrong size, this throws AFTER Q has already been overwritten
        /// with a copy of A (still un-factored); A itself is always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in doubleMxN A, ref doubleMxN Q, ref doubleMxN R, ref doubleN u, ref doubleN w)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QR.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R, ref u, ref w);
        }

        /// <summary>
        /// decomp allocating its w scratch (Allocator.Temp). See the 5-arg overload for semantics.
        /// </summary>
        /// <remarks>If R or u is the wrong size, this throws AFTER Q has already been overwritten
        /// with a copy of A (still un-factored); A itself is always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in doubleMxN A, ref doubleMxN Q, ref doubleMxN R, ref doubleN u)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QR.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R, ref u);
        }

        /// <summary>
        /// decomp allocating all scratch (Allocator.Temp) and routing through the blocked (level-3)
        /// path once N_Cols is large enough. See decompInPlace's 2-arg overload for the size gate.
        /// </summary>
        /// <remarks>If R is the wrong size, this throws AFTER Q has already been overwritten with a
        /// copy of A (still un-factored); A itself is always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in doubleMxN A, ref doubleMxN Q, ref doubleMxN R)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QR.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R);
        }

        /// <summary>
        /// decomp routed through caller-owned <see cref="doubleQRCache"/> scratch: A is copied into Q
        /// (one memcpy), then factored via decompInPlace's cache overload — zero-alloc AND gains the
        /// blocked (level-3) kernel the same way (bit-identical to the fully-allocating overload above;
        /// see decompInPlace's cache overload).
        /// </summary>
        /// <remarks>If the cache is mis-sized for A's shape, this throws AFTER Q has already been
        /// overwritten with a copy of A (still un-factored); A itself is always preserved.</remarks>
        /// <param name="Q">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the orthogonal factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo decomp(in doubleMxN A, ref doubleMxN Q, ref doubleMxN R, ref doubleQRCache cache)
        {
            if (Q.M_Rows != A.M_Rows || Q.N_Cols != A.N_Cols)
                throw new ArgumentException("QR.decomp: Q must have the same dimensions as A");

            Q.Data.CopyFrom(A.Data);
            return decompInPlace(ref Q, ref R, ref cache);
        }

        /// <summary>
        /// Solve QRx = b for x, with Q,R from a precomputed decomposition (solve for multiple
        /// b vectors reusing one decomposition). Caller provides the destination x (length
        /// Q.N_Cols); x must be distinct from b. Zero-alloc: Qᵀb is formed directly into x with
        /// the ref-dest dot — no internal temporary. dim(b) = Q.M_Rows >= dim(x) = Q.N_Cols.
        /// Always reports DirectSolveStatus.Success — this primitive assumes a valid (non-singular)
        /// triangular factor and does not itself detect a bad one.
        /// </summary>
        /// <param name="Q">Ortho matrix Q from decompInPlace.</param>
        /// <param name="R">Upper triangular matrix R from decompInPlace.</param>
        /// <param name="b">Known vector (length Q.M_Rows). Preserved (read-only).</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true. Solution destination (length Q.N_Cols), must not alias b.</param>
        public static DirectSolveInfo decompSolve(ref doubleMxN Q, ref doubleMxN R, ref doubleN b, ref doubleN x) {
            // Solve Ax = b for x
            // A = QR
            // QRx = b
            // Rx = Q^T b
            // x = R^-1 Q^T b

            if (x.N != Q.N_Cols)
                throw new ArgumentException("QR.decompSolve: x.N must equal Q.N_Cols");

            // x = Q^T b (or b^T Q). The ref-dest dot guards x-aliases-b and zeroes x first.
            Blas.dot(in b, in Q, ref x);
            // Solve Rx = Q^T b for x, in place
            return Blas.triUpper(ref R, ref x);
        }

        /// <summary>
        /// decompSolve convenience: allocates the solution vector x (length Q.N_Cols) from the arena
        /// and returns it. Use the ref-destination overload in hot loops to avoid the allocation.
        /// </summary>
        public static doubleN decompSolve(ref doubleMxN Q, ref doubleMxN R, ref doubleN b) {
            doubleN x = b.doubleTempVec(Q.N_Cols);
            decompSolve(ref Q, ref R, ref b, ref x);
            return x;
        }

        // b is transformed into y = Q^T b, then solved for x; Q and b get modified (destroyed).
        // PRECONDITION: A has FULL COLUMN RANK. This un-pivoted solve back-substitutes through R's
        // diagonal; a rank-deficient A produces a zero on that diagonal and the result x is then
        // Inf/NaN (no guard). For rank-deficient / least-norm problems use the rank-revealing paths
        // instead: QRCP.decompInPlace / QRCP.solveInPlace, SVD.pinvSolve, or CHOP.solveInPlace.
        // Caller-provided scratch overload — LEVEL-2 zero-alloc tier: u is a workspace vector of
        // length EXACTLY A.M_Rows; w is a workspace vector of length >= A.N_Cols (the reflector-apply
        // accumulator). Hoist both out of a hot loop to skip the per-call Allocator.Temp allocs. This
        // is the minimal-scratch path; for a zero-alloc path via a reusable cache struct, use the
        // ref doubleQRCache overload instead (this fused kernel never forms Q, so that overload does
        // NOT gain the level-3 blocked kernel — see the scratch-contract note at the top of this class).
        // Always reports DirectSolveStatus.Success — see the PRECONDITION note above: a
        // rank-deficient A silently divides by a zero R diagonal instead of being detected here.
        /// <param name="A">Destroyed; contents undefined after return (becomes R + stored reflectors scratch).</param>
        /// <param name="b">Destroyed; contents undefined after return (becomes Qᵀb scratch).</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo solveInPlace(ref doubleMxN A, ref doubleN b, ref doubleN x, ref doubleN u, ref doubleN w) {
            if (A.M_Rows < A.N_Cols)
                throw new ArgumentException("QR.solveInPlace: Matrix A must be square or tall (more or equal rows than cols)");

            if (b.N != A.M_Rows)
                throw new ArgumentException("QR.solveInPlace: b.N must equal A.M_Rows");

            if (x.N != A.N_Cols)
                throw new ArgumentException("QR.solveInPlace: x.N must equal A.N_Cols");

            if (u.N != A.M_Rows)
                throw new ArgumentException("QR.solveInPlace: scratch vector u.N must equal A.M_Rows");

            if (w.N < A.N_Cols)
                throw new ArgumentException("QR.solveInPlace: scratch vector w.N must be at least A.N_Cols");

            int qrSteps = A.N_Cols;

            // scale-relative zero-column threshold (see genHouseholder); LInf(A) == max |entry|.
            double zeroThreshold = Consts.doubleZeroThreshold * Norms.LInf(in A);

            double dotProduct = 0;
            for (int d = 0; d < qrSteps; d++) {

                genHouseholder(ref A, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix (vectorised, see applyReflectorRight).
                applyReflectorRight(ref A, ref u, ref w, d);

                // apply same transformation to b vector (O(n) — left scalar)
                dotProduct = 0;
                for (int r = d; r < A.M_Rows; r++)
                    dotProduct += u[r] * b[r];

                for (int r = d; r < A.M_Rows; r++)
                    b[r] -= u[r] * dotProduct;
            }

            // copy b into x (x may be smaller dimension than b)
            for (int r = 0; r < A.N_Cols; r++)
                x[r] = b[r];

            // b was transformed to y, where y = Q^T b
            // Solve Rx = y

            return Blas.triUpper(ref A, ref x);
        }

        // Allocating wrapper: allocates the reflector-apply accumulator w (Allocator.Temp) and
        // delegates to the 5-arg (u, w) primitive above. Behaviour (and results) identical to it;
        // use that one to be fully zero-alloc in a hot loop.
        /// <param name="A">Destroyed; contents undefined after return (becomes R + stored reflectors scratch).</param>
        /// <param name="b">Destroyed; contents undefined after return (becomes Qᵀb scratch).</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo solveInPlace(ref doubleMxN A, ref doubleN b, ref doubleN x, ref doubleN u) {
            var w = new doubleN(A.N_Cols, Allocator.Temp, false);
            var info = solveInPlace(ref A, ref b, ref x, ref u, ref w);
            w.Dispose();
            return info;
        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates. This is
        // the allocating/convenience tier (see the scratch-contract note at the top of this class).
        /// <param name="A">Destroyed; contents undefined after return (becomes R + stored reflectors scratch).</param>
        /// <param name="b">Destroyed; contents undefined after return (becomes Qᵀb scratch).</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo solveInPlace(ref doubleMxN A, ref doubleN b, ref doubleN x) {
            var u = new doubleN(A.M_Rows, Allocator.Temp, false);
            var info = solveInPlace(ref A, ref b, ref x, ref u);
            u.Dispose();
            return info;
        }

        // Cache overload: zero-alloc (caller-owned doubleQRCache, see Arena.doubleQRCache) — routes
        // to the SAME fused, never-forms-Q kernel as the (ref u) / (ref u, ref w) overloads (using
        // cache.u/cache.w in place of caller- or Temp-provided scratch), so results are bit-identical
        // to the allocating overload above. Does NOT engage the level-3 blocked kernel: solveInPlace
        // never forms Q, so the cache's blocked-WY buffers (Vpanel/Tbuf/Wbuf/tcolBuf/VfullBuf) are
        // simply unused here — see the scratch-contract note at the top of this class. Its win is
        // purely the eliminated per-call Allocator.Temp allocation of u and w.
        /// <param name="A">Destroyed; contents undefined after return (becomes R + stored reflectors scratch).</param>
        /// <param name="b">Destroyed; contents undefined after return (becomes Qᵀb scratch).</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo solveInPlace(ref doubleMxN A, ref doubleN b, ref doubleN x, ref doubleQRCache cache) {
            RequireQRWorkspace(in cache, A.M_Rows, A.N_Cols, needBlocked: false);
            return solveInPlace(ref A, ref b, ref x, ref cache.u, ref cache.w);
        }
    }
}
