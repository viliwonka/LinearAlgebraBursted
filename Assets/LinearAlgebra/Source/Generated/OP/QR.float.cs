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

        static float sign(float x) {
            return x < 0 ? -1 : 1;
        }

        // zeroThreshold is the ABSOLUTE column-norm below which a column is treated as zero. Callers
        // pass a SCALE-RELATIVE value (Consts.floatZeroThreshold * matrix magnitude) so QR is
        // scale-invariant — a fixed absolute constant mis-classifies every column of a uniformly
        // tiny-magnitude matrix as a zero column and silently produces a garbage decomposition.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void genHouseholderPete(ref floatMxN Q, ref floatN u, int k, float zeroThreshold) {

            for (int r = k; r < u.N; r++)
                u[r] = Q[r, k];

            float xNorm = floatNorms_OP.L2Range(u, k, u.N);

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
        // Two contiguous-memory passes through the vectorising Unsafe_OP.axpy ([NoAlias]) — the same
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
        private static unsafe void applyReflectorRightCols(ref floatMxN Q, ref floatN u, ref floatN w, int d, int colEnd)
        {
            int M = Q.M_Rows;
            int N = Q.N_Cols;
            int L = colEnd - d;                     // width of the (restricted) trailing column block
            if (L <= 0)
                return;

            float* qp = Q.Data.Ptr;
            float* up = u.Data.Ptr;
            float* wp = w.Data.Ptr;

            // pass 1: w[0..L) = Σ_{r=d}^{M-1} u[r] · Q[r, d..colEnd)   (row segments are unit-stride)
            UnsafeUtility.MemClear(wp, (long)L * UnsafeUtility.SizeOf<float>());
            for (int r = d; r < M; r++)
                Unsafe_OP.axpy(wp, qp + (long)r * N + d, up[r], L);

            // pass 2: Q[r, d..colEnd) += (-u[r]) · w[0..L)  ==  Q[r, d..colEnd) -= u[r] · w
            for (int r = d; r < M; r++)
                Unsafe_OP.axpy(qp + (long)r * N + d, wp, -up[r], L);
        }

        // Un-restricted form: applies to the full trailing block [d, N_Cols). Used by every path
        // that has not been raised to the blocked (compact-WY) factorization — the zero-alloc
        // qrDecomposition overload, qrDecompositionColumnPivot, qrDirectSolve, and Q-reconstruction.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void applyReflectorRight(ref floatMxN Q, ref floatN u, ref floatN w, int d)
        {
            applyReflectorRightCols(ref Q, ref u, ref w, d, Q.N_Cols);
        }

        // Caller-provided scratch overload (zero-alloc): u is a workspace vector of length
        // EXACTLY Q.M_Rows; w is a workspace vector of length >= Q.N_Cols (the reflector-apply
        // accumulator). Hoist both out of a hot loop to skip the per-call Allocator.Temp allocs.
        // Always reports DirectSolveStatus.Success — this factorization has no failure mode (a
        // zero-norm column is handled via the sign-convention fallback in genHouseholderPete, not
        // rejected).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDecomposition(ref floatMxN Q, ref floatMxN R, ref floatN u, ref floatN w)
        {
            if (Q.M_Rows < Q.N_Cols)
                throw new ArgumentException("QR.qrDecomposition: Matrix R must be square or tall (more or equal rows than cols)");

            if (u.N != Q.M_Rows)
                throw new ArgumentException("QR.qrDecomposition: scratch vector u.N must equal Q.M_Rows");

            if (w.N < Q.N_Cols)
                throw new ArgumentException("QR.qrDecomposition: scratch vector w.N must be at least Q.N_Cols");

            int qrSteps = Q.N_Cols;

            // scale-relative zero-column threshold (see genHouseholderPete); LInf(Q) == max |entry|.
            float zeroThreshold = Consts.floatZeroThreshold * floatNorms_OP.LInf(in Q);

            for (int d = 0; d < qrSteps; d++)
            {
                genHouseholderPete(ref Q, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix: Q[d:, d:] -= u·(uᵀ·Q[d:, d:]).
                // Vectorised, zero-alloc (w is caller scratch). See applyReflectorRight.
                applyReflectorRight(ref Q, ref u, ref w, d);

                R[d, d] = Q[d, d];

                // copy v into Q below diagonal, will be used to reconstruct Q
                for (int i = d; i < Q.M_Rows; i++)
                {
                    Q[i, d] = u[i];
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
                    R[r, c] = Q[r, c];
                }
            }

            // Reconstruct Q from vectors stored inside Q columns

            // Initialize upper part of Q to identity matrix, including diagonals
            for (int r = 0; r < Q.M_Rows; r++)
            {
                for (int c = r; c < Q.N_Cols; c++)
                {
                    if (c > r)
                    {
                        Q[r, c] = 0;
                    }
                }
            }

            // Apply Householder transformations in reverse order
            // Reconstruct the Householder vector v from the original Q
            for (int d = Q.N_Cols - 1; d >= 0; d--)
            {
                // includes diagonal elements
                for (int i = d; i < Q.M_Rows; i++)
                {
                    u[i] = Q[i, d];
                    Q[i, d] = i == d? 1 : 0;
                }

                // Apply the reflector to the trailing columns: Q[d:, d:] -= u·(uᵀ·Q[d:, d:]).
                // Same vectorised, zero-alloc helper as the factorization apply above.
                applyReflectorRight(ref Q, ref u, ref w, d);
            }

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Blocked (level-3 / compact-WY, GEMM trailing-update) factorization core. τ≡1 convention
        // throughout (see file-header notes on genHouseholderPete / applyReflectorRight): each
        // H_i = I - u_i u_iᵀ, so the compact-WY T has T[i,i] = 1 (not LAPACK's τ-scaled diagonal).
        //
        // Panels of QR_BLOCK columns are factored with the existing rank-1 sweep (cheap — pb is
        // small), but their combined effect on the REST of the matrix is applied once per panel as
        // two GEMM-shaped passes (Unsafe_OP.wyVtC / wySubVW, unit-stride inner loop) instead of pb
        // separate rank-1 (applyReflectorRight) passes — the memory-traffic-bound part of the
        // algorithm. Reconstruction of Q is similarly batched, applying panels right-to-left.
        //
        // Direction matters and is easy to get backwards (see spec landmines):
        //   factorization applies (I - V T Vᵀ)ᵀ = I - V Tᵀ Vᵀ   → wyTriTransMul (Tᵀ)
        //   reconstruction applies  I - V T Vᵀ  (un-transposed)  → wyTriMul (T)
        //
        // Scratch (all caller-provided, sized by the qrDecomposition(Q,R) allocating wrapper):
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
        private static unsafe void qrDecompositionBlockedCore(ref floatMxN Q, ref floatMxN R,
            ref floatN u, ref floatN w,
            ref floatN Vpanel, ref floatN Tbuf, ref floatN Wbuf, ref floatN tcolBuf, ref floatN VfullBuf)
        {
            // Panel width for the blocked (level-3 / compact-WY) factorization path. 32 columns
            // keeps the panel (and its T factor) tiny relative to cache while still amortising the
            // trailing-update GEMM over enough columns to reach GEMM-shaped throughput. A method-
            // local const (not a class field) — QR is a partial class shared by the float/double
            // generated files, so a class-level const of the same name would collide (CS0102).
            const int QR_BLOCK = 32;

            int m = Q.M_Rows;
            int n = Q.N_Cols;

            // scale-relative zero-column threshold (see genHouseholderPete); LInf(Q) == max |entry|.
            float zeroThreshold = Consts.floatZeroThreshold * floatNorms_OP.LInf(in Q);

            float* Qp = Q.Data.Ptr;
            float* Vp = Vpanel.Data.Ptr;
            float* T = Tbuf.Data.Ptr;
            float* Wmat = Wbuf.Data.Ptr;
            float* tcol = tcolBuf.Data.Ptr;

            // ---- factorization: panels left to right ----
            for (int p0 = 0; p0 < n; p0 += QR_BLOCK)
            {
                int pb = math.min(QR_BLOCK, n - p0);

                // (1) factor panel columns d in [p0, p0+pb); reflector apply restricted to the
                //     panel's OWN remaining columns [d, p0+pb) — cols beyond the panel are updated
                //     once below as a single block GEMM instead of once per column.
                for (int d = p0; d < p0 + pb; d++)
                {
                    genHouseholderPete(ref Q, ref u, d, zeroThreshold);
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
                    float* Vrow = Vp + (long)t * pb;
                    for (int i = 0; i < pb; i++)
                        Vrow[i] = (t >= i) ? Qp[(long)r * n + (p0 + i)] : (float)0;
                }

                // (3) form the pb x pb compact-WY T (τ≡1) from the panel.
                Unsafe_OP.formT(Vp, pb, rows, pb, T, tcol, Wmat);

                // (4) trailing block update on cols [p0+pb, n): C -= V*(Tᵀ*(Vᵀ*C)). One untiled
                //     GEMM call per panel — Unsafe_OP.wyVtC/wySubVW already reach full GEMM
                //     throughput (~70 GFLOP/s, matched matMatDot) at this width without tiling;
                //     column-tiling was tried and measured SLOWER (added MemClear/call overhead
                //     for no cache-locality benefit), so it is deliberately not done here.
                int cStart = p0 + pb;
                int cw = n - cStart;
                if (cw > 0)
                {
                    float* Cp = Qp + (long)p0 * n + cStart;
                    UnsafeUtility.MemClear(Wmat, (long)pb * cw * UnsafeUtility.SizeOf<float>());
                    Unsafe_OP.wyVtC(Vp, pb, Cp, n, rows, pb, cw, Wmat);
                    Unsafe_OP.wyTriTransMul(T, pb, Wmat, cw);      // Tᵀ — factorization direction
                    Unsafe_OP.wySubVW(Vp, pb, Cp, n, rows, pb, cw, Wmat);
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

            // Snapshot the stored reflectors into a clean, masked (r >= c ? Q[r,c] : 0) full copy
            // BEFORE Q is overwritten below — reconstruction both reads V and writes Q in place.
            float* Vfull = VfullBuf.Data.Ptr;
            for (int r = 0; r < m; r++)
            {
                float* qrow = Qp + (long)r * n;
                float* vrow = Vfull + (long)r * n;
                for (int c = 0; c < n; c++)
                    vrow[c] = (r >= c) ? qrow[c] : (float)0;
            }

            // Seed Q = [I_n; 0] (m x n).
            UnsafeUtility.MemClear(Qp, (long)m * n * UnsafeUtility.SizeOf<float>());
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
                    float* Vrow = Vp + (long)t * pb;
                    float* Vfrow = Vfull + (long)r * n;
                    for (int i = 0; i < pb; i++)
                        Vrow[i] = Vfrow[p0 + i];
                }

                Unsafe_OP.formT(Vp, pb, rows, pb, T, tcol, Wmat);

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
                float* Cp = Qp + (long)p0 * n + p0;
                UnsafeUtility.MemClear(Wmat, (long)pb * cw * UnsafeUtility.SizeOf<float>());
                Unsafe_OP.wyVtC(Vp, pb, Cp, n, rows, pb, cw, Wmat);
                Unsafe_OP.wyTriMul(T, pb, Wmat, cw);               // T — reconstruction direction
                Unsafe_OP.wySubVW(Vp, pb, Cp, n, rows, pb, cw, Wmat);
            }
        }

        // Back-compat workspace overload: takes only the u scratch (length Q.M_Rows) and allocates
        // the small w accumulator (length Q.N_Cols) from Allocator.Temp. Behaviour is identical to
        // the 4-arg primitive; use that one to be fully zero-alloc in a hot loop.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDecomposition(ref floatMxN Q, ref floatMxN R, ref floatN u)
        {
            var w = new floatN(Q.N_Cols, Allocator.Temp, false);
            var info = qrDecomposition(ref Q, ref R, ref u, ref w);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDecomposition(ref floatMxN Q, ref floatMxN R)
        {
            // See qrDecompositionBlockedCore for why this is a method-local const, not a class field.
            const int QR_BLOCK = 32;

            if (Q.M_Rows < Q.N_Cols)
                throw new ArgumentException("QR.qrDecomposition: Matrix R must be square or tall (more or equal rows than cols)");

            if (Q.N_Cols < 2 * QR_BLOCK)
            {
                var uSmall = new floatN(Q.M_Rows, Allocator.Temp, false);
                var wSmall = new floatN(Q.N_Cols, Allocator.Temp, false);
                var infoSmall = qrDecomposition(ref Q, ref R, ref uSmall, ref wSmall);
                wSmall.Dispose();
                uSmall.Dispose();
                return infoSmall;
            }

            int m = Q.M_Rows;
            int n = Q.N_Cols;

            var u = new floatN(m, Allocator.Temp, true);
            var w = new floatN(n, Allocator.Temp, true);
            var Vpanel = new floatN(m * QR_BLOCK, Allocator.Temp, true);
            var Tbuf = new floatN(QR_BLOCK * QR_BLOCK, Allocator.Temp, true);
            var Wbuf = new floatN(QR_BLOCK * n, Allocator.Temp, true);
            var tcolBuf = new floatN(QR_BLOCK, Allocator.Temp, true);
            var VfullBuf = new floatN(m * n, Allocator.Temp, true);

            qrDecompositionBlockedCore(ref Q, ref R, ref u, ref w, ref Vpanel, ref Tbuf, ref Wbuf, ref tcolBuf, ref VfullBuf);

            VfullBuf.Dispose();
            tcolBuf.Dispose();
            Wbuf.Dispose();
            Tbuf.Dispose();
            Vpanel.Dispose();
            w.Dispose();
            u.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Column-pivoted (rank-revealing) QR — Businger–Golub. Factorizes A·P = Q·R, where the
        // column permutation P is chosen greedily so the pivot at each step is the trailing column
        // of largest 2-norm. This forces the magnitudes of the R diagonal to be non-increasing
        // (|R[0,0]| >= |R[1,1]| >= ... >= |R[n-1,n-1]|), so trailing near-zero diagonal entries
        // reveal the numerical rank — the stable choice for rank-deficient least squares where the
        // plain (un-pivoted) qrDecomposition above requires full column rank.
        //
        //   Q  in:  A (m x n, m >= n)              out: orthogonal Q (m x n)
        //   R  out: upper triangular R (n x n)
        //   P  out: column Pivot, size n. Reset internally. Result column j is original column P[j];
        //           equivalently A[:, P[j]] == (Q*R)[:, j].
        //   u  scratch Householder vector, length EXACTLY Q.M_Rows.
        //
        // Partial column norms are recomputed exactly at each step (rows d..m-1) rather than
        // downdated. That is the same O(n^2 m) order as the reflector sweep itself, and it sidesteps
        // the catastrophic-cancellation failure mode of norm downdating (LAPACK xGEQPF needs a
        // recompute guard precisely because the cheap downdate loses all accuracy near rank
        // deficiency) — for the modest matrices this library targets, exact recompute is both
        // simpler and unconditionally robust.
        // Always reports DirectSolveStatus.Success — this factorization has no failure mode; it does
        // NOT itself compute an integer rank (see qrcpDirectSolve for the rank-revealing consumer).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDecompositionColumnPivot(ref floatMxN Q, ref floatMxN R, ref Pivot P, ref floatN u)
        {
            if (Q.M_Rows < Q.N_Cols)
                throw new ArgumentException("QR.qrDecompositionColumnPivot: Matrix must be square or tall (M_Rows >= N_Cols)");

            if (u.N != Q.M_Rows)
                throw new ArgumentException("QR.qrDecompositionColumnPivot: scratch vector u.N must equal Q.M_Rows");

            if (P.N != Q.N_Cols)
                throw new ArgumentException("QR.qrDecompositionColumnPivot: pivot P.N must equal Q.N_Cols");

            if (R.M_Rows != Q.N_Cols || R.N_Cols != Q.N_Cols)
                throw new ArgumentException("QR.qrDecompositionColumnPivot: R must be N_Cols x N_Cols");

            P.Reset();

            int m = Q.M_Rows;
            int n = Q.N_Cols;

            // Reflector-apply accumulator (length n) + per-column squared-norm buffer for pivoting.
            // Allocated once per call (O(n) « O(n³)); this path has no zero-alloc w contract, unlike
            // qrDecomposition's 4-arg overload.
            var w = new floatN(n, Allocator.Temp, false);
            var colNorm2 = new floatN(n, Allocator.Temp, false);

            // scale-relative zero-column threshold (see genHouseholderPete); LInf(Q) == max |entry|.
            float zeroThreshold = Consts.floatZeroThreshold * floatNorms_OP.LInf(in Q);

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
                    float* qp = Q.Data.Ptr;
                    float* cn = colNorm2.Data.Ptr;
                    int L = n - d;
                    UnsafeUtility.MemClear(cn + d, (long)L * UnsafeUtility.SizeOf<float>());
                    for (int r = d; r < m; r++)
                        Unsafe_OP.addSquares(cn + d, qp + (long)r * n + d, L);
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
                    Swap_OP.Columns(ref Q, d, pivotCol);
                    P.Swap(d, pivotCol);
                }

                genHouseholderPete(ref Q, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix (vectorised, see applyReflectorRight).
                applyReflectorRight(ref Q, ref u, ref w, d);

                // R[d,d] and the stored Householder vector — see qrDecomposition (same pattern).
                R[d, d] = Q[d, d];

                for (int i = d; i < m; i++)
                    Q[i, d] = u[i];
            }

            // Copy the upper triangular part of Q into R
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                    R[r, c] = 0;
                else if (c > r)
                    R[r, c] = Q[r, c];
            }

            // Reconstruct Q from the Householder vectors stored in its columns (identical to the
            // un-pivoted qrDecomposition: pivoting only reordered the columns, not this step).
            for (int r = 0; r < m; r++)
                for (int c = r; c < n; c++)
                    if (c > r)
                        Q[r, c] = 0;

            for (int d = n - 1; d >= 0; d--)
            {
                for (int i = d; i < m; i++)
                {
                    u[i] = Q[i, d];
                    Q[i, d] = i == d ? 1 : 0;
                }

                // Apply the reflector to the trailing columns (vectorised, see applyReflectorRight).
                applyReflectorRight(ref Q, ref u, ref w, d);
            }

            colNorm2.Dispose();
            w.Dispose();

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates.
        // The caller still owns P (its size carries the column count and it is reset internally).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDecompositionColumnPivot(ref floatMxN Q, ref floatMxN R, ref Pivot P)
        {
            var u = new floatN(Q.M_Rows, Allocator.Temp, false);
            var info = qrDecompositionColumnPivot(ref Q, ref R, ref P, ref u);
            u.Dispose();
            return info;
        }

        // b is transformed into y = Q^T b, then solved for x; Q and b get modified (destroyed).
        // PRECONDITION: A has FULL COLUMN RANK. This un-pivoted solve back-substitutes through R's
        // diagonal; a rank-deficient A produces a zero on that diagonal and the result x is then
        // Inf/NaN (no guard). For rank-deficient / least-norm problems use the rank-revealing paths
        // instead: QR.qrDecompositionColumnPivot (QRCP), SVD.pinvSolve, or
        // Cholesky.choleskyPivotSolve.
        // Caller-provided scratch overload (zero-alloc): u is a workspace vector of length
        // EXACTLY A.M_Rows. Hoist u out of a hot loop to skip the per-call Allocator.Temp alloc.
        // Always reports DirectSolveStatus.Success — see the PRECONDITION note above: a
        // rank-deficient A silently divides by a zero R diagonal instead of being detected here.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDirectSolve(ref floatMxN A, ref floatN b, ref floatN x, ref floatN u) {
            if (A.M_Rows < A.N_Cols)
                throw new ArgumentException("QR.qrDirectSolve: Matrix A must be square or tall (more or equal rows than cols)");

            if (b.N != A.M_Rows)
                throw new ArgumentException("QR.qrDirectSolve: b.N must equal A.M_Rows");

            if (x.N != A.N_Cols)
                throw new ArgumentException("QR.qrDirectSolve: x.N must equal A.N_Cols");

            if (u.N != A.M_Rows)
                throw new ArgumentException("QR.qrDirectSolve: scratch vector u.N must equal A.M_Rows");

            int qrSteps = A.N_Cols;

            // Reflector-apply accumulator (length N_Cols). Allocated once per call (O(n) « O(n³)).
            var w = new floatN(A.N_Cols, Allocator.Temp, false);

            // scale-relative zero-column threshold (see genHouseholderPete); LInf(A) == max |entry|.
            float zeroThreshold = Consts.floatZeroThreshold * floatNorms_OP.LInf(in A);

            float dotProduct = 0;
            for (int d = 0; d < qrSteps; d++) {

                genHouseholderPete(ref A, ref u, d, zeroThreshold);

                // Apply the reflector to the trailing submatrix (vectorised, see applyReflectorRight).
                applyReflectorRight(ref A, ref u, ref w, d);

                // apply same transformation to b vector (O(n) — left scalar)
                dotProduct = 0;
                for (int r = d; r < A.M_Rows; r++)
                    dotProduct += u[r] * b[r];

                for (int r = d; r < A.M_Rows; r++)
                    b[r] -= u[r] * dotProduct;
            }

            w.Dispose();

            // copy b into x (x may be smaller dimension than b)
            for (int r = 0; r < A.N_Cols; r++)
                x[r] = b[r];

            // b was transformed to y, where y = Q^T b
            // Solve Rx = y

            return Solvers.solveUpperTriangular(ref A, ref x);
        }

        // Allocating wrapper: allocates the scratch vector u (Allocator.Temp) and delegates.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DirectSolveInfo qrDirectSolve(ref floatMxN A, ref floatN b, ref floatN x) {
            var u = new floatN(A.M_Rows, Allocator.Temp, false);
            var info = qrDirectSolve(ref A, ref b, ref x, ref u);
            u.Dispose();
            return info;
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
        /// </summary>
        /// <param name="A">m x n matrix (m >= n). Not modified (copied into Q scratch).</param>
        /// <param name="b">Right-hand side, length m. Must not alias x.</param>
        /// <param name="x">Solution, length n.</param>
        /// <param name="Q">Scratch: m x n (receives orthogonal factor; consumed).</param>
        /// <param name="R">Scratch: n x n (receives upper-triangular factor; consumed).</param>
        /// <param name="P">Scratch: column Pivot of size n (reset internally).</param>
        /// <param name="u">Scratch: length EXACTLY m (Householder workspace; first n entries are
        /// repurposed for the un-permute scatter after the decomposition).</param>
        /// <param name="relTol">Rank threshold ratio; tol = relTol * |R[0,0]|. Negative = auto default.</param>
        /// <returns>Status Success (r == n, full rank) or RankDeficient (r &lt; n, still a usable
        /// truncated least-squares solution); rank = detected r. See
        /// <see cref="RankRevealingInfo.Solved"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankRevealingInfo qrcpDirectSolve(ref floatMxN A, ref floatN b, ref floatN x,
                                           ref floatMxN Q, ref floatMxN R, ref Pivot P,
                                           ref floatN u, float relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m < n)
                throw new ArgumentException("QR.qrcpDirectSolve: A must be square or tall (M_Rows >= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("QR.qrcpDirectSolve: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("QR.qrcpDirectSolve: x.N must equal A.N_Cols");
            if (Q.M_Rows != m || Q.N_Cols != n)
                throw new ArgumentException("QR.qrcpDirectSolve: Q must be M_Rows x N_Cols");
            if (R.M_Rows != n || R.N_Cols != n)
                throw new ArgumentException("QR.qrcpDirectSolve: R must be N_Cols x N_Cols");
            if (P.N != n)
                throw new ArgumentException("QR.qrcpDirectSolve: P.N must equal A.N_Cols");
            if (u.N != m)
                throw new ArgumentException("QR.qrcpDirectSolve: u.N must equal A.M_Rows");

            // Negative relTol is an "auto" sentinel: use the library-standard rank threshold
            // (same default as SVD.pinvSolve / MatrixMetrics.rank). This also makes the threshold
            // divide-safe (tol >= 0), so a stray negative can never inflate rank into a divide-by-tiny.
            if (relTol < (float)0)
                relTol = (float)(math.max(m, n)) * Consts.floatZeroThreshold;

            // Degenerate: zero-column system.
            if (n == 0) return new RankRevealingInfo { status = DirectSolveStatus.Success, rank = 0 };

            // Step 1: copy A into Q (qrDecompositionColumnPivot destroys its input).
            Q.Data.CopyFrom(A.Data);

            // Step 2: QRCP — A·P = Q·R. P is reset and built inside this call.
            qrDecompositionColumnPivot(ref Q, ref R, ref P, ref u);

            // Step 3: determine numerical rank r from R's non-increasing diagonal.
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

            // Step 4: zero matrix (rank == 0) → x = 0, done.
            if (rank == 0)
            {
                for (int j = 0; j < n; j++)
                    x[j] = (float)0;
                return new RankRevealingInfo { status = DirectSolveStatus.RankDeficient, rank = 0 };
            }

            int r = rank;

            // Step 5: form c = Qᵀ b into x.
            // dot(in b, in Q, ref x) computes x[j] = Σ_i Q[i,j]·b[i] = (Qᵀb)[j].
            // dot zeroes x via MemClear before accumulating, so x needs no prior initialisation.
            // Guard: x must not alias b (enforced inside dot by pointer comparison).
            Linear_OP.dot(in b, in Q, ref x);

            // Step 6: back-solve the leading r×r block of R in place.
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

            // Step 7: un-permute — scatter from permuted ordering back to original column ordering.
            // QRCP gives A·P = Q·R where P[j] = original column index promoted to position j.
            // The permuted solution z (in x) satisfies: x_final[P[j]] = z[j].
            // Borrow u[0..n-1] as scatter scratch (u is no longer needed after Step 2).
            for (int j = 0; j < n; j++)
                u[j] = x[j];
            for (int j = 0; j < n; j++)
                x[P[j]] = u[j];

            return new RankRevealingInfo
            {
                status = (r < n) ? DirectSolveStatus.RankDeficient : DirectSolveStatus.Success,
                rank = r
            };
        }

        // Default-tolerance overload: passes the auto sentinel (relTol < 0), so the primitive
        // uses max(m,n) * Consts.floatZeroThreshold (consistent with SVD.pinvSolve / MatrixMetrics.rank).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankRevealingInfo qrcpDirectSolve(ref floatMxN A, ref floatN b, ref floatN x,
                                           ref floatMxN Q, ref floatMxN R, ref Pivot P,
                                           ref floatN u)
        {
            return qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, (float)(-1));
        }

        /// <summary>
        /// Allocating convenience wrapper: allocates Q (m×n), R (n×n), P (n-Pivot) and u (m)
        /// from Allocator.Temp and delegates to the zero-alloc primitive. Use the primitive in
        /// hot loops to avoid repeated Temp allocs.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankRevealingInfo qrcpDirectSolve(ref floatMxN A, ref floatN b, ref floatN x,
                                           float relTol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            var Q = new floatMxN(m, n, Allocator.Temp, false);
            var R = new floatMxN(n, n, Allocator.Temp, false);
            var P = new Pivot(n, Allocator.Temp);
            var u = new floatN(m, Allocator.Temp, false);
            var info = qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, relTol);
            u.Dispose();
            P.Dispose();
            R.Dispose();
            Q.Dispose();
            return info;
        }

        /// <summary>
        /// Allocating convenience wrapper with default tolerance (max(m,n) * Consts.floatZeroThreshold,
        /// matching SVD.pinvSolve / MatrixMetrics.rank).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static RankRevealingInfo qrcpDirectSolve(ref floatMxN A, ref floatN b, ref floatN x)
        {
            return qrcpDirectSolve(ref A, ref b, ref x, (float)(-1));
        }
    }
}
