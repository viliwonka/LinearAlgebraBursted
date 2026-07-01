#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class LQ {

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float sign(float x) => x < 0 ? (float)(-1) : (float)1;

        // Build a Householder reflector from ROW `row` of matrix M, columns colStart..N_Cols-1.
        // Stores result in v[colStart..N_Cols-1]; entries v[0..colStart-1] are not accessed.
        // Convention: G = I - v*vᵀ with ||v||² = 2 (same construction as Bidiag.genHouseholderRow).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void genHouseholderRow(ref floatMxN M, ref floatN v, int row, int colStart, float zeroThreshold)
        {
            int n = M.N_Cols;
            for (int c = colStart; c < n; c++)
                v[c] = M[row, c];

            float xNorm = floatNorms_OP.L2Range(v, colStart, n);

            if (math.abs(xNorm) > zeroThreshold)
            {
                for (int c = colStart; c < n; c++)
                    v[c] = v[c] / xNorm;
                v[colStart] = v[colStart] + sign(v[colStart]);
                float div = math.sqrt(math.abs(v[colStart]));
                for (int c = colStart; c < n; c++)
                    v[c] = v[c] / div;
            }
            else
            {
                v[colStart] = math.SQRT2;
            }
        }

        // Row-contraction dot with 4 independent accumulators. A right-multiply reflector update
        // needs Mv (a per-row reduction) — the awkward direction for row-major storage, unlike a
        // left-multiply's uᵀM (a sum of scaled rows, expressible as pure axpy — see QR/Bidiag's
        // applyReflectorRight/applyHouseholderLeft). A single running-sum reduction can't be
        // auto-vectorized under strict FloatMode (see docs/perf-vectorization-lessons.md); 4
        // independent accumulator chains restore ILP across the unrolled lanes. Measured: 4 wins
        // decisively over 8 (register pressure from 8 live accumulators regressed ~1.7x at N=1024
        // vs 4's ~3x win over the naive single-accumulator form — see benchmark-tallwide.txt history).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float dot4(float* a, float* b, int n)
        {
            float s0 = (float)0, s1 = (float)0, s2 = (float)0, s3 = (float)0;
            int i = 0;
            for (; i + 4 <= n; i += 4)
            {
                s0 += a[i] * b[i];
                s1 += a[i + 1] * b[i + 1];
                s2 += a[i + 2] * b[i + 2];
                s3 += a[i + 3] * b[i + 3];
            }
            float sum = (s0 + s1) + (s2 + s3);
            for (; i < n; i++)
                sum += a[i] * b[i];
            return sum;
        }

        // Apply Householder G = I - v*vᵀ from the RIGHT to M[rowStart:rowEnd, colStart:]:
        //   M[r, colStart:] -= (M[r, colStart:] · v[colStart:]) · v[colStart:]  for rowStart <= r < rowEnd
        // v[0..colStart-1] are treated as zero (not accessed). Per-row dot4 + axpy.
        //
        // rowEnd lets the blocked (compact-WY) factorization restrict the per-row reflector apply to
        // just its own panel (rows [d, p0+pb)) instead of the whole trailing matrix — the panel's
        // remaining rows [p0+pb, M) are updated once per PANEL as a block GEMM instead of once per
        // ROW; see lqDecompositionBlockedCore.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void applyHouseholderRightRows(ref floatMxN M, ref floatN v, int rowStart, int rowEnd, int colStart)
        {
            int cols = M.N_Cols;
            int L = cols - colStart;
            if (L <= 0) return;

            float* mp = M.Data.Ptr;
            float* vp = v.Data.Ptr + colStart;

            for (int r = rowStart; r < rowEnd; r++)
            {
                float* rowPtr = mp + (long)r * cols + colStart;
                float dot = dot4(rowPtr, vp, L);
                Unsafe_OP.axpy(rowPtr, vp, -dot, L);
            }
        }

        // Un-restricted form: applies to the full trailing block [rowStart, M_Rows). Used by every
        // path that has not been raised to the blocked (compact-WY) factorization — the zero-alloc
        // lqDecomposition overload and the unblocked lqKernel.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void applyHouseholderRight(ref floatMxN M, ref floatN v, int rowStart, int colStart)
        {
            applyHouseholderRightRows(ref M, ref v, rowStart, M.M_Rows, colStart);
        }

        // ---- LQ decomposition ----

        // Shared kernel: reduces the working copy W (m x n, already holding a copy of A) to
        // L (m x m lower-triangular) via m row-Householder reflectors, then reconstructs Q (m x n,
        // orthonormal rows) from those same reflectors applied in reverse. v is length-n scratch.
        //
        // Forward sweep (d = 0..m-1): build a reflector from row d, columns d..n-1, and apply it from
        // the right to W[d:, d:]. This zeroes row d's entries past column d and updates every row
        // below it — mirrors Bidiag's row-Householder step, but run to full completion each time
        // (colStart = d, not d+1) so every row ends up fully triangular, not merely bidiagonal.
        // The reflector for step d is stashed into W[d, d..n-1] (overwriting the now-redundant
        // [pivot, 0, ..., 0] row Householder leaves behind, after L's diagonal entry is captured)
        // for reuse by the backward pass below — avoids a second m x n scratch buffer.
        //
        // Backward pass (d = m-1..0): Q_lq = Q_qrᵀ = H_{m-1} · H_{m-2} ··· H_0 (Householder reflectors
        // are symmetric, so QR's left-multiply-then-transpose identity becomes a reverse-order
        // right-multiply here). Seed Q = [I_m | 0], then apply each stashed reflector from the right.
        // rowStart = d, matching the forward pass: row r < d still holds its untouched e_r seed (its
        // own reflector, colStart = r, hasn't been processed yet in this decreasing-d order), so it
        // contributes a provably-zero dot product — skip it rather than compute a guaranteed no-op.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void lqKernel(ref floatMxN W, ref floatMxN L, ref floatMxN Q, ref floatN v, float zeroThreshold)
        {
            int m = W.M_Rows;
            int n = W.N_Cols;

            for (int d = 0; d < m; d++)
            {
                genHouseholderRow(ref W, ref v, d, d, zeroThreshold);
                applyHouseholderRight(ref W, ref v, d, d);

                L[d, d] = W[d, d];
                for (int c = d; c < n; c++)
                    W[d, c] = v[c];
            }

            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < r; c++)
                    L[r, c] = W[r, c];
                for (int c = r + 1; c < m; c++)
                    L[r, c] = (float)0;
            }

            unsafe { UnsafeUtility.MemClear(Q.Data.Ptr, (long)m * n * UnsafeUtility.SizeOf<float>()); }
            for (int i = 0; i < m; i++)
                Q[i, i] = (float)1;

            // rowStart = d (not 0): row r < d still holds its untouched e_r seed at this point (its
            // own reflector, colStart = r > d in processing order d = m-1..0, hasn't run yet), so its
            // dot product against v[d:] is provably zero — including it would be a guaranteed no-op.
            for (int d = m - 1; d >= 0; d--)
            {
                for (int c = d; c < n; c++)
                    v[c] = W[d, c];
                applyHouseholderRight(ref Q, ref v, d, d);
            }
        }

        // Blocked (level-3 / compact-WY, GEMM trailing-update) factorization+reconstruction core.
        // τ≡1 convention throughout (see genHouseholderRow / applyHouseholderRight): each
        // G_i = I - v_i v_iᵀ, so the compact-WY T has T[i,i] = 1 (not LAPACK's τ-scaled diagonal).
        //
        // LQ right-multiplies (unlike QR's left-multiply), so panels are ROW blocks and the T-vs-Tᵀ
        // usage is FLIPPED relative to QR:
        //   factorization applies   C := C·(I - Vᵀ T V)   = C - (C·Vᵀ)·(T·V)     → wyTriMul   (T)
        //   reconstruction applies  Q := Q·(I - Vᵀ Tᵀ V)  = Q - (Q·Vᵀ)·(Tᵀ·V)    → wyTriTransMul (Tᵀ)
        // (QR was the opposite: factorization used Tᵀ, reconstruction used T — see QR's blocked core.)
        //
        // Here V is the pb×(n-p0) panel whose ROWS are the reflectors (v_i occupies local columns
        // c' >= i, masked to zero for c' < i); Vt is its (n-p0)×pb transpose, built once per panel so
        // both the Gram contraction (formT) and the C·Vᵀ folding step (Unsafe_OP.lqYeqCVt) can walk
        // it with a unit-stride inner loop. Vt MUST be built, and Y = C·Vᵀ computed, BEFORE the
        // in-place wyTriMul/wyTriTransMul overwrites Vpanel with T·V (or Tᵀ·V) — Vt is a separate,
        // untouched buffer, but Vpanel itself is the one that gets clobbered in place.
        //
        // Reflectors are stashed into W (the working copy of A, upper-right of each processed row);
        // Q is a SEPARATE m×n buffer seeded to [I_m | 0] before reconstruction — unlike QR, which
        // reconstructs in place over the same buffer that stores the reflectors, LQ needs no
        // clean-snapshot copy (nothing here overwrites W while reconstruction is still reading it).
        //
        // Scratch (all caller-provided, sized by the lqDecomposition(A,L,Q) allocating wrapper):
        //   v       length N_Cols        — Householder vector (per-row panel factor step).
        //   Vpanel  length LQ_BLOCK*N_Cols — clean contiguous panel (pb x (n-p0)), reused for
        //           factor+reconstruct; leading dimension == the CURRENT (n-p0), not N_Cols.
        //   Vt      length N_Cols*LQ_BLOCK — transpose of Vpanel ((n-p0) x pb), leading dim == pb.
        //   Tbuf    length LQ_BLOCK*LQ_BLOCK — compact-WY T, reused per panel.
        //   Y       length M_Rows*LQ_BLOCK — C·Vᵀ folding buffer AND formT's Gram scratch (reused
        //           sequentially within a panel: formT's G write happens before Y's C·Vᵀ write).
        //   tcolBuf length LQ_BLOCK      — formT scratch.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void lqDecompositionBlockedCore(ref floatMxN W, ref floatMxN L, ref floatMxN Q,
            ref floatN v, ref floatN Vpanel, ref floatN Vt, ref floatN Tbuf, ref floatN Y, ref floatN tcolBuf,
            float zeroThreshold)
        {
            // Panel width for the blocked (level-3 / compact-WY) factorization path. A method-local
            // const (not a class field) — LQ is a partial class shared by the float/double generated
            // files, so a class-level const of the same name would collide (CS0102). Matches QR_BLOCK.
            const int LQ_BLOCK = 64;

            int m = W.M_Rows;
            int n = W.N_Cols;

            float* Wp = W.Data.Ptr;
            float* Vp = Vpanel.Data.Ptr;
            float* Vtp = Vt.Data.Ptr;
            float* T = Tbuf.Data.Ptr;
            float* Yp = Y.Data.Ptr;
            float* tcol = tcolBuf.Data.Ptr;

            // ---- factorization: row panels top to bottom ----
            for (int p0 = 0; p0 < m; p0 += LQ_BLOCK)
            {
                int pb = math.min(LQ_BLOCK, m - p0);

                // (1) factor panel rows d in [p0, p0+pb); reflector apply restricted to the panel's
                //     OWN remaining rows [d, p0+pb) — rows beyond the panel are updated once below as
                //     a single block GEMM instead of once per row.
                for (int d = p0; d < p0 + pb; d++)
                {
                    genHouseholderRow(ref W, ref v, d, d, zeroThreshold);
                    applyHouseholderRightRows(ref W, ref v, d, p0 + pb, d);

                    L[d, d] = W[d, d];
                    for (int c = d; c < n; c++)
                        W[d, c] = v[c];
                }

                // (2) gather the clean panel V (pb x (n-p0)): local row i (global row p0+i), local
                //     col c' (global col p0+c'); masked to zero left of each reflector's own diagonal
                //     (c' < i).
                int cn = n - p0;
                for (int i = 0; i < pb; i++)
                {
                    int r = p0 + i;
                    float* Vrow = Vp + (long)i * cn;
                    float* Wrow = Wp + (long)r * n + p0;
                    for (int c = 0; c < cn; c++)
                        Vrow[c] = (c >= i) ? Wrow[c] : (float)0;
                }

                // (3) Vt = transpose(Vpanel): Vt[c'*pb + i] = Vpanel[i*cn + c']   ((n-p0) x pb).
                for (int i = 0; i < pb; i++)
                {
                    float* Vrow = Vp + (long)i * cn;
                    for (int c = 0; c < cn; c++)
                        Vtp[(long)c * pb + i] = Vrow[c];
                }

                // (4) form the pb x pb compact-WY T (τ≡1) from the panel, via its transpose Vt.
                Unsafe_OP.formT(Vtp, pb, cn, pb, T, tcol, Yp);

                // (5) trailing block update, rows [p0+pb, m): C := C·(I - Vᵀ T V) = C - Y·(T·V),
                //     where Y = C·Vᵀ. Y and Vt MUST be built/read before wyTriMul overwrites Vpanel.
                int rowsTrail = m - (p0 + pb);
                if (rowsTrail > 0)
                {
                    float* Cp = Wp + (long)(p0 + pb) * n + p0;
                    UnsafeUtility.MemClear(Yp, (long)rowsTrail * pb * UnsafeUtility.SizeOf<float>());
                    Unsafe_OP.lqYeqCVt(Cp, n, Vtp, cn, rowsTrail, pb, Yp);
                    Unsafe_OP.wyTriMul(T, pb, Vp, cn);                        // T — factorization direction
                    Unsafe_OP.wySubVW(Yp, pb, Cp, n, rowsTrail, pb, cn, Vp);
                }
            }

            // L extraction (unchanged from lqKernel).
            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < r; c++)
                    L[r, c] = W[r, c];
                for (int c = r + 1; c < m; c++)
                    L[r, c] = (float)0;
            }

            // ---- reconstruct Q from the stored reflectors, panels bottom to top ----
            // Q is a SEPARATE buffer from W, so no clean-snapshot copy is needed — W's stashed
            // reflector rows are never overwritten by this reconstruction (see file-header notes).
            UnsafeUtility.MemClear(Q.Data.Ptr, (long)m * n * UnsafeUtility.SizeOf<float>());
            for (int i = 0; i < m; i++)
                Q[i, i] = (float)1;

            float* Qp = Q.Data.Ptr;

            int lastP0 = ((m - 1) / LQ_BLOCK) * LQ_BLOCK;
            for (int p0 = lastP0; p0 >= 0; p0 -= LQ_BLOCK)
            {
                int pb = math.min(LQ_BLOCK, m - p0);
                int cn = n - p0;

                // Gather Vpanel from W (masked c' < i -> 0; same as factorization step 2).
                for (int i = 0; i < pb; i++)
                {
                    int r = p0 + i;
                    float* Vrow = Vp + (long)i * cn;
                    float* Wrow = Wp + (long)r * n + p0;
                    for (int c = 0; c < cn; c++)
                        Vrow[c] = (c >= i) ? Wrow[c] : (float)0;
                }

                // Build Vt and form T (same as factorization steps 3-4).
                for (int i = 0; i < pb; i++)
                {
                    float* Vrow = Vp + (long)i * cn;
                    for (int c = 0; c < cn; c++)
                        Vtp[(long)c * pb + i] = Vrow[c];
                }
                Unsafe_OP.formT(Vtp, pb, cn, pb, T, tcol, Yp);

                // Apply the block to Q rows [p0, m), cols [p0, n): Q := Q·(I - Vᵀ Tᵀ V)
                // = Q - Y·(Tᵀ·V). Row restriction [p0, m) is valid by the same e_t induction the
                // unblocked lqKernel already used (see its doc comment): rows < p0 are still their
                // seed e_r, whose dot with Vᵀ is provably zero.
                int rows = m - p0;
                float* Cp = Qp + (long)p0 * n + p0;
                UnsafeUtility.MemClear(Yp, (long)rows * pb * UnsafeUtility.SizeOf<float>());
                Unsafe_OP.lqYeqCVt(Cp, n, Vtp, cn, rows, pb, Yp);
                Unsafe_OP.wyTriTransMul(T, pb, Vp, cn);                       // Tᵀ — reconstruction direction
                Unsafe_OP.wySubVW(Yp, pb, Cp, n, rows, pb, cn, Vp);
            }
        }

        /// <summary>
        /// LQ decomposition of A (m × n, m ≤ n): A = L · Q where L is m × m lower-triangular
        /// and Q is m × n with orthonormal rows (Q Qᵀ = I_m). Direct row-Householder reduction
        /// (mirrors LAPACK GELQF): m reflectors, each built from a row and applied from the right,
        /// unit-stride in the column axis (cache-friendly; see dot4's doc comment for why the
        /// per-row reduction itself only gets ILP, not full SIMD, in this row-major layout).
        /// A is not modified. Allocates Allocator.Temp scratch internally.
        /// </summary>
        /// <param name="A">Input m × n matrix (m ≤ n). Not modified.</param>
        /// <param name="L">Output m × m lower-triangular factor (caller-allocated, m × m).</param>
        /// <param name="Q">Output m × n row-orthonormal factor (caller-allocated, m × n).</param>
        // Allocating wrapper: allocates scratch (Allocator.Temp) and delegates. This is the fast
        // path — it routes to the BLOCKED (level-3 / compact-WY) factorization core once M_Rows is
        // large enough to amortise the extra panel bookkeeping; smaller matrices fall back to the
        // plain rank-1 sweep (lqKernel), which has no panel/GEMM overhead and is already fast enough
        // at that size. The zero-alloc ws overload is NOT blocked — it keeps calling lqKernel to
        // preserve its workspace contract; only this allocating convenience wrapper (used by, e.g.,
        // the benchmark and lqMinNormSolve's allocating overload) gets the speedup.
        //
        // LQ_BLOCK_MIN_M is a measured (not derived) crossover, unlike QR's simple ">= 2*QR_BLOCK"
        // gate: LQ's fold step (Unsafe_OP.lqYeqCVt) is reduction-shaped rather than axpy-shaped (see
        // its doc comment), so its per-panel overhead amortises more slowly, and double's crossover
        // measured LATER than float's — a matrix that already pays off for float can still REGRESS
        // for double (measured: k=256 was a ~9% win for float but a ~20% loss for double). Since the
        // routing gate is shared by both generated types (this is a proxy template), it must be
        // conservative enough for the SLOWER-to-cross type. Measured on TallWideSolveBenchmark
        // (A is k x 2k): k=256 (4 row-panels) regressed for double; k=512 (8 row-panels) was a clear
        // win for both float (~30%) and double (~7%) at N=512, and both improved further at N=1024
        // (float ~39%, double ~19%) — so this gate gives every blocked size a verified improvement.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lqDecomposition(ref floatMxN A, ref floatMxN L, ref floatMxN Q)
        {
            // See lqDecompositionBlockedCore for why this is a method-local const, not a class field.
            const int LQ_BLOCK = 64;
            const int LQ_BLOCK_MIN_M = 512;

            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQ.lqDecomposition: A must be wide or square (M_Rows <= N_Cols)");
            if (L.M_Rows != m || L.N_Cols != m)
                throw new ArgumentException("LQ.lqDecomposition: L must be m x m");
            if (Q.M_Rows != m || Q.N_Cols != n)
                throw new ArgumentException("LQ.lqDecomposition: Q must be m x n");

            if (m == 0 || n == 0)
                return;

            var W = new floatMxN(m, n, Allocator.Temp, false);
            var v = new floatN(n, Allocator.Temp, false);

            W.Data.CopyFrom(A.Data);
            float zeroThreshold = Consts.floatZeroThreshold * floatNorms_OP.LInf(in A);

            if (m < LQ_BLOCK_MIN_M)
            {
                lqKernel(ref W, ref L, ref Q, ref v, zeroThreshold);
            }
            else
            {
                var Vpanel = new floatN(LQ_BLOCK * n, Allocator.Temp, true);
                var Vt = new floatN(n * LQ_BLOCK, Allocator.Temp, true);
                var Tbuf = new floatN(LQ_BLOCK * LQ_BLOCK, Allocator.Temp, true);
                var Y = new floatN(m * LQ_BLOCK, Allocator.Temp, true);
                var tcolBuf = new floatN(LQ_BLOCK, Allocator.Temp, true);

                lqDecompositionBlockedCore(ref W, ref L, ref Q, ref v, ref Vpanel, ref Vt, ref Tbuf, ref Y, ref tcolBuf, zeroThreshold);

                tcolBuf.Dispose();
                Y.Dispose();
                Tbuf.Dispose();
                Vt.Dispose();
                Vpanel.Dispose();
            }

            v.Dispose();
            W.Dispose();
        }

        /// <summary>
        /// lqDecomposition using a reusable workspace (Arena.floatLQ_WS(m, n)) — zero-alloc.
        /// Semantics identical to the allocating overload; see that one for full documentation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lqDecomposition(ref floatMxN A, ref floatMxN L, ref floatMxN Q, ref floatLQ_WS ws)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQ.lqDecomposition: A must be wide or square (M_Rows <= N_Cols)");
            if (L.M_Rows != m || L.N_Cols != m)
                throw new ArgumentException("LQ.lqDecomposition: L must be m x m");
            if (Q.M_Rows != m || Q.N_Cols != n)
                throw new ArgumentException("LQ.lqDecomposition: Q must be m x n");
            RequireLQWorkspace(in ws, m, n);

            if (m == 0 || n == 0)
                return;

            var W = ws.W;
            var v = ws.v;

            W.Data.CopyFrom(A.Data);
            float zeroThreshold = Consts.floatZeroThreshold * floatNorms_OP.LInf(in A);

            lqKernel(ref W, ref L, ref Q, ref v, zeroThreshold);
        }

        // ---- LQ minimum-norm solver ----

        /// <summary>
        /// Minimum-2-norm solution to the underdetermined system A x = b (m ≤ n, A full row rank).
        /// Uses LQ: A = L Q, so x = Qᵀ L⁻¹ b.
        /// Steps: (1) forward-solve L y = b for y (m-vector); (2) x = Qᵀ y (n-vector).
        /// A is not modified. Allocates Allocator.Temp for L, Q, and y.
        /// </summary>
        /// <param name="A">m × n coefficient matrix (m ≤ n, full row rank). Not modified.</param>
        /// <param name="b">Right-hand side vector, length m.</param>
        /// <param name="x">Solution output (min-2-norm), length n. Must not alias b.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lqMinNormSolve(ref floatMxN A, ref floatN b, ref floatN x)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQ.lqMinNormSolve: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQ.lqMinNormSolve: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQ.lqMinNormSolve: x.N must equal A.N_Cols");

            var L = new floatMxN(m, m, Allocator.Temp, false);
            var Q = new floatMxN(m, n, Allocator.Temp, false);

            lqDecomposition(ref A, ref L, ref Q);

            // Step 1: forward-solve L y = b.  y starts as a copy of b (solveLowerTriangular is in-place).
            var y = new floatN(m, Allocator.Temp, false);
            y.Data.CopyFrom(b.Data);
            Solvers.solveLowerTriangular(ref L, ref y);

            // Step 2: x = Qᵀ y.  dot(in y, in Q, ref x) computes yᵀ Q = (Qᵀ y)ᵀ → n-vector.
            Linear_OP.dot(in y, in Q, ref x);

            y.Dispose();
            Q.Dispose();
            L.Dispose();
        }

        /// <summary>
        /// lqMinNormSolve using a reusable workspace (Arena.floatLQMinNormSolve_WS(m, n)) —
        /// zero-alloc end to end (including the nested lqDecomposition call).
        /// Semantics identical to the allocating overload; see that one for full documentation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lqMinNormSolve(ref floatMxN A, ref floatN b, ref floatN x, ref floatLQMinNormSolve_WS ws)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQ.lqMinNormSolve: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQ.lqMinNormSolve: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQ.lqMinNormSolve: x.N must equal A.N_Cols");
            RequireLQMinNormSolveWorkspace(in ws, m, n);

            var L = ws.L;
            var Q = ws.Q;

            lqDecomposition(ref A, ref L, ref Q, ref ws.LQWs);

            // Step 1: forward-solve L y = b.  y starts as a copy of b (solveLowerTriangular is in-place).
            var y = ws.y;
            y.Data.CopyFrom(b.Data);
            Solvers.solveLowerTriangular(ref L, ref y);

            // Step 2: x = Qᵀ y.  dot(in y, in Q, ref x) computes yᵀ Q = (Qᵀ y)ᵀ → n-vector.
            Linear_OP.dot(in y, in Q, ref x);
        }
    }
}
