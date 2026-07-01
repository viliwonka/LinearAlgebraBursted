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

        // Apply Householder G = I - v*vᵀ from the RIGHT to M[rowStart:, colStart:]:
        //   M[r, colStart:] -= (M[r, colStart:] · v[colStart:]) · v[colStart:]  for each r >= rowStart
        // v[0..colStart-1] are treated as zero (not accessed). Per-row dot4 + axpy.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void applyHouseholderRight(ref floatMxN M, ref floatN v, int rowStart, int colStart)
        {
            int rows = M.M_Rows;
            int cols = M.N_Cols;
            int L = cols - colStart;
            if (L <= 0) return;

            float* mp = M.Data.Ptr;
            float* vp = v.Data.Ptr + colStart;

            for (int r = rowStart; r < rows; r++)
            {
                float* rowPtr = mp + (long)r * cols + colStart;
                float dot = dot4(rowPtr, vp, L);
                Unsafe_OP.axpy(rowPtr, vp, -dot, L);
            }
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lqDecomposition(ref floatMxN A, ref floatMxN L, ref floatMxN Q)
        {
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

            lqKernel(ref W, ref L, ref Q, ref v, zeroThreshold);

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
