#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class LQ {

        // ---- LQ decomposition ----

        /// <summary>
        /// LQ decomposition of A (m × n, m ≤ n): A = L · Q where L is m × m lower-triangular
        /// and Q is m × n with orthonormal rows (Q Qᵀ = I_m). Implemented via the
        /// transpose-of-QR identity: QR(Aᵀ) = Q_qr · R_qr gives A = R_qrᵀ · Q_qrᵀ, so
        /// L = R_qrᵀ (lower-tri) and Q = Q_qrᵀ (orthonormal rows).
        /// A is not modified. Allocates Allocator.Temp scratch internally.
        /// </summary>
        /// <param name="A">Input m × n matrix (m ≤ n). Not modified.</param>
        /// <param name="L">Output m × m lower-triangular factor (caller-allocated, m × m).</param>
        /// <param name="Q">Output m × n row-orthonormal factor (caller-allocated, m × n).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void lqDecomposition(ref fProxyMxN A, ref fProxyMxN L, ref fProxyMxN Q)
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

            // T = Aᵀ (n × m). Since n >= m, T satisfies M_Rows >= N_Cols for QR.
            var T   = new fProxyMxN(n, m, Allocator.Temp, false);
            var Rqr = new fProxyMxN(m, m, Allocator.Temp, false);

            Linear_OP.trans(in A, ref T);

            // QR(T): destroys T → Q_qr (n × m, orthonormal columns); fills Rqr (m × m, upper-tri).
            QR.qrDecomposition(ref T, ref Rqr);

            // L = Rqrᵀ  (m × m, lower-triangular).
            Linear_OP.trans(in Rqr, ref L);

            // Q_lq = Q_qrᵀ = Tᵀ  (m × n, orthonormal rows).
            Linear_OP.trans(in T, ref Q);

            Rqr.Dispose();
            T.Dispose();
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
        public static void lqMinNormSolve(ref fProxyMxN A, ref fProxyN b, ref fProxyN x)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            if (m > n)
                throw new ArgumentException("LQ.lqMinNormSolve: A must be wide or square (M_Rows <= N_Cols)");
            if (b.N != m)
                throw new ArgumentException("LQ.lqMinNormSolve: b.N must equal A.M_Rows");
            if (x.N != n)
                throw new ArgumentException("LQ.lqMinNormSolve: x.N must equal A.N_Cols");

            var L = new fProxyMxN(m, m, Allocator.Temp, false);
            var Q = new fProxyMxN(m, n, Allocator.Temp, false);

            lqDecomposition(ref A, ref L, ref Q);

            // Step 1: forward-solve L y = b.  y starts as a copy of b (solveLowerTriangular is in-place).
            var y = new fProxyN(m, Allocator.Temp, false);
            y.Data.CopyFrom(b.Data);
            Solvers.solveLowerTriangular(ref L, ref y);

            // Step 2: x = Qᵀ y.  dot(in y, in Q, ref x) computes yᵀ Q = (Qᵀ y)ᵀ → n-vector.
            Linear_OP.dot(in y, in Q, ref x);

            y.Dispose();
            Q.Dispose();
            L.Dispose();
        }
    }
}
