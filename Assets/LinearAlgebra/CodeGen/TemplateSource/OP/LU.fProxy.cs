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
    /// Inpl = inplace
    /// </summary>
    public static partial class LU {

        // LU decomposition with no pivoting
        /// <summary>
        /// U = A (input matrix, overwritten with upper triangular U)
        /// L = I (identity matrix, overwritten with lower triangular L)
        /// A = L * U
        /// Returns true on success; false if a zero pivot is encountered (singular matrix).
        /// On false: no NaN/Inf is written.
        /// </summary>
        public static bool luDecompositionNoPivot(ref fProxyMxN U, ref fProxyMxN L)
        {
            if (!U.IsSquare)
                throw new System.ArgumentException("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.ArgumentException("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.ArgumentException("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;

            if (m == 0) return true;

            for(int k = 0; k < m - 1; k++) {

                // Calculate L and U
                fProxy Ukk = U[k, k];

                if (Ukk == 0)
                    return false;

                for(int j = k + 1; j < m; j++) {

                    fProxy Ljk = U[j, k] / Ukk;

                    L[j, k] = Ljk;

                    for (int i = k + 1; i < m; i++) {
                        U[j, i] -= Ljk * U[k, i];
                    }

                    // U is exactly upper-triangular
                    U[j, k] = 0;
                }
            }

            // Check last diagonal
            if (U[m - 1, m - 1] == 0)
                return false;

            return true;
        }

        // PA = L * U
        // U is originally A
        // L is originally I
        // P is pivot, that is reset, and is modified in place
        /// <summary>
        /// Performs LU decomposition with partial pivoting.
        /// Returns true on success; false if a zero pivot is encountered (singular matrix).
        /// On false: no NaN/Inf is written, P remains a valid permutation.
        /// </summary>
        public static bool luDecomposition(ref fProxyMxN U, ref fProxyMxN L, ref Pivot P) {
            if (!U.IsSquare)
                throw new System.ArgumentException("luDecomposition: U (A) needs to be square");

            if (!L.IsSquare)
                throw new System.ArgumentException("luDecomposition: L needs to be square");

            if (U.M_Rows != L.M_Rows)
                throw new System.ArgumentException("luDecomposition: U and L need to have the same dimensions");

            int m = U.M_Rows;

            if (P.N != m) throw new System.ArgumentException("pivot size must equal matrix dimension");

            P.Reset();

            if (m == 0) return true;

            // Panel width for the blocked (level-3) path. Method-local const — LU is a partial class
            // shared by the float/double generated files, so a class-level const of the same name
            // would collide across them (CS0102; see QR_BLOCK / CHOL_BLOCK).
            const int LU_BLOCK = 32;

            // Size gate: MEASURED crossover, not the naive 4*LU_BLOCK (see docs/level3-blocking-guide.md
            // "size gate" — Cholesky needed the same kind of margin, CHOL_BLOCK=32 but gate n>=256, i.e.
            // 8x the block width). Benchmarked: n=128 (4 panels) is a wash/slightly slower for float —
            // the panel/TRSM/GEMM bookkeeping isn't amortised yet — while n=256 (8 panels) is the first
            // size that clearly wins for both float and double. Below this, the plain per-column sweep
            // is used unchanged.
            const int LU_BLOCK_MIN_N = 8 * LU_BLOCK;

            if (m < LU_BLOCK_MIN_N) {
                // Small matrix: plain right-looking rank-1 sweep with partial pivoting, unchanged.
                for (int k = 0; k < m - 1; k++) {

                    int pivotIndex = k;
                    fProxy pivotValue = math.abs(U[k, k]);

                    // Find largest pivot in rows
                    for(int r = k + 1; r < m; r++) {
                        fProxy absValue = math.abs(U[r, k]);
                        if(absValue > pivotValue) {
                            pivotIndex = r;
                            pivotValue = absValue;
                        }
                    }

                    // Check for zero pivot before any division
                    if (pivotValue == 0)
                        return false;

                    // Swap rows
                    P.Swap(k, pivotIndex);

                    // swap submatrix U rows
                    Swap_OP.Rows(ref U, k, pivotIndex, k, m);

                    // swap already calculated L rows
                    Swap_OP.Rows(ref L, k, pivotIndex, 0, k);

                    // Calculate L and U. The trailing-row elimination U[j, k+1:] -= Ljk * U[k, k+1:] is
                    // an axpy over two DISTINCT rows (j > k) along the unit-stride column axis; routed
                    // through the vectorising Unsafe_OP.axpy ([NoAlias], the GEMM pointer path) so Burst
                    // SIMD-vectorises this O(n^3) hot loop (float ~2x double). Bitwise identical to the
                    // scalar form: each column i is updated independently, and (-Ljk)*U[k,i] added to
                    // U[j,i] equals U[j,i] - Ljk*U[k,i] exactly in IEEE.
                    fProxy Ukk = U[k, k];
                    unsafe
                    {
                        fProxy* up = U.Data.Ptr;
                        fProxy* rowK = up + (long)k * m;
                        int len = m - (k + 1);
                        for (int j = k + 1; j < m; j++) {

                            fProxy Ljk = U[j, k] / Ukk;

                            L[j, k] = Ljk;

                            fProxy* rowJ = up + (long)j * m;
                            Unsafe_OP.axpy(rowJ + (k + 1), rowK + (k + 1), -Ljk, len);

                            // U is exactly upper-triangular
                            U[j, k] = 0;
                        }
                    }
                }

                // Check last diagonal
                if (U[m - 1, m - 1] == 0)
                    return false;

                return true;
            }

            // ---- blocked (level-3) path — LAPACK-style right-looking GETRF ----
            // Each LU_BLOCK-wide panel is factored with the SAME rank-1 sweep as the small-matrix
            // path above (partial pivoting over the FULL remaining column height, so the pivot
            // sequence is bit-identical to the unblocked form — see "why the pivot sequence stays
            // identical" below, and docs/level3-blocking-guide.md recipe B), but its elimination axpy
            // is narrowed to the panel's own columns (DGETF2-style). The panel's contribution to the
            // columns to its right is then applied ONCE per panel as a level-3 TRSM (U12 = L11^-1 *
            // A12, unit-lower forward substitution) followed by a single GEMM trailing update
            // (A22 -= L21*U12, via Unsafe_OP.wySubVW) instead of one rank-1 update per panel column —
            // trading O(n) re-streams of the trailing matrix for O(n/LU_BLOCK), the memory-bandwidth-
            // bound part of the algorithm.
            //
            // WHY THE PIVOT SEQUENCE STAYS IDENTICAL: at panel step k, column k has already been
            // fully updated by (a) every earlier panel's trailing GEMM, which updates the WHOLE
            // trailing block (all columns from that panel's right edge through m, not just the next
            // panel) and (b) the within-panel eliminations for this panel's own earlier columns. So
            // the max-|abs| search over rows [k,m) sees exactly the values the unblocked algorithm
            // would see at the same step -> identical pivot, identical P. L and U differ from the
            // unblocked result only by GEMM summation-order rounding.
            //
            // LAST COLUMN: the unblocked sweep above never pivots column m-1 (its k loop stops at
            // m-2) and only diagonal-checks it at the end. The panel loop below reproduces this by
            // capping its own k-loop at min(panelEnd, m-1) — column m-1 is only ever ELIMINATED (via
            // within-panel axpy, when a panel's own width reaches exactly to m, or via an earlier
            // panel's TRSM+GEMM trailing update otherwise), never pivot-SEARCHED. The final
            // `if (U[m-1,m-1]==0)` check after the loop matches the unblocked form exactly.
            unsafe
            {
                fProxy* up = U.Data.Ptr;
                fProxy* lp = L.Data.Ptr;

                // Ubuf: contiguous copy of the strided U12 panel block (kb x ntrail, row stride
                // ntrail), sized for the worst case (first panel, k0=0: kb=LU_BLOCK, ntrail<=m).
                // wySubVW's W operand must be contiguous; U12 lives strided in U (leading dim m).
                var Ubuf = new fProxyN(LU_BLOCK * m, Allocator.Temp, false);
                fProxy* ubufp = Ubuf.Data.Ptr;

                for (int k0 = 0; k0 < m - 1; k0 += LU_BLOCK) {

                    int kb = math.min(LU_BLOCK, m - k0);
                    int panelEnd = k0 + kb;
                    // Never pivot-search/select-as-pivot column m-1 (matches unblocked's k<m-1 bound).
                    int kMax = math.min(panelEnd, m - 1);

                    // (1) PANEL FACTOR columns [k0, kMax) with partial pivoting over the FULL column
                    //     height [k,m) — same rank-1 sweep as the small-matrix path, just narrowed to
                    //     this panel's own columns for the elimination width.
                    for (int k = k0; k < kMax; k++) {

                        int pivotIndex = k;
                        fProxy pivotValue = math.abs(U[k, k]);

                        for (int r = k + 1; r < m; r++) {
                            fProxy absValue = math.abs(U[r, k]);
                            if (absValue > pivotValue) {
                                pivotIndex = r;
                                pivotValue = absValue;
                            }
                        }

                        // Check for zero pivot before any division
                        if (pivotValue == 0) {
                            Ubuf.Dispose();
                            return false;
                        }

                        // Swap rows
                        P.Swap(k, pivotIndex);

                        // swap FULL trailing width [k,m) — not just the panel — so the trailing block
                        // A22 is already pre-permuted when the GEMM below runs.
                        Swap_OP.Rows(ref U, k, pivotIndex, k, m);

                        // swap already calculated L rows (multipliers already computed travel too)
                        Swap_OP.Rows(ref L, k, pivotIndex, 0, k);

                        fProxy Ukk = U[k, k];
                        fProxy* rowK = up + (long)k * m;
                        int len = panelEnd - (k + 1);
                        for (int j = k + 1; j < m; j++) {

                            fProxy Ljk = U[j, k] / Ukk;

                            L[j, k] = Ljk;

                            if (len > 0) {
                                fProxy* rowJ = up + (long)j * m;
                                Unsafe_OP.axpy(rowJ + (k + 1), rowK + (k + 1), -Ljk, len);
                            }

                            // U is exactly upper-triangular
                            U[j, k] = 0;
                        }
                    }

                    int rStart = panelEnd;
                    if (rStart < m) {
                        int ntrail = m - rStart;

                        // (2) TRSM: U12 = L11^-1 * A12 in place in U[k0:rStart, rStart:m]. L11 =
                        //     L[k0:rStart, k0:rStart] is unit-lower (implicit diagonal = 1), so this is
                        //     forward substitution with no divide — row r of U12 is corrected by the
                        //     already-solved rows 0..r-1 of U12, scaled by L11's strict-lower entries.
                        for (int r = 1; r < kb; r++) {
                            fProxy* uR = up + (long)(k0 + r) * m + rStart;
                            for (int p = 0; p < r; p++) {
                                fProxy Lrp = L[k0 + r, k0 + p];
                                fProxy* uP = up + (long)(k0 + p) * m + rStart;
                                Unsafe_OP.axpy(uR, uP, -Lrp, ntrail);
                            }
                        }

                        // (3) copy U12 (strided, leading dim m) into contiguous Ubuf (kb x ntrail,
                        //     leading dim ntrail) — wySubVW's W operand must be contiguous.
                        for (int r = 0; r < kb; r++) {
                            fProxy* uR = up + (long)(k0 + r) * m + rStart;
                            fProxy* bufR = ubufp + (long)r * ntrail;
                            for (int c = 0; c < ntrail; c++)
                                bufR[c] = uR[c];
                        }

                        // (4) GEMM trailing update: A22 -= L21 * U12, one level-3 call per panel
                        //     instead of kb rank-1 axpy passes over the trailing block. L21 =
                        //     L[rStart:m, k0:rStart] (ntrail x kb, strided leading dim m), A22 =
                        //     U[rStart:m, rStart:m] (strided leading dim m). [NoAlias] is truthful:
                        //     Ubuf is a separate Temp buffer; L21 (in L) and A22 (in U) are different
                        //     matrices.
                        Unsafe_OP.wySubVW(lp + (long)rStart * m + k0, m, up + (long)rStart * m + rStart, m, ntrail, kb, ntrail, ubufp);
                    }
                }

                Ubuf.Dispose();
            }

            // Check last diagonal (mirrors the unblocked form's final check; the blocked k-loop above
            // never pivot-searches column m-1, matching the unblocked k<m-1 bound).
            if (U[m - 1, m - 1] == 0)
                return false;

            return true;
        }

        // A = LU
        // LU is originally A
        // P is pivot, that is reset, and is modified in place
        /// <summary>
        /// Performs LU decomposition inplace with partial pivoting (compact LU form).
        /// Factor row i lives at physical row P[i].
        /// Returns true on success; false if a zero pivot is encountered (singular matrix).
        /// On false: no NaN/Inf is written, P remains a valid permutation.
        /// </summary>
        public static bool luDecompositionInpl(ref fProxyMxN LU, ref Pivot P) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("luDecomposition: LU (A) needs to be square");

            int m = LU.M_Rows;

            if (P.N != m) throw new System.ArgumentException("pivot size must equal matrix dimension");

            P.Reset();

            if (m == 0) return true;

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                fProxy pivotValue = math.abs(LU[P[k], k]);

                // Find largest pivot in rows
                for (int r = k + 1; r < m; r++) {
                    fProxy absValue = math.abs(LU[P[r], k]);
                    if (absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Check for zero pivot before any division
                if (pivotValue == 0)
                    return false;

                // Swap rows
                P.Swap(k, pivotIndex);

                int Pk = P[k];

                // Calculate L and U. Same vectorised axpy elimination as luDecomposition, but on the
                // physical (pivot-indirected) rows Pj, Pk — still distinct (Pj != Pk), so [NoAlias]
                // holds. Bitwise identical to the scalar form.
                fProxy Ukk = LU[Pk, k];
                unsafe
                {
                    fProxy* lup = LU.Data.Ptr;
                    fProxy* rowPk = lup + (long)Pk * m;
                    int len = m - (k + 1);
                    for (int j = k + 1; j < m; j++) {

                        int Pj = P[j];

                        fProxy Ljk = LU[Pj, k] / Ukk;

                        fProxy* rowPj = lup + (long)Pj * m;
                        Unsafe_OP.axpy(rowPj + (k + 1), rowPk + (k + 1), -Ljk, len);

                        LU[Pj, k] = Ljk;
                    }
                }
            }

            // Check last diagonal (the k < m-1 loop never inspects it)
            if (LU[P[m - 1], m - 1] == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Solve LUx = b for x using the compact inplace LU form with pivot.
        /// b is overwritten with x.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        public static void luSolve(ref fProxyMxN LU, in Pivot pivot, ref fProxyN b) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("luSolve: LU must be square");

            if (b.N != LU.M_Rows)
                throw new System.ArgumentException("luSolve: b.N must equal LU.M_Rows");

            if (pivot.N != b.N)
                throw new System.ArgumentException("luSolve: pivot.N must equal b.N");

            pivot.ApplyInverseVec(ref b);

            // Solve Ly = b
            Solvers.solveLowerTriangularLU(ref LU, in pivot, ref b);
            // Solve Ux = y
            Solvers.solveUpperTriangularLU(ref LU, in pivot, ref b);

        }

        /// <summary>
        /// Solve LUx = Pb for x using separate L and U matrices with pivot.
        /// b is overwritten with x.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        public static void luSolve(ref fProxyMxN L, ref fProxyMxN U, in Pivot pivot, ref fProxyN b) {

            if (!U.IsSquare)
                throw new System.ArgumentException("luSolve: U must be square");

            if (b.N != U.M_Rows)
                throw new System.ArgumentException("luSolve: b.N must equal U.M_Rows");

            if (pivot.N != b.N)
                throw new System.ArgumentException("luSolve: pivot.N must equal b.N");

            // apply pivot to b
            pivot.ApplyInverseVec(ref b);

            // Solver linear system LUx = b, b is overwritten with x

            // Solve Ly = b
            Solvers.solveLowerTriangular(ref L, ref b);
            // Solve Ux = y
            Solvers.solveUpperTriangular(ref U, ref b);

        }

        /// <summary>
        /// Compute the determinant from the compact inplace LU form with pivot.
        /// Returns P.Sign * product of diagonal elements LU[P[i], i].
        /// Throws ArgumentException if LU is not square or P.N != LU.M_Rows.
        /// </summary>
        public static fProxy determinant(in fProxyMxN LU, in Pivot P) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("determinant: LU must be square");

            if (P.N != LU.M_Rows)
                throw new System.ArgumentException("determinant: P.N must equal LU.M_Rows");

            int m = LU.M_Rows;
            fProxy det = P.Sign;

            for (int i = 0; i < m; i++)
                det *= LU[P[i], i];

            return det;
        }
    }
}
