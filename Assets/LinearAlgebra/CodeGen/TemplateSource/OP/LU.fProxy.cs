#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class LU {

        /// <summary>
        /// LU decomposition with no pivoting, preserving A: A is copied into U (one memcpy), then
        /// factored in place. A = L * U.
        /// U (caller-allocated, same dimensions as A) receives the upper-triangular factor.
        /// L must be caller-initialized to the identity matrix (same dimensions); it is overwritten
        /// below the diagonal with the lower-triangular factor.
        /// Returns Success; Singular if a zero pivot is encountered (singular matrix).
        /// On Singular: no NaN/Inf is written.
        /// </summary>
        /// <param name="U">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the upper-triangular factor.</param>
        public static DirectSolveInfo decompNoPivot(in fProxyMxN A, ref fProxyMxN L, ref fProxyMxN U)
        {
            if (!A.IsSquare)
                throw new System.ArgumentException("decompNoPivot: A needs to be square");

            if (!L.IsSquare)
                throw new System.ArgumentException("decompNoPivot: L needs to be square");

            if (!U.IsSquare)
                throw new System.ArgumentException("decompNoPivot: U needs to be square");

            if (A.M_Rows != L.M_Rows || A.M_Rows != U.M_Rows)
                throw new System.ArgumentException("decompNoPivot: A, L and U need to have the same dimensions");

            U.Data.CopyFrom(A.Data);

            int m = U.M_Rows;

            if (m == 0) return new DirectSolveInfo { status = DirectSolveStatus.Success };

            for(int k = 0; k < m - 1; k++) {

                // Calculate L and U
                fProxy Ukk = U[k, k];

                if (Ukk == 0)
                    return new DirectSolveInfo { status = DirectSolveStatus.Singular };

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
                return new DirectSolveInfo { status = DirectSolveStatus.Singular };

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // PA = L * U (U starts as a copy of A, L initially I, P reset and modified in place)
        /// <summary>
        /// Performs LU decomposition with partial pivoting, preserving A: A is copied into U (one
        /// memcpy), then factored in place. L must be caller-initialized to the identity matrix.
        /// Returns Success; Singular if a zero pivot is encountered (singular matrix).
        /// On Singular: no NaN/Inf is written, P remains a valid permutation.
        /// Arity-distinct from the compact decompInPlace(ref A_to_LU, ref Pivot) overload below.
        /// </summary>
        /// <param name="U">Output only; prior contents ignored; safe to allocate with uninit: true. Receives the upper-triangular factor.</param>
        public static DirectSolveInfo decomp(in fProxyMxN A, ref fProxyMxN L, ref fProxyMxN U, ref Pivot P) {
            if (!A.IsSquare)
                throw new System.ArgumentException("decomp: A needs to be square");

            if (!L.IsSquare)
                throw new System.ArgumentException("decomp: L needs to be square");

            if (!U.IsSquare)
                throw new System.ArgumentException("decomp: U needs to be square");

            if (A.M_Rows != L.M_Rows || A.M_Rows != U.M_Rows)
                throw new System.ArgumentException("decomp: A, L and U need to have the same dimensions");

            U.Data.CopyFrom(A.Data);

            int m = U.M_Rows;

            if (P.N != m) throw new System.ArgumentException("pivot size must equal matrix dimension");

            P.Reset();

            if (m == 0) return new DirectSolveInfo { status = DirectSolveStatus.Success };

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
                        return new DirectSolveInfo { status = DirectSolveStatus.Singular };

                    // Swap rows
                    P.Swap(k, pivotIndex);

                    // swap submatrix U rows
                    Swap.Rows(ref U, k, pivotIndex, k, m);

                    // swap already calculated L rows
                    Swap.Rows(ref L, k, pivotIndex, 0, k);

                    // Calculate L and U. The trailing-row elimination U[j, k+1:] -= Ljk * U[k, k+1:] is
                    // an axpy over two DISTINCT rows (j > k) along the unit-stride column axis; routed
                    // through the vectorising UnsafeOP.axpy ([NoAlias], the GEMM pointer path) so Burst
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
                            UnsafeOP.axpy(rowJ + (k + 1), rowK + (k + 1), -Ljk, len);

                            // U is exactly upper-triangular
                            U[j, k] = 0;
                        }
                    }
                }

                // Check last diagonal
                if (U[m - 1, m - 1] == 0)
                    return new DirectSolveInfo { status = DirectSolveStatus.Singular };

                return new DirectSolveInfo { status = DirectSolveStatus.Success };
            }

            // ---- blocked (level-3) path — LAPACK-style right-looking GETRF ----
            // Each LU_BLOCK-wide panel is factored with the SAME rank-1 sweep as the small-matrix
            // path above (partial pivoting over the FULL remaining column height, so the pivot
            // sequence is bit-identical to the unblocked form — see "why the pivot sequence stays
            // identical" below, and docs/level3-blocking-guide.md recipe B), but its elimination axpy
            // is narrowed to the panel's own columns (DGETF2-style). The panel's contribution to the
            // columns to its right is then applied ONCE per panel as a level-3 TRSM (U12 = L11^-1 *
            // A12, unit-lower forward substitution) followed by a single GEMM trailing update
            // (A22 -= L21*U12, via UnsafeOP.wySubVW) instead of one rank-1 update per panel column —
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
                            return new DirectSolveInfo { status = DirectSolveStatus.Singular };
                        }

                        // Swap rows
                        P.Swap(k, pivotIndex);

                        // swap FULL trailing width [k,m) — not just the panel — so the trailing block
                        // A22 is already pre-permuted when the GEMM below runs.
                        Swap.Rows(ref U, k, pivotIndex, k, m);

                        // swap already calculated L rows (multipliers already computed travel too)
                        Swap.Rows(ref L, k, pivotIndex, 0, k);

                        fProxy Ukk = U[k, k];
                        fProxy* rowK = up + (long)k * m;
                        int len = panelEnd - (k + 1);
                        for (int j = k + 1; j < m; j++) {

                            fProxy Ljk = U[j, k] / Ukk;

                            L[j, k] = Ljk;

                            if (len > 0) {
                                fProxy* rowJ = up + (long)j * m;
                                UnsafeOP.axpy(rowJ + (k + 1), rowK + (k + 1), -Ljk, len);
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
                                UnsafeOP.axpy(uR, uP, -Lrp, ntrail);
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
                        UnsafeOP.wySubVW(lp + (long)rStart * m + k0, m, up + (long)rStart * m + rStart, m, ntrail, kb, ntrail, ubufp);
                    }
                }

                Ubuf.Dispose();
            }

            // Check last diagonal (mirrors the unblocked form's final check; the blocked k-loop above
            // never pivot-searches column m-1, matching the unblocked k<m-1 bound).
            if (U[m - 1, m - 1] == 0)
                return new DirectSolveInfo { status = DirectSolveStatus.Singular };

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        // A = LU (A_to_LU initially A, P reset and modified in place)
        /// <summary>
        /// Performs LU decomposition in-place with partial pivoting (compact LU form).
        /// Factor row i lives at physical row P[i].
        /// Returns Success; Singular if a zero pivot is encountered (singular matrix).
        /// On Singular: no NaN/Inf is written, P remains a valid permutation.
        /// </summary>
        /// <param name="A_to_LU">On entry A; on exit the compact LU factor (L below the diagonal, U on/above it).</param>
        public static DirectSolveInfo decompInPlace(ref fProxyMxN A_to_LU, ref Pivot P) {

            if (!A_to_LU.IsSquare)
                throw new System.ArgumentException("decompInPlace: A_to_LU needs to be square");

            int m = A_to_LU.M_Rows;

            if (P.N != m) throw new System.ArgumentException("pivot size must equal matrix dimension");

            P.Reset();

            if (m == 0) return new DirectSolveInfo { status = DirectSolveStatus.Success };

            for (int k = 0; k < m - 1; k++) {

                int pivotIndex = k;
                fProxy pivotValue = math.abs(A_to_LU[P[k], k]);

                // Find largest pivot in rows
                for (int r = k + 1; r < m; r++) {
                    fProxy absValue = math.abs(A_to_LU[P[r], k]);
                    if (absValue > pivotValue) {
                        pivotIndex = r;
                        pivotValue = absValue;
                    }
                }

                // Check for zero pivot before any division
                if (pivotValue == 0)
                    return new DirectSolveInfo { status = DirectSolveStatus.Singular };

                // Swap rows
                P.Swap(k, pivotIndex);

                int Pk = P[k];

                // Calculate L and U. Same vectorised axpy elimination as decomp(in A, ref L, ref U, ...),
                // but on the physical (pivot-indirected) rows Pj, Pk — still distinct (Pj != Pk), so
                // [NoAlias] holds. Bitwise identical to the scalar form.
                fProxy Ukk = A_to_LU[Pk, k];
                unsafe
                {
                    fProxy* lup = A_to_LU.Data.Ptr;
                    fProxy* rowPk = lup + (long)Pk * m;
                    int len = m - (k + 1);
                    for (int j = k + 1; j < m; j++) {

                        int Pj = P[j];

                        fProxy Ljk = A_to_LU[Pj, k] / Ukk;

                        fProxy* rowPj = lup + (long)Pj * m;
                        UnsafeOP.axpy(rowPj + (k + 1), rowPk + (k + 1), -Ljk, len);

                        A_to_LU[Pj, k] = Ljk;
                    }
                }
            }

            // Check last diagonal (the k < m-1 loop never inspects it)
            if (A_to_LU[P[m - 1], m - 1] == 0)
                return new DirectSolveInfo { status = DirectSolveStatus.Singular };

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Solve LUx = b for x using the compact in-place LU form with pivot.
        /// b is overwritten with x. Always reports DirectSolveStatus.Success — this assumes a
        /// valid factor from a decompInPlace(ref A_to_LU, ref Pivot) that returned Success; it does
        /// not re-verify it.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo decompSolve(ref fProxyMxN LU, in Pivot pivot, ref fProxyN b_to_x) {

            if (!LU.IsSquare)
                throw new System.ArgumentException("decompSolve: LU must be square");

            if (b_to_x.N != LU.M_Rows)
                throw new System.ArgumentException("decompSolve: b_to_x.N must equal LU.M_Rows");

            if (pivot.N != b_to_x.N)
                throw new System.ArgumentException("decompSolve: pivot.N must equal b_to_x.N");

            pivot.ApplyInverseVec(ref b_to_x);

            // Solve Ly = b
            Solvers.triLowerLU(ref LU, in pivot, ref b_to_x);
            // Solve Ux = y
            Solvers.triUpperLU(ref LU, in pivot, ref b_to_x);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Solve LUx = Pb for x using separate L and U matrices with pivot.
        /// b is overwritten with x. Always reports DirectSolveStatus.Success — this assumes a
        /// valid factor from a decomp(in A, ref L, ref U, ref Pivot) that returned Success;
        /// it does not re-verify it.
        /// Throws ArgumentException if dimensions are inconsistent.
        /// </summary>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo decompSolve(ref fProxyMxN L, ref fProxyMxN U, in Pivot pivot, ref fProxyN b_to_x) {

            if (!U.IsSquare)
                throw new System.ArgumentException("decompSolve: U must be square");

            if (b_to_x.N != U.M_Rows)
                throw new System.ArgumentException("decompSolve: b_to_x.N must equal U.M_Rows");

            if (pivot.N != b_to_x.N)
                throw new System.ArgumentException("decompSolve: pivot.N must equal b_to_x.N");

            // apply pivot to b
            pivot.ApplyInverseVec(ref b_to_x);

            // Solver linear system LUx = b, b is overwritten with x

            // Solve Ly = b
            Solvers.triLower(ref L, ref b_to_x);
            // Solve Ux = y
            Solvers.triUpper(ref U, ref b_to_x);

            return new DirectSolveInfo { status = DirectSolveStatus.Success };
        }

        /// <summary>
        /// Compute the determinant from the compact in-place LU form with pivot.
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

        /// <summary>
        /// Factor-and-solve LUx = b in one call (GESV): factors A in place (compact LU form, partial
        /// pivoting) then solves for x. b is overwritten with x. Returns Singular (forwarded from the
        /// factorization) WITHOUT solving if A is singular.
        /// A_to_LU holds the compact LU factorization on return; valid input to decompSolve for
        /// solving additional right-hand sides without refactoring.
        /// </summary>
        /// <param name="A_to_LU">On entry A; on exit the compact LU factor (L below the diagonal, U on/above it).</param>
        /// <param name="b_to_x">On entry b; on exit the solution x.</param>
        public static DirectSolveInfo solveInPlace(ref fProxyMxN A_to_LU, ref Pivot P, ref fProxyN b_to_x) {
            var info = decompInPlace(ref A_to_LU, ref P);
            if (!info.Solved)
                return info;

            decompSolve(ref A_to_LU, in P, ref b_to_x);
            return info;
        }
    }
}
