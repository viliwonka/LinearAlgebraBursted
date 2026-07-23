using System;
using LinearAlgebra;
using Unity.Mathematics;
using Unity.Collections;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Block incomplete-Cholesky IC(0) preconditioner over a square SPD BSR: A ≈ L·Lᵀ where L is
    /// constrained to A's lower block pattern — all fill is discarded (Saad, "Iterative Methods
    /// for Sparse Linear Systems" 2nd ed., ch. 10.3). Apply solves z = (L·Lᵀ)⁻¹ r with one
    /// forward and one backward block-triangular sweep. On a block pattern with no fill (e.g. a
    /// block-tridiagonal chain) IC(0) IS the exact Cholesky factorization.
    ///
    /// Factorization can break down on SPD matrices that are far from diagonally dominant; the
    /// constructor retries with an escalating diagonal shift A + α·I (Manteuffel) and records the
    /// shift that succeeded in <see cref="Shift"/> (0 = clean factorization). Throws if the
    /// largest shift still breaks down — the matrix is likely not SPD.
    ///
    /// Symmetric-storage A needs NO mirror: its stored lower-block pattern (diagonal included) IS
    /// exactly the pattern IC(0) factorizes, so a symmetric-storage A is consumed directly,
    /// zero-copy. Full-storage A is accepted too (only its lower blocks are read). A must store
    /// every diagonal block. This instance owns L standalone and Dispose() must be called when done.
    /// </summary>
    public readonly struct fProxyIC0 : IfProxyPreconditioner, IDisposable
    {
        /// <summary>Lower block pattern of A (diagonal included). Diagonal blocks hold their lower
        /// Cholesky factor (upper halves zeroed); off-diagonal blocks the IC(0) L values.</summary>
        public readonly fProxyBSR L;

        /// <summary>Diagonal shift α that made the factorization succeed; 0 for a clean pass.</summary>
        public readonly fProxy Shift;

        public int Rows => L.M_Rows;

        /// <summary>
        /// Factorizes A's lower block pattern in place of the copy it allocates from
        /// <paramref name="allocator"/>. Throws if A is not square (BlockRows==BlockCols,
        /// BR==BC), if a diagonal block is absent, or if the factorization still breaks down at
        /// the largest diagonal shift. Use the out-info overload to receive the outcome as a
        /// <see cref="PreconditionerInfo"/> instead of an exception. Dispose the result with
        /// <see cref="Dispose"/>.
        /// </summary>
        public unsafe fProxyIC0(in fProxyBSR a, Allocator allocator)
        {
            this = new fProxyIC0(in a, allocator, out PreconditionerInfo info);
            if (!info.Solved)
            {
                Dispose();
                throw new ArgumentException("fProxyIC0: factorization broke down at every diagonal shift — is A symmetric positive definite?");
            }
        }

        /// <summary>Non-throwing build: info carries the outcome (L is allocated and left populated
        /// with the failed factorization even on breakdown -- Dispose is still valid).</summary>
        public unsafe fProxyIC0(in fProxyBSR a, Allocator allocator, out PreconditionerInfo info)
        {
            if (a.BlockRows != a.BlockCols || a.BR != a.BC)
                throw new ArgumentException("fProxyIC0: A must be square (BlockRows==BlockCols, BR==BC)");

            var A = a;

            int nb = A.BlockRows;
            int BR = A.BR;
            int blockLen = BR * BR;

            int nnzbL = 0;
            for (int i = 0; i < nb; i++)
            {
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                bool hasDiag = false;
                for (int k = s; k < e; k++)
                {
                    int col = A.ColInd[k];
                    if (col > i) break;
                    nnzbL++;
                    if (col == i) hasDiag = true;
                }
                if (!hasDiag)
                    throw new ArgumentException("fProxyIC0: missing diagonal block in A");
            }

            var Lm = new fProxyBSR(nb, nb, BR, BR, nnzbL, allocator, true);
            var lRowPtr = Lm.RowPtr; var lColInd = Lm.ColInd;
            {
                int outIdx = 0;
                for (int i = 0; i < nb; i++)
                {
                    lRowPtr[i] = outIdx;
                    int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                    for (int k = s; k < e; k++)
                    {
                        int col = A.ColInd[k];
                        if (col > i) break;
                        lColInd[outIdx] = col;
                        outIdx++;
                    }
                }
                lRowPtr[nb] = outIdx;
            }

            fProxy diagMax = 0;
            for (int i = 0; i < nb; i++)
            {
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                for (int k = s; k < e; k++)
                {
                    if (A.ColInd[k] != i) continue;
                    int off = k * blockLen;
                    for (int r = 0; r < BR; r++)
                    {
                        fProxy av = math.abs(A.Values[off + r * BR + r]);
                        if (av > diagMax) diagMax = av;
                    }
                    break;
                }
            }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;

            fProxy shift = 0;
            bool ok = false;
            int attempts = 0;
            for (int attempt = 0; attempt < 6; attempt++)
            {
                attempts = attempt + 1;
                CopyLowerFromA(in A, in Lm, shift);
                if (FactorizeInPlace(in Lm, diagMax)) { ok = true; break; }
                shift = shift == (fProxy)0 ? (fProxy)1e-3 * diagMax : shift * (fProxy)10;
            }
            L = Lm;
            Shift = shift;
            info = new PreconditionerInfo
            {
                status = ok ? DirectSolveStatus.Success : DirectSolveStatus.NotPositiveDefinite,
                shift = (double)shift,
                attempts = attempts,
            };
        }

        /// <summary>Disposes L.</summary>
        public unsafe void Dispose() => L.Dispose();

        // Refills L's values from A's lower blocks, adding `shift` to the diagonal entries.
        static void CopyLowerFromA(in fProxyBSR A, in fProxyBSR Lm, fProxy shift)
        {
            int nb = A.BlockRows;
            int BR = A.BR;
            int blockLen = BR * BR;

            var lRowPtr = Lm.RowPtr; var lValues = Lm.Values;
            var aRowPtr = A.RowPtr; var aColInd = A.ColInd; var aValues = A.Values;

            for (int i = 0; i < nb; i++)
            {
                int outIdx = lRowPtr[i];
                int s = aRowPtr[i], e = aRowPtr[i + 1];
                for (int k = s; k < e; k++)
                {
                    int col = aColInd[k];
                    if (col > i) break;

                    int srcOff = k * blockLen, dstOff = outIdx * blockLen;
                    for (int t = 0; t < blockLen; t++)
                        lValues[dstOff + t] = aValues[srcOff + t];

                    if (col == i)
                        for (int r = 0; r < BR; r++)
                            lValues[dstOff + r * BR + r] += shift;

                    outIdx++;
                }
            }
        }

        // Up-looking block IC(0) over L's pattern (values pre-loaded with A's lower blocks).
        // Returns false on a non-positive pivot (breakdown).
        static bool FactorizeInPlace(in fProxyBSR Lm, fProxy diagMax)
        {
            int nb = Lm.BlockRows;
            int BR = Lm.BR;
            int blockLen = BR * BR;
            fProxy pivotFloor = (fProxy)16 * Consts.fProxyEpsilon * diagMax;

            var rowPtr = Lm.RowPtr; var colInd = Lm.ColInd; var values = Lm.Values;

            for (int i = 0; i < nb; i++)
            {
                int s = rowPtr[i], e = rowPtr[i + 1];   // last entry in the row is the diagonal

                // Off-diagonal blocks L_ij (ascending j).
                for (int idx = s; idx < e - 1; idx++)
                {
                    int j = colInd[idx];
                    int sOff = idx * blockLen;

                    // S -= sum over k in (row i, cols < j) ∩ (row j, cols < j) of L_ik · L_jkᵀ.
                    int p = s;
                    int q = rowPtr[j];
                    int qEnd = rowPtr[j + 1] - 1;       // exclude row j's diagonal
                    while (p < idx && q < qEnd)
                    {
                        int pc = colInd[p], qc = colInd[q];
                        if (pc < qc) { p++; continue; }
                        if (qc < pc) { q++; continue; }
                        BlockMulSubABt(values, sOff, p * blockLen, q * blockLen, BR);
                        p++; q++;
                    }

                    // S := S · L_jj^{-T}  (row-wise forward substitutions with lower L_jj).
                    int jjOff = (rowPtr[j + 1] - 1) * blockLen;
                    for (int r = 0; r < BR; r++)
                        ForwardSolveRow(values, sOff + r * BR, jjOff, BR);
                }

                // Diagonal block: D = A_ii - sum L_ik · L_ikᵀ, then D = chol(D).
                int dOff = (e - 1) * blockLen;
                for (int idx = s; idx < e - 1; idx++)
                    BlockMulSubABt(values, dOff, idx * blockLen, idx * blockLen, BR);

                if (!CholBlockLower(values, dOff, BR, pivotFloor))
                    return false;
            }

            return true;
        }

        // C -= A · Bᵀ for BR x BR row-major blocks at the given flat offsets.
        static void BlockMulSubABt(Unity.Collections.LowLevel.Unsafe.UnsafeList<fProxy> v, int cOff, int aOff, int bOff, int BR)
        {
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                {
                    fProxy sum = 0;
                    for (int t = 0; t < BR; t++)
                        sum += v[aOff + r * BR + t] * v[bOff + c * BR + t];
                    v[cOff + r * BR + c] -= sum;
                }
        }

        // Solves x · Lᵀ = s for one row s of length BR (equivalently L xᵀ = sᵀ, a forward
        // substitution with the lower factor at ljjOff); overwrites the row in place.
        static void ForwardSolveRow(Unity.Collections.LowLevel.Unsafe.UnsafeList<fProxy> v, int rowOff, int ljjOff, int BR)
        {
            for (int c = 0; c < BR; c++)
            {
                fProxy sum = v[rowOff + c];
                for (int t = 0; t < c; t++)
                    sum -= v[ljjOff + c * BR + t] * v[rowOff + t];
                v[rowOff + c] = sum / v[ljjOff + c * BR + c];
            }
        }

        // In-place lower Cholesky of the BR x BR block at dOff; zeroes the upper half. Returns
        // false if a pivot falls at or below the floor (breakdown).
        static bool CholBlockLower(Unity.Collections.LowLevel.Unsafe.UnsafeList<fProxy> v, int dOff, int BR, fProxy pivotFloor)
        {
            for (int r = 0; r < BR; r++)
            {
                for (int c = 0; c <= r; c++)
                {
                    fProxy sum = v[dOff + r * BR + c];
                    for (int t = 0; t < c; t++)
                        sum -= v[dOff + r * BR + t] * v[dOff + c * BR + t];

                    if (c == r)
                    {
                        if (!(sum > pivotFloor)) return false;
                        v[dOff + r * BR + r] = math.sqrt(sum);
                    }
                    else
                    {
                        v[dOff + r * BR + c] = sum / v[dOff + c * BR + c];
                    }
                }
                for (int c = r + 1; c < BR; c++)
                    v[dOff + r * BR + c] = 0;
            }
            return true;
        }

        /// <summary>z = (L·Lᵀ)⁻¹ r: forward block sweep (L y = r, written into z), then in-place
        /// backward block sweep (Lᵀ z = y). z must not alias r.</summary>
        public bool IsIdentity => false;
        public bool IsSpd => true;
        public bool IsConstant => true;

        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;
            if (r.N != n)
                throw new ArgumentException("fProxyIC0.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("fProxyIC0.Apply: z.N must equal Rows");
            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("fProxyIC0.Apply: z must not alias r");

            int nb = L.BlockRows;
            int BR = L.BR;
            int blockLen = BR * BR;

            var rowPtr = L.RowPtr; var colInd = L.ColInd; var values = L.Values;

            // Forward: z_i = L_ii^{-1} (r_i - sum_{j<i} L_ij z_j), ascending i.
            for (int i = 0; i < nb; i++)
            {
                int s = rowPtr[i], e = rowPtr[i + 1];
                int rowBase = i * BR;

                for (int t = 0; t < BR; t++)
                    z[rowBase + t] = r[rowBase + t];

                for (int idx = s; idx < e - 1; idx++)
                {
                    int jBase = colInd[idx] * BR;
                    int off = idx * blockLen;
                    for (int lr = 0; lr < BR; lr++)
                    {
                        fProxy sum = 0;
                        for (int lc = 0; lc < BR; lc++)
                            sum += values[off + lr * BR + lc] * z[jBase + lc];
                        z[rowBase + lr] -= sum;
                    }
                }

                int dOff = (e - 1) * blockLen;
                for (int lr = 0; lr < BR; lr++)
                {
                    fProxy sum = z[rowBase + lr];
                    for (int lc = 0; lc < lr; lc++)
                        sum -= values[dOff + lr * BR + lc] * z[rowBase + lc];
                    z[rowBase + lr] = sum / values[dOff + lr * BR + lr];
                }
            }

            // Backward: solve Lᵀ z = y in place, descending i, scatter-style.
            for (int i = nb - 1; i >= 0; i--)
            {
                int s = rowPtr[i], e = rowPtr[i + 1];
                int rowBase = i * BR;

                int dOff = (e - 1) * blockLen;
                for (int lr = BR - 1; lr >= 0; lr--)
                {
                    fProxy sum = z[rowBase + lr];
                    for (int lc = lr + 1; lc < BR; lc++)
                        sum -= values[dOff + lc * BR + lr] * z[rowBase + lc];
                    z[rowBase + lr] = sum / values[dOff + lr * BR + lr];
                }

                for (int idx = s; idx < e - 1; idx++)
                {
                    int jBase = colInd[idx] * BR;
                    int off = idx * blockLen;
                    for (int lc = 0; lc < BR; lc++)
                    {
                        fProxy sum = 0;
                        for (int lr = 0; lr < BR; lr++)
                            sum += values[off + lr * BR + lc] * z[rowBase + lr];   // L_ijᵀ z_i
                        z[jBase + lc] -= sum;
                    }
                }
            }
        }
    }
}
