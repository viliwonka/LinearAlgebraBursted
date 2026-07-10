using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeOP
    {
        // General BR x BC block-CSR matvec: y = A * x. y must already be zeroed by the caller
        // (accumulates into y, like matVecDot). Correctness-first fallback -- BSR.spMV
        // routes here for rectangular blocks (BR != BC) and any square block size NOT covered
        // by the register-tile specializations below (bsrMatVecB1/B2/B3/B4/B6). BR/BC are
        // ordinary runtime fields here, so Burst cannot unroll/vectorize the inner block-multiply
        // loops -- that is exactly what the specializations below fix for the common square sizes.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVec([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                      [NoAlias] double* x, [NoAlias] double* y,
                                      int blockRows, int BR, int BC)
        {
            int blockLen = BR * BC;

            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = colInd[k];
                    int xBase = bc * BC;
                    double* block = values + k * blockLen;

                    for (int r = 0; r < BR; r++)
                    {
                        double sum = 0;
                        int rowOff = r * BC;
                        for (int c = 0; c < BC; c++)
                            sum += block[rowOff + c] * x[xBase + c];
                        y[yBase + r] += sum;
                    }
                }
            }
        }

        // General BR x BC block-CSR transpose matvec: y = A^T * x. y must already be zeroed by
        // the caller. Walks the SAME row-major block-CSR storage as bsrMatVec (no separate
        // transposed copy) -- each stored block K contributes y[j-block] += K^T * x[i-block].
        // Safe single-threaded (no scatter race): every (br,k) pair touches a distinct block,
        // and different blocks in the same block-row write to DIFFERENT y[j-block] ranges only
        // when ColInd is duplicate-free per row, which ToBSR guarantees.
        // Correctness-first fallback -- BSR.spMVT routes here for rectangular blocks and
        // any square size not covered by bsrMatVecTB1/B2/B3/B4/B6 below.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecT([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                       [NoAlias] double* x, [NoAlias] double* y,
                                       int blockRows, int BR, int BC)
        {
            int blockLen = BR * BC;

            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = colInd[k];
                    int yBase = bc * BC;
                    double* block = values + k * blockLen;

                    for (int c = 0; c < BC; c++)
                    {
                        double sum = 0;
                        for (int r = 0; r < BR; r++)
                            sum += block[r * BC + c] * x[xBase + r];
                        y[yBase + c] += sum;
                    }
                }
            }
        }

        // Symmetric BR x BR block-CSR matvec: y = A * x where A is stored as its UPPER block-
        // triangle only (ColInd >= blockRow for every stored block) and the strictly-lower triangle
        // is IMPLICIT (block (bj,bi) == transpose of stored block (bi,bj)). y must already be zeroed
        // by the caller (accumulates into y). For each stored block K at (bi,bj):
        //   - diagonal (bi==bj): y_i += K * x_j   (once -- K is used as the full BR x BR block as
        //     given, no packing/half-storage assumption; see doubleBSR.Symmetric doc)
        //   - off-diagonal (bi<bj, guaranteed since only the upper triangle is stored): y_i += K * x_j
        //     AND y_j += K^T * x_i (the implicit mirrored lower block)
        // Single-threaded caller (IJob.Run, no parallel-for) -> the y_j scatter write from an
        // off-diagonal block is race-free, matching every other kernel in this file. Correctness-
        // first fallback -- BSR.spMV routes here for symmetric matrices whose BR is not one
        // of the register-tile specializations below (bsrMatVecSymB1/B2/B3/B4/B6).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSym([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* x, [NoAlias] double* y,
                                         int blockRows, int BR)
        {
            int blockLen = BR * BR;

            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * BR;
                    double* block = values + k * blockLen;

                    // y_i += K * x_j  (always -- diagonal or off-diagonal)
                    for (int r = 0; r < BR; r++)
                    {
                        double sum = 0;
                        int rowOff = r * BR;
                        for (int c = 0; c < BR; c++)
                            sum += block[rowOff + c] * x[xBaseJ + c];
                        y[yBaseI + r] += sum;
                    }

                    if (bi != bj)
                    {
                        // y_j += K^T * x_i  (the implicit mirrored lower block)
                        int yBaseJ = bj * BR;
                        int xBaseI = bi * BR;
                        for (int c = 0; c < BR; c++)
                        {
                            double sum = 0;
                            for (int r = 0; r < BR; r++)
                                sum += block[r * BR + c] * x[xBaseI + r];
                            y[yBaseJ + c] += sum;
                        }
                    }
                }
            }
        }

        // =====================================================================================
        // Milestone D: block-size-specialized (fully unrolled) square-block kernels, b in
        // {1,2,3,4,6} (3x3 is the FEM/cloth workhorse). The general kernels above have RUNTIME
        // trip counts for the inner BR x BC loops (BR/BC are ordinary int fields), which Burst
        // cannot unroll or register-allocate. These kernels hardcode b as a literal constant in
        // the method body, so the whole block-multiply is a straight-line sequence of named
        // scalar locals -- Burst can register-allocate and (auto-)vectorize it.
        //
        // Accumulation order is chosen to be BIT-IDENTICAL to the general kernel: the general
        // kernel computes each output row's dot product as a running accumulator seeded at zero
        // (`double sum = 0; sum += p0; sum += p1; ...`), which is a left-to-right fold; since
        // `0 + p0 == p0` exactly in IEEE754 for any finite p0, that fold is arithmetically
        // identical to the left-associative expression `p0 + p1 + p2 + ...` used below, and both
        // then do a single `y[...] += sum` per (block, row/col).
        //
        // History (docs/draft-spec-krylov-optimization.md, R2/R8): R2 introduced a 2-accumulator
        // even/odd pairing here for b=2/3/4/6 (b=1 kept single-chain, A/B'd as a no-win exception)
        // as an ARCHITECTURAL JUDGMENT -- the BR=4 benchmark section was too machine-noisy at the
        // time to attribute a clean win either way. R8 revisited it with a dedicated, repeated
        // (3x) clean-room measurement (BR=4/1.5% fill and the b=1 stencil, both dtypes): pairing
        // showed NO reproducible win for b=4 -- every paired-vs-unpaired difference was smaller
        // than the run-to-run swing measured on the IDENTICAL kernel across repeats (up to ~10%),
        // with no consistent direction for double and a shrinking-to-noise edge for float. REVERTED
        // back to the single left-to-right accumulator fold for b=2/3/4/6, matching b=1's own
        // already-settled finding -- every kernel in this family is bit-identical to the general
        // fallback again. See the spec doc's R2/R8 addendum for the full numbers. (R8 also spiked
        // software prefetch, Common.Prefetch on x[colInd[k+dist]] a few blocks ahead: consistently
        // SLOWER, 8-56%, on every dtype/fill/pairing combination tried -- not shipped, see the same
        // addendum.)
        //
        // Dispatch lives in BSR.spMV / spMVT / spMVDot (SparseOP.double.cs).
        // =====================================================================================

        // ---- bsrMatVec: y = A * x, square block b -----------------------------------------

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];

                for (int k = rowStart; k < rowEnd; k++)
                    y[br] += values[k] * x[colInd[k]];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 2;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 2;
                    double* block = values + k * 4;
                    double x0 = x[xBase + 0];
                    double x1 = x[xBase + 1];

                    y[yBase + 0] += block[0] * x0 + block[1] * x1;
                    y[yBase + 1] += block[2] * x0 + block[3] * x1;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 3;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 3;
                    double* block = values + k * 9;
                    double x0 = x[xBase + 0];
                    double x1 = x[xBase + 1];
                    double x2 = x[xBase + 2];

                    y[yBase + 0] += block[0] * x0 + block[1] * x1 + block[2] * x2;
                    y[yBase + 1] += block[3] * x0 + block[4] * x1 + block[5] * x2;
                    y[yBase + 2] += block[6] * x0 + block[7] * x1 + block[8] * x2;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 4;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 4;
                    double* block = values + k * 16;
                    double x0 = x[xBase + 0];
                    double x1 = x[xBase + 1];
                    double x2 = x[xBase + 2];
                    double x3 = x[xBase + 3];

                    y[yBase + 0] += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3;
                    y[yBase + 1] += block[4]  * x0 + block[5]  * x1 + block[6]  * x2 + block[7]  * x3;
                    y[yBase + 2] += block[8]  * x0 + block[9]  * x1 + block[10] * x2 + block[11] * x3;
                    y[yBase + 3] += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 6;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 6;
                    double* block = values + k * 36;
                    double x0 = x[xBase + 0];
                    double x1 = x[xBase + 1];
                    double x2 = x[xBase + 2];
                    double x3 = x[xBase + 3];
                    double x4 = x[xBase + 4];
                    double x5 = x[xBase + 5];

                    y[yBase + 0] += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3 + block[4]  * x4 + block[5]  * x5;
                    y[yBase + 1] += block[6]  * x0 + block[7]  * x1 + block[8]  * x2 + block[9]  * x3 + block[10] * x4 + block[11] * x5;
                    y[yBase + 2] += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3 + block[16] * x4 + block[17] * x5;
                    y[yBase + 3] += block[18] * x0 + block[19] * x1 + block[20] * x2 + block[21] * x3 + block[22] * x4 + block[23] * x5;
                    y[yBase + 4] += block[24] * x0 + block[25] * x1 + block[26] * x2 + block[27] * x3 + block[28] * x4 + block[29] * x5;
                    y[yBase + 5] += block[30] * x0 + block[31] * x1 + block[32] * x2 + block[33] * x3 + block[34] * x4 + block[35] * x5;
                }
            }
        }

        // ---- bsrMatVecT: y = A^T * x, square block b ---------------------------------------

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                double xv = x[br];

                for (int k = rowStart; k < rowEnd; k++)
                    y[colInd[k]] += values[k] * xv;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 2;
                double x0 = x[xBase + 0];
                double x1 = x[xBase + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 2;
                    double* block = values + k * 4;

                    y[yBase + 0] += block[0] * x0 + block[2] * x1;
                    y[yBase + 1] += block[1] * x0 + block[3] * x1;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 3;
                double x0 = x[xBase + 0];
                double x1 = x[xBase + 1];
                double x2 = x[xBase + 2];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 3;
                    double* block = values + k * 9;

                    y[yBase + 0] += block[0] * x0 + block[3] * x1 + block[6] * x2;
                    y[yBase + 1] += block[1] * x0 + block[4] * x1 + block[7] * x2;
                    y[yBase + 2] += block[2] * x0 + block[5] * x1 + block[8] * x2;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 4;
                double x0 = x[xBase + 0];
                double x1 = x[xBase + 1];
                double x2 = x[xBase + 2];
                double x3 = x[xBase + 3];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 4;
                    double* block = values + k * 16;

                    y[yBase + 0] += block[0] * x0 + block[4]  * x1 + block[8]  * x2 + block[12] * x3;
                    y[yBase + 1] += block[1] * x0 + block[5]  * x1 + block[9]  * x2 + block[13] * x3;
                    y[yBase + 2] += block[2] * x0 + block[6]  * x1 + block[10] * x2 + block[14] * x3;
                    y[yBase + 3] += block[3] * x0 + block[7]  * x1 + block[11] * x2 + block[15] * x3;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 6;
                double x0 = x[xBase + 0];
                double x1 = x[xBase + 1];
                double x2 = x[xBase + 2];
                double x3 = x[xBase + 3];
                double x4 = x[xBase + 4];
                double x5 = x[xBase + 5];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 6;
                    double* block = values + k * 36;

                    y[yBase + 0] += block[0] * x0 + block[6]  * x1 + block[12] * x2 + block[18] * x3 + block[24] * x4 + block[30] * x5;
                    y[yBase + 1] += block[1] * x0 + block[7]  * x1 + block[13] * x2 + block[19] * x3 + block[25] * x4 + block[31] * x5;
                    y[yBase + 2] += block[2] * x0 + block[8]  * x1 + block[14] * x2 + block[20] * x3 + block[26] * x4 + block[32] * x5;
                    y[yBase + 3] += block[3] * x0 + block[9]  * x1 + block[15] * x2 + block[21] * x3 + block[27] * x4 + block[33] * x5;
                    y[yBase + 4] += block[4] * x0 + block[10] * x1 + block[16] * x2 + block[22] * x3 + block[28] * x4 + block[34] * x5;
                    y[yBase + 5] += block[5] * x0 + block[11] * x1 + block[17] * x2 + block[23] * x3 + block[29] * x4 + block[35] * x5;
                }
            }
        }

        // ---- bsrMatVecSym: y = A * x, symmetric upper-block-triangle storage, square block b --
        //
        // Each specialization is the fused pair of the corresponding bsrMatVecB{b} row-pass
        // (y_i += K * x_j, always) and bsrMatVecTB{b} column-pass (y_j += K^T * x_i, only when
        // bi != bj) applied to the SAME stored block K -- see bsrMatVecSym above for the general
        // version this mirrors.

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    double v = values[k];

                    y[bi] += v * x[bj];
                    if (bi != bj)
                        y[bj] += v * x[bi];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 2;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 2;
                    double* block = values + k * 4;

                    double xj0 = x[xBaseJ + 0];
                    double xj1 = x[xBaseJ + 1];

                    y[yBaseI + 0] += block[0] * xj0 + block[1] * xj1;
                    y[yBaseI + 1] += block[2] * xj0 + block[3] * xj1;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 2;
                        int xBaseI = bi * 2;
                        double xi0 = x[xBaseI + 0];
                        double xi1 = x[xBaseI + 1];

                        y[yBaseJ + 0] += block[0] * xi0 + block[2] * xi1;
                        y[yBaseJ + 1] += block[1] * xi0 + block[3] * xi1;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 3;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 3;
                    double* block = values + k * 9;

                    double xj0 = x[xBaseJ + 0];
                    double xj1 = x[xBaseJ + 1];
                    double xj2 = x[xBaseJ + 2];

                    y[yBaseI + 0] += block[0] * xj0 + block[1] * xj1 + block[2] * xj2;
                    y[yBaseI + 1] += block[3] * xj0 + block[4] * xj1 + block[5] * xj2;
                    y[yBaseI + 2] += block[6] * xj0 + block[7] * xj1 + block[8] * xj2;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 3;
                        int xBaseI = bi * 3;
                        double xi0 = x[xBaseI + 0];
                        double xi1 = x[xBaseI + 1];
                        double xi2 = x[xBaseI + 2];

                        y[yBaseJ + 0] += block[0] * xi0 + block[3] * xi1 + block[6] * xi2;
                        y[yBaseJ + 1] += block[1] * xi0 + block[4] * xi1 + block[7] * xi2;
                        y[yBaseJ + 2] += block[2] * xi0 + block[5] * xi1 + block[8] * xi2;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 4;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 4;
                    double* block = values + k * 16;

                    double xj0 = x[xBaseJ + 0];
                    double xj1 = x[xBaseJ + 1];
                    double xj2 = x[xBaseJ + 2];
                    double xj3 = x[xBaseJ + 3];

                    y[yBaseI + 0] += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3;
                    y[yBaseI + 1] += block[4]  * xj0 + block[5]  * xj1 + block[6]  * xj2 + block[7]  * xj3;
                    y[yBaseI + 2] += block[8]  * xj0 + block[9]  * xj1 + block[10] * xj2 + block[11] * xj3;
                    y[yBaseI + 3] += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 4;
                        int xBaseI = bi * 4;
                        double xi0 = x[xBaseI + 0];
                        double xi1 = x[xBaseI + 1];
                        double xi2 = x[xBaseI + 2];
                        double xi3 = x[xBaseI + 3];

                        y[yBaseJ + 0] += block[0] * xi0 + block[4]  * xi1 + block[8]  * xi2 + block[12] * xi3;
                        y[yBaseJ + 1] += block[1] * xi0 + block[5]  * xi1 + block[9]  * xi2 + block[13] * xi3;
                        y[yBaseJ + 2] += block[2] * xi0 + block[6]  * xi1 + block[10] * xi2 + block[14] * xi3;
                        y[yBaseJ + 3] += block[3] * xi0 + block[7]  * xi1 + block[11] * xi2 + block[15] * xi3;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* x, [NoAlias] double* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 6;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 6;
                    double* block = values + k * 36;

                    double xj0 = x[xBaseJ + 0];
                    double xj1 = x[xBaseJ + 1];
                    double xj2 = x[xBaseJ + 2];
                    double xj3 = x[xBaseJ + 3];
                    double xj4 = x[xBaseJ + 4];
                    double xj5 = x[xBaseJ + 5];

                    y[yBaseI + 0] += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3 + block[4]  * xj4 + block[5]  * xj5;
                    y[yBaseI + 1] += block[6]  * xj0 + block[7]  * xj1 + block[8]  * xj2 + block[9]  * xj3 + block[10] * xj4 + block[11] * xj5;
                    y[yBaseI + 2] += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3 + block[16] * xj4 + block[17] * xj5;
                    y[yBaseI + 3] += block[18] * xj0 + block[19] * xj1 + block[20] * xj2 + block[21] * xj3 + block[22] * xj4 + block[23] * xj5;
                    y[yBaseI + 4] += block[24] * xj0 + block[25] * xj1 + block[26] * xj2 + block[27] * xj3 + block[28] * xj4 + block[29] * xj5;
                    y[yBaseI + 5] += block[30] * xj0 + block[31] * xj1 + block[32] * xj2 + block[33] * xj3 + block[34] * xj4 + block[35] * xj5;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 6;
                        int xBaseI = bi * 6;
                        double xi0 = x[xBaseI + 0];
                        double xi1 = x[xBaseI + 1];
                        double xi2 = x[xBaseI + 2];
                        double xi3 = x[xBaseI + 3];
                        double xi4 = x[xBaseI + 4];
                        double xi5 = x[xBaseI + 5];

                        y[yBaseJ + 0] += block[0] * xi0 + block[6]  * xi1 + block[12] * xi2 + block[18] * xi3 + block[24] * xi4 + block[30] * xi5;
                        y[yBaseJ + 1] += block[1] * xi0 + block[7]  * xi1 + block[13] * xi2 + block[19] * xi3 + block[25] * xi4 + block[31] * xi5;
                        y[yBaseJ + 2] += block[2] * xi0 + block[8]  * xi1 + block[14] * xi2 + block[20] * xi3 + block[26] * xi4 + block[32] * xi5;
                        y[yBaseJ + 3] += block[3] * xi0 + block[9]  * xi1 + block[15] * xi2 + block[21] * xi3 + block[27] * xi4 + block[33] * xi5;
                        y[yBaseJ + 4] += block[4] * xi0 + block[10] * xi1 + block[16] * xi2 + block[22] * xi3 + block[28] * xi4 + block[34] * xi5;
                        y[yBaseJ + 5] += block[5] * xi0 + block[11] * xi1 + block[17] * xi2 + block[23] * xi3 + block[29] * xi4 + block[35] * xi5;
                    }
                }
            }
        }

        // =====================================================================================
        // Krylov R5 (docs/draft-spec-krylov-optimization.md, R5): BSR SpMM -- a real block-
        // multivector kernel for ApplyBlock. AV[rv,:] += A * V[rv,:] for rv in [0,rows), V/AV
        // row-major with row strides ldV/ldAV (row rv lives at V + rv*ldV / AV + rv*ldAV) -- the
        // same K-layout convention Blas.dotRows uses for the dense operator's ApplyBlock. ldV/ldAV
        // differ whenever A is rectangular (Vrows.N_Cols == A.N_Cols, AVrows.N_Cols == A.M_Rows).
        // Every kernel below is its bsrMatVec{...}/bsrMatVecSym{...} counterpart above with an rv
        // loop ADDED around the
        // per-row body (rowStart/rowEnd/xBaseI hoisted OUTSIDE the rv loop -- rv-
        // independent, computed once per block-row): for a FIXED rv this is the exact same
        // left-to-right term order (same tail, same per-k scatter order for the Sym off-diagonal
        // part) as calling the scalar kernel `rows`
        // times -- BIT-IDENTICAL row by row, not just rounding-equivalent (R8, docs/draft-spec-
        // krylov-optimization.md: the scalar kernels' 2-accumulator pairing was reverted for
        // b=2/3/4/6 -- see that section's own comment -- so these SpMM kernels are unpaired to
        // match and preserve this invariant). Streams the BSR row
        // structure ONCE per block-row instead of once per Apply call: no per-call
        // Allocator.Temp churn (the old ApplyBlock allocated two Temp vectors and re-walked
        // rowPtr/colInd `rows` times), and each row's stored blocks are immediately reused across
        // every rv while resident in L1 instead of being evicted by a full separate matrix pass
        // between calls. Dispatch lives in BSR.spMM (SparseOP.double.cs); sole caller is
        // doubleBSROperator.ApplyBlock.
        // =====================================================================================

        // ---- bsrMatMat: AV[rv,:] += A * V[rv,:], general BR x BC block, square-block fallback --

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMat([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                      [NoAlias] double* V, [NoAlias] double* AV,
                                      int blockRows, int BR, int BC, int rows, int ldV, int ldAV)
        {
            int blockLen = BR * BC;

            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = colInd[k];
                    int xBase = bc * BC;
                    double* block = values + (long)k * blockLen;

                    for (int rv = 0; rv < rows; rv++)
                    {
                        double* x = V + (long)rv * ldV;
                        double* y = AV + (long)rv * ldAV;

                        for (int r = 0; r < BR; r++)
                        {
                            double sum = 0;
                            int rowOff = r * BC;
                            for (int c = 0; c < BC; c++)
                                sum += block[rowOff + c] * x[xBase + c];
                            y[yBase + r] += sum;
                        }
                    }
                }
            }
        }

        // b=1 not paired -- mirrors bsrMatVecB1 (same A/B finding applies: trivial per-block work).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                        y[br] += values[k] * x[colInd[k]];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 2;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int xBase = colInd[k] * 2;
                        double* block = values + k * 4;
                        double x0 = x[xBase + 0];
                        double x1 = x[xBase + 1];

                        y[yBase + 0] += block[0] * x0 + block[1] * x1;
                        y[yBase + 1] += block[2] * x0 + block[3] * x1;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 3;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int xBase = colInd[k] * 3;
                        double* block = values + k * 9;
                        double x0 = x[xBase + 0];
                        double x1 = x[xBase + 1];
                        double x2 = x[xBase + 2];

                        y[yBase + 0] += block[0] * x0 + block[1] * x1 + block[2] * x2;
                        y[yBase + 1] += block[3] * x0 + block[4] * x1 + block[5] * x2;
                        y[yBase + 2] += block[6] * x0 + block[7] * x1 + block[8] * x2;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 4;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int xBase = colInd[k] * 4;
                        double* block = values + k * 16;
                        double x0 = x[xBase + 0];
                        double x1 = x[xBase + 1];
                        double x2 = x[xBase + 2];
                        double x3 = x[xBase + 3];

                        y[yBase + 0] += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3;
                        y[yBase + 1] += block[4]  * x0 + block[5]  * x1 + block[6]  * x2 + block[7]  * x3;
                        y[yBase + 2] += block[8]  * x0 + block[9]  * x1 + block[10] * x2 + block[11] * x3;
                        y[yBase + 3] += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                        [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 6;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int xBase = colInd[k] * 6;
                        double* block = values + k * 36;
                        double x0 = x[xBase + 0];
                        double x1 = x[xBase + 1];
                        double x2 = x[xBase + 2];
                        double x3 = x[xBase + 3];
                        double x4 = x[xBase + 4];
                        double x5 = x[xBase + 5];

                        y[yBase + 0] += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3 + block[4]  * x4 + block[5]  * x5;
                        y[yBase + 1] += block[6]  * x0 + block[7]  * x1 + block[8]  * x2 + block[9]  * x3 + block[10] * x4 + block[11] * x5;
                        y[yBase + 2] += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3 + block[16] * x4 + block[17] * x5;
                        y[yBase + 3] += block[18] * x0 + block[19] * x1 + block[20] * x2 + block[21] * x3 + block[22] * x4 + block[23] * x5;
                        y[yBase + 4] += block[24] * x0 + block[25] * x1 + block[26] * x2 + block[27] * x3 + block[28] * x4 + block[29] * x5;
                        y[yBase + 5] += block[30] * x0 + block[31] * x1 + block[32] * x2 + block[33] * x3 + block[34] * x4 + block[35] * x5;
                    }
                }
            }
        }

        // ---- bsrMatMatSym: AV[rv,:] += A * V[rv,:], symmetric upper-block-triangle storage -----

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatSym([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                         [NoAlias] double* V, [NoAlias] double* AV,
                                         int blockRows, int BR, int rows, int ldV, int ldAV)
        {
            int blockLen = BR * BR;

            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * BR;
                int xBaseI = bi * BR;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int bj = colInd[k];
                        int xBaseJ = bj * BR;
                        double* block = values + (long)k * blockLen;

                        for (int r = 0; r < BR; r++)
                        {
                            double sum = 0;
                            int rowOff = r * BR;
                            for (int c = 0; c < BR; c++)
                                sum += block[rowOff + c] * x[xBaseJ + c];
                            y[yBaseI + r] += sum;
                        }

                        if (bi != bj)
                        {
                            int yBaseJ = bj * BR;
                            for (int c = 0; c < BR; c++)
                            {
                                double sum = 0;
                                for (int r = 0; r < BR; r++)
                                    sum += block[r * BR + c] * x[xBaseI + r];
                                y[yBaseJ + c] += sum;
                            }
                        }
                    }
                }
            }
        }

        // b=1 not paired -- mirrors bsrMatVecSymB1.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatSymB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int bj = colInd[k];
                        double v = values[k];

                        y[bi] += v * x[bj];
                        if (bi != bj)
                            y[bj] += v * x[bi];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatSymB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 2;
                int xBaseI = bi * 2;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int bj = colInd[k];
                        int xBaseJ = bj * 2;
                        double* block = values + k * 4;

                        double xj0 = x[xBaseJ + 0];
                        double xj1 = x[xBaseJ + 1];

                        y[yBaseI + 0] += block[0] * xj0 + block[1] * xj1;
                        y[yBaseI + 1] += block[2] * xj0 + block[3] * xj1;

                        if (bi != bj)
                        {
                            int yBaseJ = bj * 2;
                            double xi0 = x[xBaseI + 0];
                            double xi1 = x[xBaseI + 1];

                            y[yBaseJ + 0] += block[0] * xi0 + block[2] * xi1;
                            y[yBaseJ + 1] += block[1] * xi0 + block[3] * xi1;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatSymB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 3;
                int xBaseI = bi * 3;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int bj = colInd[k];
                        int xBaseJ = bj * 3;
                        double* block = values + k * 9;

                        double xj0 = x[xBaseJ + 0];
                        double xj1 = x[xBaseJ + 1];
                        double xj2 = x[xBaseJ + 2];

                        y[yBaseI + 0] += block[0] * xj0 + block[1] * xj1 + block[2] * xj2;
                        y[yBaseI + 1] += block[3] * xj0 + block[4] * xj1 + block[5] * xj2;
                        y[yBaseI + 2] += block[6] * xj0 + block[7] * xj1 + block[8] * xj2;

                        if (bi != bj)
                        {
                            int yBaseJ = bj * 3;
                            double xi0 = x[xBaseI + 0];
                            double xi1 = x[xBaseI + 1];
                            double xi2 = x[xBaseI + 2];

                            y[yBaseJ + 0] += block[0] * xi0 + block[3] * xi1 + block[6] * xi2;
                            y[yBaseJ + 1] += block[1] * xi0 + block[4] * xi1 + block[7] * xi2;
                            y[yBaseJ + 2] += block[2] * xi0 + block[5] * xi1 + block[8] * xi2;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatSymB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 4;
                int xBaseI = bi * 4;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int bj = colInd[k];
                        int xBaseJ = bj * 4;
                        double* block = values + k * 16;

                        double xj0 = x[xBaseJ + 0];
                        double xj1 = x[xBaseJ + 1];
                        double xj2 = x[xBaseJ + 2];
                        double xj3 = x[xBaseJ + 3];

                        y[yBaseI + 0] += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3;
                        y[yBaseI + 1] += block[4]  * xj0 + block[5]  * xj1 + block[6]  * xj2 + block[7]  * xj3;
                        y[yBaseI + 2] += block[8]  * xj0 + block[9]  * xj1 + block[10] * xj2 + block[11] * xj3;
                        y[yBaseI + 3] += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3;

                        if (bi != bj)
                        {
                            int yBaseJ = bj * 4;
                            double xi0 = x[xBaseI + 0];
                            double xi1 = x[xBaseI + 1];
                            double xi2 = x[xBaseI + 2];
                            double xi3 = x[xBaseI + 3];

                            y[yBaseJ + 0] += block[0] * xi0 + block[4]  * xi1 + block[8]  * xi2 + block[12] * xi3;
                            y[yBaseJ + 1] += block[1] * xi0 + block[5]  * xi1 + block[9]  * xi2 + block[13] * xi3;
                            y[yBaseJ + 2] += block[2] * xi0 + block[6]  * xi1 + block[10] * xi2 + block[14] * xi3;
                            y[yBaseJ + 3] += block[3] * xi0 + block[7]  * xi1 + block[11] * xi2 + block[15] * xi3;
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatMatSymB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
                                           [NoAlias] double* V, [NoAlias] double* AV, int blockRows, int rows, int ldV, int ldAV)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 6;
                int xBaseI = bi * 6;

                for (int rv = 0; rv < rows; rv++)
                {
                    double* x = V + (long)rv * ldV;
                    double* y = AV + (long)rv * ldAV;

                    for (int k = rowStart; k < rowEnd; k++)
                    {
                        int bj = colInd[k];
                        int xBaseJ = bj * 6;
                        double* block = values + k * 36;

                        double xj0 = x[xBaseJ + 0];
                        double xj1 = x[xBaseJ + 1];
                        double xj2 = x[xBaseJ + 2];
                        double xj3 = x[xBaseJ + 3];
                        double xj4 = x[xBaseJ + 4];
                        double xj5 = x[xBaseJ + 5];

                        y[yBaseI + 0] += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3 + block[4]  * xj4 + block[5]  * xj5;
                        y[yBaseI + 1] += block[6]  * xj0 + block[7]  * xj1 + block[8]  * xj2 + block[9]  * xj3 + block[10] * xj4 + block[11] * xj5;
                        y[yBaseI + 2] += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3 + block[16] * xj4 + block[17] * xj5;
                        y[yBaseI + 3] += block[18] * xj0 + block[19] * xj1 + block[20] * xj2 + block[21] * xj3 + block[22] * xj4 + block[23] * xj5;
                        y[yBaseI + 4] += block[24] * xj0 + block[25] * xj1 + block[26] * xj2 + block[27] * xj3 + block[28] * xj4 + block[29] * xj5;
                        y[yBaseI + 5] += block[30] * xj0 + block[31] * xj1 + block[32] * xj2 + block[33] * xj3 + block[34] * xj4 + block[35] * xj5;

                        if (bi != bj)
                        {
                            int yBaseJ = bj * 6;
                            double xi0 = x[xBaseI + 0];
                            double xi1 = x[xBaseI + 1];
                            double xi2 = x[xBaseI + 2];
                            double xi3 = x[xBaseI + 3];
                            double xi4 = x[xBaseI + 4];
                            double xi5 = x[xBaseI + 5];

                            y[yBaseJ + 0] += block[0] * xi0 + block[6]  * xi1 + block[12] * xi2 + block[18] * xi3 + block[24] * xi4 + block[30] * xi5;
                            y[yBaseJ + 1] += block[1] * xi0 + block[7]  * xi1 + block[13] * xi2 + block[19] * xi3 + block[25] * xi4 + block[31] * xi5;
                            y[yBaseJ + 2] += block[2] * xi0 + block[8]  * xi1 + block[14] * xi2 + block[20] * xi3 + block[26] * xi4 + block[32] * xi5;
                            y[yBaseJ + 3] += block[3] * xi0 + block[9]  * xi1 + block[15] * xi2 + block[21] * xi3 + block[27] * xi4 + block[33] * xi5;
                            y[yBaseJ + 4] += block[4] * xi0 + block[10] * xi1 + block[16] * xi2 + block[22] * xi3 + block[28] * xi4 + block[34] * xi5;
                            y[yBaseJ + 5] += block[5] * xi0 + block[11] * xi1 + block[17] * xi2 + block[23] * xi3 + block[29] * xi4 + block[35] * xi5;
                        }
                    }
                }
            }
        }

        // Krylov R2's ApplyDot (docs/draft-spec-krylov-optimization.md) does NOT have a fused
        // kernel family here. A "Dot" variant (fold dot(x,y) into the same pass as y=A*x, tried
        // for the full-storage square-block kernels above) was A/B'd against simply composing
        // (spMV then Blas.dot(x,y)) at the b=1 stencil section of LargeSparseBenchmark and lost
        // by a wide, reproducible margin (~45% SLOWER at N=5120/float) -- see BSR.spMVDot's doc
        // comment (SparseOP.double.cs) for the root cause and the full writeup. Composing wins
        // because it reuses the already 2x-double4-SIMD-tuned vecDot kernel instead of a bespoke
        // scalar cross-row fold. Kept out per the spec's own instruction: measure, and if it
        // doesn't win, don't ship it.

        // =====================================================================================
        // Krylov R2, doubleBlockJacobi.Apply specialization (docs/draft-spec-krylov-optimization.md,
        // R2): z = DInv * r, one dense b x b matvec per block-row (DInv holds ONE explicit inverse
        // block per block-row, no stored-block loop like the spMV kernels above -- there is nothing
        // to accumulator-split here, each output is already a single BR-term sum). Mirrors the
        // spMV unroll structure (b hardcoded as a literal so Burst can register-allocate the whole
        // block-multiply) purely for the same reason bsrMatVecB{b} exists: BR is a runtime field in
        // the general loop, so Burst cannot unroll/vectorize it. Left-to-right term order matches
        // the general loop's `sum = 0; sum += ...` fold exactly -> BIT-IDENTICAL to the general
        // fallback, not just rounding-equivalent. Dispatch lives in doubleBlockJacobi.Apply.
        // =====================================================================================

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB1([NoAlias] double* dp, [NoAlias] double* rp, [NoAlias] double* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
                zp[i] = dp[i] * rp[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB2([NoAlias] double* dp, [NoAlias] double* rp, [NoAlias] double* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 2;
                int blockOff = i * 4;
                double r0 = rp[rowBase + 0];
                double r1 = rp[rowBase + 1];
                zp[rowBase + 0] = dp[blockOff + 0] * r0 + dp[blockOff + 1] * r1;
                zp[rowBase + 1] = dp[blockOff + 2] * r0 + dp[blockOff + 3] * r1;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB3([NoAlias] double* dp, [NoAlias] double* rp, [NoAlias] double* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 3;
                int blockOff = i * 9;
                double r0 = rp[rowBase + 0];
                double r1 = rp[rowBase + 1];
                double r2 = rp[rowBase + 2];
                zp[rowBase + 0] = dp[blockOff + 0] * r0 + dp[blockOff + 1] * r1 + dp[blockOff + 2] * r2;
                zp[rowBase + 1] = dp[blockOff + 3] * r0 + dp[blockOff + 4] * r1 + dp[blockOff + 5] * r2;
                zp[rowBase + 2] = dp[blockOff + 6] * r0 + dp[blockOff + 7] * r1 + dp[blockOff + 8] * r2;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB4([NoAlias] double* dp, [NoAlias] double* rp, [NoAlias] double* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 4;
                int blockOff = i * 16;
                double r0 = rp[rowBase + 0];
                double r1 = rp[rowBase + 1];
                double r2 = rp[rowBase + 2];
                double r3 = rp[rowBase + 3];
                zp[rowBase + 0] = dp[blockOff + 0]  * r0 + dp[blockOff + 1]  * r1 + dp[blockOff + 2]  * r2 + dp[blockOff + 3]  * r3;
                zp[rowBase + 1] = dp[blockOff + 4]  * r0 + dp[blockOff + 5]  * r1 + dp[blockOff + 6]  * r2 + dp[blockOff + 7]  * r3;
                zp[rowBase + 2] = dp[blockOff + 8]  * r0 + dp[blockOff + 9]  * r1 + dp[blockOff + 10] * r2 + dp[blockOff + 11] * r3;
                zp[rowBase + 3] = dp[blockOff + 12] * r0 + dp[blockOff + 13] * r1 + dp[blockOff + 14] * r2 + dp[blockOff + 15] * r3;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB6([NoAlias] double* dp, [NoAlias] double* rp, [NoAlias] double* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 6;
                int blockOff = i * 36;
                double r0 = rp[rowBase + 0];
                double r1 = rp[rowBase + 1];
                double r2 = rp[rowBase + 2];
                double r3 = rp[rowBase + 3];
                double r4 = rp[rowBase + 4];
                double r5 = rp[rowBase + 5];
                zp[rowBase + 0] = dp[blockOff + 0]  * r0 + dp[blockOff + 1]  * r1 + dp[blockOff + 2]  * r2 + dp[blockOff + 3]  * r3 + dp[blockOff + 4]  * r4 + dp[blockOff + 5]  * r5;
                zp[rowBase + 1] = dp[blockOff + 6]  * r0 + dp[blockOff + 7]  * r1 + dp[blockOff + 8]  * r2 + dp[blockOff + 9]  * r3 + dp[blockOff + 10] * r4 + dp[blockOff + 11] * r5;
                zp[rowBase + 2] = dp[blockOff + 12] * r0 + dp[blockOff + 13] * r1 + dp[blockOff + 14] * r2 + dp[blockOff + 15] * r3 + dp[blockOff + 16] * r4 + dp[blockOff + 17] * r5;
                zp[rowBase + 3] = dp[blockOff + 18] * r0 + dp[blockOff + 19] * r1 + dp[blockOff + 20] * r2 + dp[blockOff + 21] * r3 + dp[blockOff + 22] * r4 + dp[blockOff + 23] * r5;
                zp[rowBase + 4] = dp[blockOff + 24] * r0 + dp[blockOff + 25] * r1 + dp[blockOff + 26] * r2 + dp[blockOff + 27] * r3 + dp[blockOff + 28] * r4 + dp[blockOff + 29] * r5;
                zp[rowBase + 5] = dp[blockOff + 30] * r0 + dp[blockOff + 31] * r1 + dp[blockOff + 32] * r2 + dp[blockOff + 33] * r3 + dp[blockOff + 34] * r4 + dp[blockOff + 35] * r5;
            }
        }

        // =====================================================================================
        // Krylov R3 (docs/draft-spec-krylov-optimization.md, R3): block forward/back substitution
        // over FULL-storage BSR (Symmetric upper-block-triangle storage is rejected at the
        // BSR.sweepLower/sweepUpper dispatch -- see doubleBSRMirrorToFull for the one-time
        // mirror-to-full path, Q4 ruling). Sequential across block-rows by construction (that IS
        // the math of a triangular solve, not a deficiency -- fine single-threaded, matching the
        // rest of this file). Solves:
        //   sweepLower: (D/diagScale + L) y = r   -- rows in ASCENDING order, L = stored blocks
        //     with ColInd < row (strictly lower); relies on the BSR invariant that a row's stored
        //     blocks are sorted ascending by ColInd to `break` as soon as ColInd >= row.
        //   sweepUpper: (D/diagScale + U) y = r   -- rows in DESCENDING order, U = stored blocks
        //     with ColInd > row (strictly upper); `continue`s past the diagonal/lower entries
        //     (still ascending order, so no break is available at the START of a row).
        // diagScale=1 is the plain (unscaled) forward/backward Gauss-Seidel triangular solve --
        // <see cref="LinearAlgebra.Sparse.BSR.sweepLower(in LinearAlgebra.Sparse.doubleBSR, in
        // LinearAlgebra.Sparse.doubleBlockJacobi, in doubleN, ref doubleN)"/>'s 4-arg overload
        // forwards here with diagScale=1. doubleSSOR.Apply drives both with diagScale=Omega (the
        // (D/omega+L) / (D/omega+U) systems SSOR's derivation needs -- see that struct's own doc
        // comment for the omega algebra).
        // DIAGONAL solved via the existing doubleBlockJacobi explicit block inverses (dInv, the
        // SAME buffer <see cref="blockJacobiApplyB1"/>..B6 read) -- no per-row factorization, no
        // breakdown risk. b in {1,2,3,4,6} dispatches to a fully unrolled b x b kernel (mirrors
        // bsrMatVec's/blockJacobiApply's unroll shape); any other b falls through to the general
        // runtime-BR loop (a small `stackalloc` row accumulator so the general loop stays correct
        // even if y were ever called in place -- see that loop's own comment).
        // Dispatch lives in BSR.sweepLower / BSR.sweepUpper (SparseOP.double.cs).
        // =====================================================================================

        // ---- sweepLower: (D/diagScale + L) y = r, square block b ---------------------------

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepLowerB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                double acc = r[i];
                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j >= i) break;   // ascending ColInd -> no more strictly-lower entries in this row
                    acc -= values[k] * y[j];
                }
                y[i] = diagScale * dInv[i] * acc;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepLowerB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 2;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j >= i) break;
                    int yBase = j * 2;
                    double* block = values + k * 4;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    acc0 -= block[0] * y0 + block[1] * y1;
                    acc1 -= block[2] * y0 + block[3] * y1;
                }

                int blockOff = i * 4;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0] * acc0 + dInv[blockOff + 1] * acc1);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 2] * acc0 + dInv[blockOff + 3] * acc1);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepLowerB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 3;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];
                double acc2 = r[rowBase + 2];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j >= i) break;
                    int yBase = j * 3;
                    double* block = values + k * 9;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    double y2 = y[yBase + 2];
                    acc0 -= block[0] * y0 + block[1] * y1 + block[2] * y2;
                    acc1 -= block[3] * y0 + block[4] * y1 + block[5] * y2;
                    acc2 -= block[6] * y0 + block[7] * y1 + block[8] * y2;
                }

                int blockOff = i * 9;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0] * acc0 + dInv[blockOff + 1] * acc1 + dInv[blockOff + 2] * acc2);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 3] * acc0 + dInv[blockOff + 4] * acc1 + dInv[blockOff + 5] * acc2);
                y[rowBase + 2] = diagScale * (dInv[blockOff + 6] * acc0 + dInv[blockOff + 7] * acc1 + dInv[blockOff + 8] * acc2);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepLowerB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 4;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];
                double acc2 = r[rowBase + 2];
                double acc3 = r[rowBase + 3];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j >= i) break;
                    int yBase = j * 4;
                    double* block = values + k * 16;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    double y2 = y[yBase + 2];
                    double y3 = y[yBase + 3];
                    acc0 -= block[0]  * y0 + block[1]  * y1 + block[2]  * y2 + block[3]  * y3;
                    acc1 -= block[4]  * y0 + block[5]  * y1 + block[6]  * y2 + block[7]  * y3;
                    acc2 -= block[8]  * y0 + block[9]  * y1 + block[10] * y2 + block[11] * y3;
                    acc3 -= block[12] * y0 + block[13] * y1 + block[14] * y2 + block[15] * y3;
                }

                int blockOff = i * 16;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0]  * acc0 + dInv[blockOff + 1]  * acc1 + dInv[blockOff + 2]  * acc2 + dInv[blockOff + 3]  * acc3);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 4]  * acc0 + dInv[blockOff + 5]  * acc1 + dInv[blockOff + 6]  * acc2 + dInv[blockOff + 7]  * acc3);
                y[rowBase + 2] = diagScale * (dInv[blockOff + 8]  * acc0 + dInv[blockOff + 9]  * acc1 + dInv[blockOff + 10] * acc2 + dInv[blockOff + 11] * acc3);
                y[rowBase + 3] = diagScale * (dInv[blockOff + 12] * acc0 + dInv[blockOff + 13] * acc1 + dInv[blockOff + 14] * acc2 + dInv[blockOff + 15] * acc3);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepLowerB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 6;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];
                double acc2 = r[rowBase + 2];
                double acc3 = r[rowBase + 3];
                double acc4 = r[rowBase + 4];
                double acc5 = r[rowBase + 5];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j >= i) break;
                    int yBase = j * 6;
                    double* block = values + k * 36;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    double y2 = y[yBase + 2];
                    double y3 = y[yBase + 3];
                    double y4 = y[yBase + 4];
                    double y5 = y[yBase + 5];
                    acc0 -= block[0]  * y0 + block[1]  * y1 + block[2]  * y2 + block[3]  * y3 + block[4]  * y4 + block[5]  * y5;
                    acc1 -= block[6]  * y0 + block[7]  * y1 + block[8]  * y2 + block[9]  * y3 + block[10] * y4 + block[11] * y5;
                    acc2 -= block[12] * y0 + block[13] * y1 + block[14] * y2 + block[15] * y3 + block[16] * y4 + block[17] * y5;
                    acc3 -= block[18] * y0 + block[19] * y1 + block[20] * y2 + block[21] * y3 + block[22] * y4 + block[23] * y5;
                    acc4 -= block[24] * y0 + block[25] * y1 + block[26] * y2 + block[27] * y3 + block[28] * y4 + block[29] * y5;
                    acc5 -= block[30] * y0 + block[31] * y1 + block[32] * y2 + block[33] * y3 + block[34] * y4 + block[35] * y5;
                }

                int blockOff = i * 36;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0]  * acc0 + dInv[blockOff + 1]  * acc1 + dInv[blockOff + 2]  * acc2 + dInv[blockOff + 3]  * acc3 + dInv[blockOff + 4]  * acc4 + dInv[blockOff + 5]  * acc5);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 6]  * acc0 + dInv[blockOff + 7]  * acc1 + dInv[blockOff + 8]  * acc2 + dInv[blockOff + 9]  * acc3 + dInv[blockOff + 10] * acc4 + dInv[blockOff + 11] * acc5);
                y[rowBase + 2] = diagScale * (dInv[blockOff + 12] * acc0 + dInv[blockOff + 13] * acc1 + dInv[blockOff + 14] * acc2 + dInv[blockOff + 15] * acc3 + dInv[blockOff + 16] * acc4 + dInv[blockOff + 17] * acc5);
                y[rowBase + 3] = diagScale * (dInv[blockOff + 18] * acc0 + dInv[blockOff + 19] * acc1 + dInv[blockOff + 20] * acc2 + dInv[blockOff + 21] * acc3 + dInv[blockOff + 22] * acc4 + dInv[blockOff + 23] * acc5);
                y[rowBase + 4] = diagScale * (dInv[blockOff + 24] * acc0 + dInv[blockOff + 25] * acc1 + dInv[blockOff + 26] * acc2 + dInv[blockOff + 27] * acc3 + dInv[blockOff + 28] * acc4 + dInv[blockOff + 29] * acc5);
                y[rowBase + 5] = diagScale * (dInv[blockOff + 30] * acc0 + dInv[blockOff + 31] * acc1 + dInv[blockOff + 32] * acc2 + dInv[blockOff + 33] * acc3 + dInv[blockOff + 34] * acc4 + dInv[blockOff + 35] * acc5);
            }
        }

        // General runtime-BR fallback (b not in {1,2,3,4,6}). `acc` is a small per-row scratch
        // (stackalloc'd ONCE, reused every row) holding the row's off-diagonal-subtracted residual
        // before the diagonal inverse is applied -- keeps this loop correct even if a future caller
        // ever passed y aliasing r (it does not read y[rowBase+..] after acc is fully formed).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepLower([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                       double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows, int BR)
        {
            int blockLen = BR * BR;
            double* acc = stackalloc double[BR];

            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * BR;
                for (int lr = 0; lr < BR; lr++) acc[lr] = r[rowBase + lr];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j >= i) break;
                    int yBase = j * BR;
                    double* block = values + (long)k * blockLen;
                    for (int lr = 0; lr < BR; lr++)
                    {
                        double sum = 0;
                        for (int lc = 0; lc < BR; lc++)
                            sum += block[lr * BR + lc] * y[yBase + lc];
                        acc[lr] -= sum;
                    }
                }

                int blockOff = i * blockLen;
                for (int lr = 0; lr < BR; lr++)
                {
                    double sum = 0;
                    for (int lc = 0; lc < BR; lc++)
                        sum += dInv[blockOff + lr * BR + lc] * acc[lc];
                    y[rowBase + lr] = diagScale * sum;
                }
            }
        }

        // ---- sweepUpper: (D/diagScale + U) y = r, square block b ---------------------------
        // Rows in DESCENDING order; U = stored blocks with ColInd > row. Ascending ColInd storage
        // means the strictly-upper entries are a SUFFIX of the row, not a prefix -- no `break` at
        // the row start, so each kernel `continue`s past ColInd <= row instead.

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepUpperB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = blockRows - 1; i >= 0; i--)
            {
                double acc = r[i];
                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j <= i) continue;
                    acc -= values[k] * y[j];
                }
                y[i] = diagScale * dInv[i] * acc;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepUpperB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = blockRows - 1; i >= 0; i--)
            {
                int rowBase = i * 2;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j <= i) continue;
                    int yBase = j * 2;
                    double* block = values + k * 4;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    acc0 -= block[0] * y0 + block[1] * y1;
                    acc1 -= block[2] * y0 + block[3] * y1;
                }

                int blockOff = i * 4;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0] * acc0 + dInv[blockOff + 1] * acc1);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 2] * acc0 + dInv[blockOff + 3] * acc1);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepUpperB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = blockRows - 1; i >= 0; i--)
            {
                int rowBase = i * 3;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];
                double acc2 = r[rowBase + 2];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j <= i) continue;
                    int yBase = j * 3;
                    double* block = values + k * 9;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    double y2 = y[yBase + 2];
                    acc0 -= block[0] * y0 + block[1] * y1 + block[2] * y2;
                    acc1 -= block[3] * y0 + block[4] * y1 + block[5] * y2;
                    acc2 -= block[6] * y0 + block[7] * y1 + block[8] * y2;
                }

                int blockOff = i * 9;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0] * acc0 + dInv[blockOff + 1] * acc1 + dInv[blockOff + 2] * acc2);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 3] * acc0 + dInv[blockOff + 4] * acc1 + dInv[blockOff + 5] * acc2);
                y[rowBase + 2] = diagScale * (dInv[blockOff + 6] * acc0 + dInv[blockOff + 7] * acc1 + dInv[blockOff + 8] * acc2);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepUpperB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = blockRows - 1; i >= 0; i--)
            {
                int rowBase = i * 4;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];
                double acc2 = r[rowBase + 2];
                double acc3 = r[rowBase + 3];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j <= i) continue;
                    int yBase = j * 4;
                    double* block = values + k * 16;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    double y2 = y[yBase + 2];
                    double y3 = y[yBase + 3];
                    acc0 -= block[0]  * y0 + block[1]  * y1 + block[2]  * y2 + block[3]  * y3;
                    acc1 -= block[4]  * y0 + block[5]  * y1 + block[6]  * y2 + block[7]  * y3;
                    acc2 -= block[8]  * y0 + block[9]  * y1 + block[10] * y2 + block[11] * y3;
                    acc3 -= block[12] * y0 + block[13] * y1 + block[14] * y2 + block[15] * y3;
                }

                int blockOff = i * 16;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0]  * acc0 + dInv[blockOff + 1]  * acc1 + dInv[blockOff + 2]  * acc2 + dInv[blockOff + 3]  * acc3);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 4]  * acc0 + dInv[blockOff + 5]  * acc1 + dInv[blockOff + 6]  * acc2 + dInv[blockOff + 7]  * acc3);
                y[rowBase + 2] = diagScale * (dInv[blockOff + 8]  * acc0 + dInv[blockOff + 9]  * acc1 + dInv[blockOff + 10] * acc2 + dInv[blockOff + 11] * acc3);
                y[rowBase + 3] = diagScale * (dInv[blockOff + 12] * acc0 + dInv[blockOff + 13] * acc1 + dInv[blockOff + 14] * acc2 + dInv[blockOff + 15] * acc3);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepUpperB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                         double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows)
        {
            for (int i = blockRows - 1; i >= 0; i--)
            {
                int rowBase = i * 6;
                double acc0 = r[rowBase + 0];
                double acc1 = r[rowBase + 1];
                double acc2 = r[rowBase + 2];
                double acc3 = r[rowBase + 3];
                double acc4 = r[rowBase + 4];
                double acc5 = r[rowBase + 5];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j <= i) continue;
                    int yBase = j * 6;
                    double* block = values + k * 36;
                    double y0 = y[yBase + 0];
                    double y1 = y[yBase + 1];
                    double y2 = y[yBase + 2];
                    double y3 = y[yBase + 3];
                    double y4 = y[yBase + 4];
                    double y5 = y[yBase + 5];
                    acc0 -= block[0]  * y0 + block[1]  * y1 + block[2]  * y2 + block[3]  * y3 + block[4]  * y4 + block[5]  * y5;
                    acc1 -= block[6]  * y0 + block[7]  * y1 + block[8]  * y2 + block[9]  * y3 + block[10] * y4 + block[11] * y5;
                    acc2 -= block[12] * y0 + block[13] * y1 + block[14] * y2 + block[15] * y3 + block[16] * y4 + block[17] * y5;
                    acc3 -= block[18] * y0 + block[19] * y1 + block[20] * y2 + block[21] * y3 + block[22] * y4 + block[23] * y5;
                    acc4 -= block[24] * y0 + block[25] * y1 + block[26] * y2 + block[27] * y3 + block[28] * y4 + block[29] * y5;
                    acc5 -= block[30] * y0 + block[31] * y1 + block[32] * y2 + block[33] * y3 + block[34] * y4 + block[35] * y5;
                }

                int blockOff = i * 36;
                y[rowBase + 0] = diagScale * (dInv[blockOff + 0]  * acc0 + dInv[blockOff + 1]  * acc1 + dInv[blockOff + 2]  * acc2 + dInv[blockOff + 3]  * acc3 + dInv[blockOff + 4]  * acc4 + dInv[blockOff + 5]  * acc5);
                y[rowBase + 1] = diagScale * (dInv[blockOff + 6]  * acc0 + dInv[blockOff + 7]  * acc1 + dInv[blockOff + 8]  * acc2 + dInv[blockOff + 9]  * acc3 + dInv[blockOff + 10] * acc4 + dInv[blockOff + 11] * acc5);
                y[rowBase + 2] = diagScale * (dInv[blockOff + 12] * acc0 + dInv[blockOff + 13] * acc1 + dInv[blockOff + 14] * acc2 + dInv[blockOff + 15] * acc3 + dInv[blockOff + 16] * acc4 + dInv[blockOff + 17] * acc5);
                y[rowBase + 3] = diagScale * (dInv[blockOff + 18] * acc0 + dInv[blockOff + 19] * acc1 + dInv[blockOff + 20] * acc2 + dInv[blockOff + 21] * acc3 + dInv[blockOff + 22] * acc4 + dInv[blockOff + 23] * acc5);
                y[rowBase + 4] = diagScale * (dInv[blockOff + 24] * acc0 + dInv[blockOff + 25] * acc1 + dInv[blockOff + 26] * acc2 + dInv[blockOff + 27] * acc3 + dInv[blockOff + 28] * acc4 + dInv[blockOff + 29] * acc5);
                y[rowBase + 5] = diagScale * (dInv[blockOff + 30] * acc0 + dInv[blockOff + 31] * acc1 + dInv[blockOff + 32] * acc2 + dInv[blockOff + 33] * acc3 + dInv[blockOff + 34] * acc4 + dInv[blockOff + 35] * acc5);
            }
        }

        // General runtime-BR fallback (b not in {1,2,3,4,6}). Same acc-scratch reasoning as
        // sweepLower's general fallback above.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sweepUpper([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values, [NoAlias] double* dInv,
                                       double diagScale, [NoAlias] double* r, [NoAlias] double* y, int blockRows, int BR)
        {
            int blockLen = BR * BR;
            double* acc = stackalloc double[BR];

            for (int i = blockRows - 1; i >= 0; i--)
            {
                int rowBase = i * BR;
                for (int lr = 0; lr < BR; lr++) acc[lr] = r[rowBase + lr];

                int rs = rowPtr[i], re = rowPtr[i + 1];
                for (int k = rs; k < re; k++)
                {
                    int j = colInd[k];
                    if (j <= i) continue;
                    int yBase = j * BR;
                    double* block = values + (long)k * blockLen;
                    for (int lr = 0; lr < BR; lr++)
                    {
                        double sum = 0;
                        for (int lc = 0; lc < BR; lc++)
                            sum += block[lr * BR + lc] * y[yBase + lc];
                        acc[lr] -= sum;
                    }
                }

                int blockOff = i * blockLen;
                for (int lr = 0; lr < BR; lr++)
                {
                    double sum = 0;
                    for (int lc = 0; lc < BR; lc++)
                        sum += dInv[blockOff + lr * BR + lc] * acc[lc];
                    y[rowBase + lr] = diagScale * sum;
                }
            }
        }
    }
}
