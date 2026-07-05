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
        public static void bsrMatVec([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                      [NoAlias] float* x, [NoAlias] float* y,
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
                    float* block = values + k * blockLen;

                    for (int r = 0; r < BR; r++)
                    {
                        float sum = 0;
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
        public static void bsrMatVecT([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                       [NoAlias] float* x, [NoAlias] float* y,
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
                    float* block = values + k * blockLen;

                    for (int c = 0; c < BC; c++)
                    {
                        float sum = 0;
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
        //     given, no packing/half-storage assumption; see floatBSR.Symmetric doc)
        //   - off-diagonal (bi<bj, guaranteed since only the upper triangle is stored): y_i += K * x_j
        //     AND y_j += K^T * x_i (the implicit mirrored lower block)
        // Single-threaded caller (IJob.Run, no parallel-for) -> the y_j scatter write from an
        // off-diagonal block is race-free, matching every other kernel in this file. Correctness-
        // first fallback -- BSR.spMV routes here for symmetric matrices whose BR is not one
        // of the register-tile specializations below (bsrMatVecSymB1/B2/B3/B4/B6).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSym([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                         [NoAlias] float* x, [NoAlias] float* y,
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
                    float* block = values + k * blockLen;

                    // y_i += K * x_j  (always -- diagonal or off-diagonal)
                    for (int r = 0; r < BR; r++)
                    {
                        float sum = 0;
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
                            float sum = 0;
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
        // (`float sum = 0; sum += p0; sum += p1; ...`), which is a left-to-right fold; since
        // `0 + p0 == p0` exactly in IEEE754 for any finite p0, that fold is arithmetically
        // identical to the left-associative expression `p0 + p1 + p2 + ...` used below, and both
        // then do a single `y[...] += sum` per (block, row/col). Dispatch lives in
        // BSR.spMV / spMVT (SparseOP.float.cs).
        // =====================================================================================

        // ---- bsrMatVec: y = A * x, square block b -----------------------------------------

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                        [NoAlias] float* x, [NoAlias] float* y, int blockRows)
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
        public static void bsrMatVecB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                        [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 2;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 2;
                    float* block = values + k * 4;
                    float x0 = x[xBase + 0];
                    float x1 = x[xBase + 1];

                    y[yBase + 0] += block[0] * x0 + block[1] * x1;
                    y[yBase + 1] += block[2] * x0 + block[3] * x1;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                        [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 3;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 3;
                    float* block = values + k * 9;
                    float x0 = x[xBase + 0];
                    float x1 = x[xBase + 1];
                    float x2 = x[xBase + 2];

                    y[yBase + 0] += block[0] * x0 + block[1] * x1 + block[2] * x2;
                    y[yBase + 1] += block[3] * x0 + block[4] * x1 + block[5] * x2;
                    y[yBase + 2] += block[6] * x0 + block[7] * x1 + block[8] * x2;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                        [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 4;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 4;
                    float* block = values + k * 16;
                    float x0 = x[xBase + 0];
                    float x1 = x[xBase + 1];
                    float x2 = x[xBase + 2];
                    float x3 = x[xBase + 3];

                    y[yBase + 0] += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3;
                    y[yBase + 1] += block[4]  * x0 + block[5]  * x1 + block[6]  * x2 + block[7]  * x3;
                    y[yBase + 2] += block[8]  * x0 + block[9]  * x1 + block[10] * x2 + block[11] * x3;
                    y[yBase + 3] += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                        [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 6;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int xBase = colInd[k] * 6;
                    float* block = values + k * 36;
                    float x0 = x[xBase + 0];
                    float x1 = x[xBase + 1];
                    float x2 = x[xBase + 2];
                    float x3 = x[xBase + 3];
                    float x4 = x[xBase + 4];
                    float x5 = x[xBase + 5];

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
        public static void bsrMatVecTB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                         [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                float xv = x[br];

                for (int k = rowStart; k < rowEnd; k++)
                    y[colInd[k]] += values[k] * xv;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                         [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 2;
                float x0 = x[xBase + 0];
                float x1 = x[xBase + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 2;
                    float* block = values + k * 4;

                    y[yBase + 0] += block[0] * x0 + block[2] * x1;
                    y[yBase + 1] += block[1] * x0 + block[3] * x1;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                         [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 3;
                float x0 = x[xBase + 0];
                float x1 = x[xBase + 1];
                float x2 = x[xBase + 2];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 3;
                    float* block = values + k * 9;

                    y[yBase + 0] += block[0] * x0 + block[3] * x1 + block[6] * x2;
                    y[yBase + 1] += block[1] * x0 + block[4] * x1 + block[7] * x2;
                    y[yBase + 2] += block[2] * x0 + block[5] * x1 + block[8] * x2;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                         [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 4;
                float x0 = x[xBase + 0];
                float x1 = x[xBase + 1];
                float x2 = x[xBase + 2];
                float x3 = x[xBase + 3];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 4;
                    float* block = values + k * 16;

                    y[yBase + 0] += block[0] * x0 + block[4]  * x1 + block[8]  * x2 + block[12] * x3;
                    y[yBase + 1] += block[1] * x0 + block[5]  * x1 + block[9]  * x2 + block[13] * x3;
                    y[yBase + 2] += block[2] * x0 + block[6]  * x1 + block[10] * x2 + block[14] * x3;
                    y[yBase + 3] += block[3] * x0 + block[7]  * x1 + block[11] * x2 + block[15] * x3;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                         [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 6;
                float x0 = x[xBase + 0];
                float x1 = x[xBase + 1];
                float x2 = x[xBase + 2];
                float x3 = x[xBase + 3];
                float x4 = x[xBase + 4];
                float x5 = x[xBase + 5];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int yBase = colInd[k] * 6;
                    float* block = values + k * 36;

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
        public static void bsrMatVecSymB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                           [NoAlias] float* x, [NoAlias] float* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    float v = values[k];

                    y[bi] += v * x[bj];
                    if (bi != bj)
                        y[bj] += v * x[bi];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                           [NoAlias] float* x, [NoAlias] float* y, int blockRows)
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
                    float* block = values + k * 4;

                    float xj0 = x[xBaseJ + 0];
                    float xj1 = x[xBaseJ + 1];

                    y[yBaseI + 0] += block[0] * xj0 + block[1] * xj1;
                    y[yBaseI + 1] += block[2] * xj0 + block[3] * xj1;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 2;
                        int xBaseI = bi * 2;
                        float xi0 = x[xBaseI + 0];
                        float xi1 = x[xBaseI + 1];

                        y[yBaseJ + 0] += block[0] * xi0 + block[2] * xi1;
                        y[yBaseJ + 1] += block[1] * xi0 + block[3] * xi1;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                           [NoAlias] float* x, [NoAlias] float* y, int blockRows)
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
                    float* block = values + k * 9;

                    float xj0 = x[xBaseJ + 0];
                    float xj1 = x[xBaseJ + 1];
                    float xj2 = x[xBaseJ + 2];

                    y[yBaseI + 0] += block[0] * xj0 + block[1] * xj1 + block[2] * xj2;
                    y[yBaseI + 1] += block[3] * xj0 + block[4] * xj1 + block[5] * xj2;
                    y[yBaseI + 2] += block[6] * xj0 + block[7] * xj1 + block[8] * xj2;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 3;
                        int xBaseI = bi * 3;
                        float xi0 = x[xBaseI + 0];
                        float xi1 = x[xBaseI + 1];
                        float xi2 = x[xBaseI + 2];

                        y[yBaseJ + 0] += block[0] * xi0 + block[3] * xi1 + block[6] * xi2;
                        y[yBaseJ + 1] += block[1] * xi0 + block[4] * xi1 + block[7] * xi2;
                        y[yBaseJ + 2] += block[2] * xi0 + block[5] * xi1 + block[8] * xi2;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                           [NoAlias] float* x, [NoAlias] float* y, int blockRows)
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
                    float* block = values + k * 16;

                    float xj0 = x[xBaseJ + 0];
                    float xj1 = x[xBaseJ + 1];
                    float xj2 = x[xBaseJ + 2];
                    float xj3 = x[xBaseJ + 3];

                    y[yBaseI + 0] += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3;
                    y[yBaseI + 1] += block[4]  * xj0 + block[5]  * xj1 + block[6]  * xj2 + block[7]  * xj3;
                    y[yBaseI + 2] += block[8]  * xj0 + block[9]  * xj1 + block[10] * xj2 + block[11] * xj3;
                    y[yBaseI + 3] += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3;

                    if (bi != bj)
                    {
                        int yBaseJ = bj * 4;
                        int xBaseI = bi * 4;
                        float xi0 = x[xBaseI + 0];
                        float xi1 = x[xBaseI + 1];
                        float xi2 = x[xBaseI + 2];
                        float xi3 = x[xBaseI + 3];

                        y[yBaseJ + 0] += block[0] * xi0 + block[4]  * xi1 + block[8]  * xi2 + block[12] * xi3;
                        y[yBaseJ + 1] += block[1] * xi0 + block[5]  * xi1 + block[9]  * xi2 + block[13] * xi3;
                        y[yBaseJ + 2] += block[2] * xi0 + block[6]  * xi1 + block[10] * xi2 + block[14] * xi3;
                        y[yBaseJ + 3] += block[3] * xi0 + block[7]  * xi1 + block[11] * xi2 + block[15] * xi3;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                           [NoAlias] float* x, [NoAlias] float* y, int blockRows)
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
                    float* block = values + k * 36;

                    float xj0 = x[xBaseJ + 0];
                    float xj1 = x[xBaseJ + 1];
                    float xj2 = x[xBaseJ + 2];
                    float xj3 = x[xBaseJ + 3];
                    float xj4 = x[xBaseJ + 4];
                    float xj5 = x[xBaseJ + 5];

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
                        float xi0 = x[xBaseI + 0];
                        float xi1 = x[xBaseI + 1];
                        float xi2 = x[xBaseI + 2];
                        float xi3 = x[xBaseI + 3];
                        float xi4 = x[xBaseI + 4];
                        float xi5 = x[xBaseI + 5];

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
}
