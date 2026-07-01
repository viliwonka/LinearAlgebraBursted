using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra.Internal
{
    public static unsafe partial class Unsafe_OP
    {
        // General BR x BC block-CSR matvec: y = A * x. y must already be zeroed by the caller
        // (accumulates into y, like matVecDot). Correctness-first fallback -- every block size
        // routes through this same general loop today.
        // TODO Phase-7 perf: unrolled register-tile specializations for BR,BC in {1,2,3,4,6}
        // (switch-dispatch on (BR,BC) here; reuse the register-tile GEMM lessons).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsmMatVec([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
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
        // the caller. Walks the SAME row-major block-CSR storage as bsmMatVec (no separate
        // transposed copy) -- each stored block K contributes y[j-block] += K^T * x[i-block].
        // Safe single-threaded (no scatter race): every (br,k) pair touches a distinct block,
        // and different blocks in the same block-row write to DIFFERENT y[j-block] ranges only
        // when ColInd is duplicate-free per row, which ToBSM guarantees.
        // TODO Phase-7 perf: unrolled register-tile specializations for BR,BC in {1,2,3,4,6}.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsmMatVecT([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
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
        //     given, no packing/half-storage assumption; see doubleBSM.Symmetric doc)
        //   - off-diagonal (bi<bj, guaranteed since only the upper triangle is stored): y_i += K * x_j
        //     AND y_j += K^T * x_i (the implicit mirrored lower block)
        // Single-threaded caller (IJob.Run, no parallel-for) -> the y_j scatter write from an
        // off-diagonal block is race-free, matching every other kernel in this file. Correctness-first
        // general kernel (BR==BC by construction, no size specialization yet) -- same perf-follow-up
        // note as bsmMatVec/bsmMatVecT (Phase-7 register-tile specialization is future work, not this
        // change).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsmMatVecSym([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] double* values,
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
    }
}
